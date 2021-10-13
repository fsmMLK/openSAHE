#!/bin/python
# -*- coding: utf-8 -*-
"""
definition of basic general use functions
"""

from __future__ import division, print_function

import inspect
import os
import shlex
from datetime import datetime
import math
import re

import matplotlib as mpl
import numpy as np
import psutil
from lxml import etree as ETree

try:
    from nptyping import Array
except ImportError:
    pass

from scipy import sparse as scipySparse

def isFeLap():
    """returns True if computer name is felap3"""
    return os.uname()[1] == 'felap3'

def isUkko2():
    return os.uname()[1].startswith('ukko2')

# set number of processors
if isFeLap():
    nCores = max(1,os.cpu_count() - 1)
else:
    nCores = 29

#openBF folder
openBFDir = '~/.julia/packages/openBF/pcekr/'  # do not forget the last / !!!

# relative to src folder
singularityImgPath='../sigularityImages/nipype_ANTS_SPM12_SingularityImg.simg'

def printMemoryUsage(outputFile=None, mode='w',extraMessage=''):
    """
    prints the current memory usage. optionaly, can save the information in a file

    Parameters
    ----------
    outputFile: str
        optional output file. If 'None', no file is saved
    mode: str
        write mode of the output file. valid options: 'w' (write), 'a' (append)

    extraMessage: str
        additional message to print. Optional
    """

    # get caller function, file and line
    # https://stackoverflow.com/questions/900392/getting-the-caller-function-name-inside-another-function-in-python
    stack = inspect.stack()[1]
    fileName = os.path.basename(stack.filename)
    callerFunction = stack.function
    linNumber = stack.lineno

    callerInfo = '%s (function: %s, line: %d)' % (fileName, callerFunction, linNumber)

    # get memory usage in Gb
    process = psutil.Process(os.getpid())
    RSS_Gb = process.memory_info().rss / float(2 ** 30)

    # get current time
    now = getNow()

    # print simple message in stdout
    print('%s: [%s] current memory usage : %f Gb\n' % (fileName, extraMessage, RSS_Gb))

    if outputFile is not None:
        with open(outputFile, mode) as f:
            if mode == 'w':
                f.write('time;caller information;message;memory [Gb]\n')
            f.write('%s;%s;%s;%fGb\n' % (now, callerInfo, extraMessage, RSS_Gb))

def getNow():
    """
    get current date and time in the following format:
            2020-08Aug-31_13-19-04
            yyyy-mmMMM-dd_hh-mm-ss  <--hours in 24h format
    Returns
    -------
        time: string
            current date and time
    """
    return datetime.now().strftime("%Y-%m%b-%d_%H-%M-%S")

# call memory usage

def showModulesVersion():
    print("\n")
    print("======== machine name =========")
    print("Machine: %s" % os.uname()[1])
    print("Using %d cores of a total of %d..." % (nCores, os.cpu_count()))
    print("======== modules Version =========")
    print('Numpy version: ' + np.version.version)
    print('Matplotlib version: ' + mpl.__version__)
    print("==================================")
    print("\n")


def saveSparseCOO(array, fileName):
    """
    stores an array in sparse coo format

    Parameters
    ----------
    array: numpy array
        array to be saved. this matrix can be sparse or dense.

    fileName: str
        name of the file. Each line of the file contais row, column and value of the nonzero elements. ATENTION: row and col numbers start in 1 to
        be compatible with matlab!

    """
    sparse = scipySparse.coo_matrix(array)
    with open(fileName, 'w') as file:
        for i, j, val in zip(sparse.row, sparse.col, sparse.data):
            file.write('%d %d %1.15e\n' % (i + 1, j + 1, val))

def convToMetre(inputVal, inputUnit):
    """
    convert the input (scalar or numpy array) to metres

    Parameters
    ----------
    inputVal: scalar or numpy array
        input values
    inputUnit: string
        unit of the input. Implemented units: 'mm', 'cm', 'm', 'in'
    Returns
    -------
    outputVal: scalar or numpy array
        values converted to metres
    """
    if inputUnit.lower() == 'mm':
        return inputVal * 0.001
    if inputUnit.lower() == 'cm':
        return inputVal * 0.01
    if inputUnit.lower() == 'm':
        return inputVal
    if inputUnit.lower() == 'in':
        return inputVal * 0.0254


def convFromMetre(inputVal, outputUnit):
    """
    convert the input em metres (scalar or numpy array) to output unit

    Parameters
    ----------
    inputVal: scalar or numpy array
        input values
    outputUnit: string
        unit of the output. Implemented units: 'mm', 'cm', 'm', 'in'
    Returns
    -------
    outputVal: scalar or numpy array
        values converted to specified unit

    """
    if outputUnit.lower() == 'mm':
        return inputVal / 0.001
    if outputUnit.lower() == 'cm':
        return inputVal / 0.01
    if outputUnit.lower() == 'm':
        return inputVal
    if outputUnit.lower() == 'in':
        return inputVal / 0.0254

def rotMatrix(axis='x',angle_rad=0.0):
    if axis.lower()=='x':
        R = np.array([[                  1,                  0,                  0],
                      [                  0,  np.cos(angle_rad), -np.sin(angle_rad)],
                      [                  0,  np.sin(angle_rad),  np.cos(angle_rad)]])
    if axis.lower()=='y':
        R = np.array([[  np.cos(angle_rad),                  0,  np.sin(angle_rad)],
                      [                  0,                  1,                  0],
                      [ -np.sin(angle_rad),                  0,  np.cos(angle_rad)]])
    if axis.lower()=='z':
        R = np.array([[  np.cos(angle_rad), -np.sin(angle_rad),                  0],
                      [  np.sin(angle_rad),  np.cos(angle_rad),                  0],
                      [                  0,                  0,                  1]])
    return R

def compareMatrices(numpyData,  # type: Array
                    file,  # type: str
                    label,  # type: str
                    isComplex=False  # type: bool
                    ):
    """
    compare two matrices, one from memory and one stored in a file.

    This function is used to debug the code. The comparison is:
        max( max ( abs(M1-M2) ) )

    Parameters
    ----------
    numpyData: numpy array
        array to be compared

    file: str
        file containing the matrix to be compared. The file must be compatible with numpy.loadtxt function

    label: str
        A label to identify the matrix. This is used to help identifying the result in the output console

    isComplex: bool, optional
        use True if the matrices are complex. Use False otherwise
    """
    if isComplex:
        dataFile = np.loadtxt(file).view(complex)
    else:
        dataFile = np.loadtxt(file)

    # np.testing.assert_allclose(numpyData, dataFile,  atol=1e-14, equal_nan=True, err_msg='differenca em %s' % label , verbose=True)

    print("  -> largest difference (absolute value) of %s : %1.5e" % (label, np.amax(np.absolute(numpyData - dataFile))))


def compareMatricesSparse(numpyArray,  # type: Array
                          file,  # type: str
                          label,  # type: str
                          transposeData=False  # type: bool
                          ):
    """
    compare two matrices, one from memory and one stored in a file. The matrix in the file must be sparse in the form RCV

    This function is used to debug the code. The comparison is:
        max( max ( abs(M1-M2) ) )

    Parameters
    ----------
    numpyArray: numpy array
        array to be compared

    file: str
        file containing the matrix to be compared. each line of the file must contain: row column value of non zero
        elements

    label: str
        A label to identify the matrix. This is used to help identifying the result in the output console

    transposeData: bool, optional
        compare the transposed of numpyData matrix
    """

    data = np.genfromtxt(file, dtype=[('row', int), ('col', int), ('val', float)])
    rows = data['row'] - 1
    cols = data['col'] - 1
    vals = data['val']

    if transposeData:
        sparseData = scipySparse.coo_matrix(numpyArray.T)
    else:
        sparseData = scipySparse.coo_matrix(numpyArray)

    print("============array: %s =============" % label)
    # inverted rows<-> cols because matlab process columnwise, and numpy rowwise
    print("largest diff (absolute value) of row index          : %d" % np.amax(np.absolute(sparseData.row - cols)))
    print("largest diff (absolute value) of col index          : %d" % np.amax(np.absolute(sparseData.col - rows)))
    print("largest diff (absolute value) of values             : %1.5e" % np.amax(np.absolute(sparseData.data - vals)))
    print("==================================")


def readNthLineData(filePath,  # type: str
                    lineNbr=0,  # type: int
                    separator=' '  # type: str
                    ):
    """
    read one line of a file containing an array

    Parameters
    ----------
    filePath: str
        path of the file
    lineNbr: int
        number of the line. this number starts at 0 (first line)
    separator: string
        string used as separator
    Returns
    -------
    values : numpy array
        data contained in the line. If the file has less lines than lineNbr, then line = None
    """
    temp = readNthLine(filePath, lineNbr)
    if temp is not None:
        return np.fromstring(temp, sep=separator)
    else:
        return None

def roundDown(value, digits=8):
    return math.floor(value * 10**digits) / 10**digits

def roundUp(value, digits=8):
    return math.ceil(value * 10**digits) / 10**digits

def getNumberOfLines(file):
    """
    counts the number of lines, excluding eventual empty lines
    Parameters
    ----------
    file: string
        file to be analyzed
    Returns
    -------
    N: int
        number of lines
    """
    f = open(file, 'r')
    nLines = sum(1 for line in f if line.rstrip())
    f.close()
    return nLines


def readNthLine(filePath,  # type: str
                lineNbr=0  # type: int
                ):
    """
    Parameters
    ----------
    filePath: str
        path of the file

    lineNbr: int
        number of the line. this number starts at 0 (first line)
    Returns
    -------
    line : string
        nth line of the file. If the file has less lines than lineNbr, then line = None
    """
    with open(filePath) as fp:
        for i, line in enumerate(fp):
            if i == lineNbr:
                return line
    return None


# case: 'upper', 'lower', None
def setFileExtension(filePath,  # type: str
                     extension,  # type: str
                     case=None  # type: str
                     ):
    """
    given a string representing a file path, sets/replaces the extension of the file path.

    Obs: this function does not change the file, it changes the string of the path only.

    Parameters
    ----------
    filePath: str
        path of the file

    extension: str
        target extension

    case: str {'upper', 'lower', 'same'}, optional
        'upper': forces the extension to be uppercase
        'lower': forces the extension to be lowercase
        'same': does not change the case


    Returns
    -------
    newPath: str
        the path of the file, with the extension added/replaced
    """
    baseFile, _ = os.path.splitext(filePath)
    extFile = None
    if case is None:
        extFile = extension
    else:
        if case.lower() == 'upper':
            extFile = extension.upper()
        if case.lower() == 'lower':
            extFile = extension.lower()

    if not extension.startswith('.'):
        extFile = '.' + extFile

    return baseFile + extFile


def createDir(directory):
    """
    creates a directory
    Parameters
    ----------
    directory: string
        path of the directory. can be relative or absolute paths. if 'directory' contains new subdirectories, this function will
        create them.
    Returns
    -------
        True if sucesfull an False otherwise
    """
    if not os.path.isdir(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError:
            print("Creation of the directory %s failed" % directory)
            return False
        else:
            print("Successfully created the directory %s " % directory)
            return True


def splitPath(path):
    """
    splits a path between its components
    Ex:  /path/to/file.ext
            dirName: /path/to/   <-- includes last /!!
            filePrefixName: file
            fileExtension: .ext
    Parameters
    ----------
    path: string
        string representing a path
    Returns
    -------
        [dirName, filePrefixName, fileExtension]
    """
    dirName = os.path.dirname(path) + '/'
    filename_w_ext = os.path.basename(path)
    filePrefixName, fileExtension = os.path.splitext(filename_w_ext)

    return [dirName, filePrefixName, fileExtension]

def fileExtensionCase(filePath,  # type: str
                      case='lower'  # type: str
                      ):
    """
    given a string representing a file path, forces the extension to be lowercase or uppercase

    Obs: this function does not change the file, it changes the string of the path only.

    Parameters
    ----------
    filePath: str
        path of the file

    case: str {'upper','lower'}
        case of the extension


    Returns
    -------
    newPath: str
        new file path
    """
    base, ext = os.path.splitext(filePath)

    if case.lower() == 'lower':
        return base + '.' + ext.lower()
    if case.lower() == 'upper':
        return base + '.' + ext.upper()


def isPointInsideTetra(point, coords):
    """
    determines if a point lies inside a give tetrahedron

    based on
    https://github.com/ncullen93/mesh2nifti/blob/master/msh2nifti.py

    Parameters
    ----------
    point: numpy array
        coordinates of the point

    coords: numpy 2D array
       tetrahedron coordinates. Each line is a vertice, each column one coordinate [x,y,z]

    Returns
    -------
        out: boolean
            True: inside, False, outside
    """

    vals = np.ones((5, 4))
    vals[0, :3] = point
    vals[1:, :3] = coords

    idxs = [[1, 2, 3, 4], [0, 2, 3, 4], [1, 0, 3, 4], [1, 2, 0, 4], [1, 2, 3, 0]]

    dets = np.linalg.det(vals[idxs, :])
    return np.all(dets > 0) if dets[0] > 0 else np.all(dets < 0)


def isActiveElement(element, xpath):
    """
    given one element with the attribute 'active', return its state

    Parameters
    ----------
    element: lxml.etree of lxml.element
        main element
    xpath: str
        xpath of one element. use xpath='.' to consider the element itself.

    Returns
    -------
    activity status: bool

    """
    return getElemAttrXpath(element, xpath, attrName='active', attrType='bool')


def activateElem(element, xpath):
    """
    activates an element. i.e., sets its 'active' attribute to 'True'

    Parameters
    ----------
    element: lxml.etree of lxml.element
        main element
    xpath: str
        xpath of one element. use xpath='.' to consider the element itself.
    """
    elem = element.xpath(xpath)[0]
    elem.set('active', 'True')


def deactivateElem(element, xpath, clearElem=False):
    """
    deactivates an element. i.e., sets its 'active' attribute to 'False'

    Parameters
    ----------
    element: lxml.etree of lxml.element
        main element
    xpath: str
        xpath of one element. use xpath='.' to consider the element itself.
    clearElem: bool
        clears completely the element, removing all children elements and also any other attibutes, except the 'active' attribute
    """
    elem = element.xpath(xpath)[0]
    if clearElem:
        elem.clear()
    elem.set('active', 'False')  # elem.text = ''


def getElemValueXpath(element,  # type: ETree
                      xpath='.',  # type: str
                      valType='int'  # type: str
                      ):
    """
    given a lxml.element or lxml.elementTree, returns the text of a sub element of the xpath

    If there are more than one element at the xpath. this function will return the value of the first occurrence.
    Parameters
    ----------
    element: lxml.etree of lxml.element
        main element
    xpath: str
        xpath of one element. use xpath='.' to consider the element itself.

    valType: str {'int', 'float', 'str', 'bool', 'list_int', 'list_float', 'list_list_int', 'list_list_float'}
        expected type of the value.

    Returns
    -------
    value:
        the value, already converted of the expected type

    Examples
    --------
    let the xml file be

    <?xml version="1.0" ?>
    <EITmodel>
        <current>
            <pattern>bipolar_skip</pattern>
            <skip>4</skip>
            <value unit="mA">10</value>
        </current>
        <FEMmodel>
            <general>
                <meshFile>./myMesh.msh</meshFile>
                <meshFileUnit>mm</meshFileUnit>
                <dimension>3</dimension>
                <height2D>2.0</height2D>
                <nElectrodes>2</nElectrodes>
            </general>
        </FEMmodel>
    </EITmodel>


    root = ETree.parse('xmlFile.xml'.getroot()  # EITmodel is the root in this example
    pattern = getElemValueXpath(root, xpath='current/pattern', valType='str')
    height = getElemValueXpath(root, xpath='FEMmodel/general/height2D', valType='float')
    """

    elemList = element.xpath(xpath)

    if len(elemList) == 0:
        print('ERROR! Xpath -> %s <- not found!' % xpath)
        exit()

    elemText = elemList[0].text.rstrip()

    return convStr(elemText, valType)


# 'int', 'float', 'str', 'bool', 'list_int', 'list_float', list_list_int, list_list_float
def getElemAttrXpath(element,  # type: ETree
                     xpath,  # type: str
                     attrName,  # type: str
                     attrType='int'  # type: str
                     ):
    """
    given a lxml.element or lxml.elementTree, returns the text of a sub element of the xpath

    If there are more than one element at the xpath. this function will return the value of the first occurrence.
    Parameters
    ----------
    element: lxml.etree of lxml.element
        main element
    xpath: str
        xpath of one element. use xpath='.' to consider the element itself.

    attrName: str
        name of the attribute

    attrType: str {'int', 'float', 'str', 'bool', 'list_int', 'list_float', 'list_list_int', 'list_list_float'}
        expected type of the attribute.

    Returns
    -------
    value:
        the value, already converted of the expected type

    Examples
    --------
    let the xml file be

    <?xml version="1.0" ?>
    <EITmodel>
        <current>
            <pattern>bipolar_skip</pattern>
            <skip>4</skip>
            <value unit="mA">10</value>
        </current>
    </EITmodel>


    root = ETree.parse('xmlFile.xml'.getroot()  # EITmodel is the root in this example
    currUnit = tools.getElemAttrXpath(root, xpath='current/value', attrName='unit', attrType='str')
    """
    elemList = element.xpath(xpath)

    if len(elemList) == 0:
        print('ERROR! Xpath -> %s <- not found!' % xpath)
        exit()

    try:
        elemAttr = elemList[0].attrib[attrName]
    except KeyError:
        print('XML attribute ->%s<- not found. Returning None' % attrName)
        return None


    return convStr(elemAttr, attrType)


def convStr(string,  # type: str
            outputType='int'  # type: str
            ):
    """
    given a string, converts it the specified type

    In case of list, it is expected that the string has the following format:
    '[1.0 2.0 3.0 4.0 ]' --> convert to a list of floats
    '[1 2 3 4 ]' --> convert to a list of integers
    '['abc' 'def ghi']' --> convert to a list of integers. spaces within quotes is kept
    '[[1 2] [3 4]]' --> convert to a list of list of integers
    '[[1.0 2.0] [3.0 4.0]]' --> convert to a list of list of floats

    Parameters
    ----------
    string: str
        string to be converted

    outputType: str {'int', 'float', 'str', 'bool', 'list_int', 'list_float', 'list_str', 'list_list_int', 'list_list_float'}
        expected type of the attribute.


    Returns
    -------
    value:
        the value, already converted of the expected type


    -------

    """

    if outputType not in ['int', 'float', 'str', 'bool', 'list_int', 'list_float', 'list_list_int', 'list_list_float']:
        print('Error: outputType -> %s <- not recognized. Valid options: int, float, str, bool, list_int, list_float, list_list_int, list_list_float' % outputType)
        exit(-1)

    if string == 'None':
        return None

    if outputType == 'int':
        return int(string)

    if outputType == 'float':
        return float(string)

    if outputType == 'str':
        return string

    if outputType == 'bool':
        if string.lower() == 'true':
            return True
        else:
            return False

    if outputType == 'list_int':
        listProcessed = string.strip('][').split()
        if listProcessed == '':
            return []
        else:
            return [int(x) for x in listProcessed]

    if outputType == 'list_str':
        stringCleaned = string.strip('][')
        return shlex.split(stringCleaned)

    if outputType == 'list_float':
        listProcessed = string.strip('][').split()
        if listProcessed == '':
            return []
        else:
            return [float(x) for x in listProcessed]

    if outputType == 'list_list_int':
        #extract outer brackets using greedy match between [ ]
        string = re.search('(?<=\[).+(?=\])', string).group()
        # extract each row  using non-greedy match between [] of each row
        rows = re.findall('\[(.*?)\]', string)
        result = []
        for row in rows:
            listProcessed = row.split()
            if listProcessed == '':
                result.append( [])
            else:
                result.append([int(x) for x in listProcessed])
        return result

    if outputType == 'list_list_float':
        #extract outer brackets using greedy match between [ ]
        string = re.search('(?<=\[).+(?=\])', string).group()
        # extract each row  using non-greedy match between [] of each row
        rows = re.findall('\[(.*?)\]', string)
        result = []
        for row in rows:
            listProcessed = row.split()
            if listProcessed == '':
                result.append( [])
            else:
                result.append([float(x) for x in listProcessed])
        return result
