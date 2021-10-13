#!/bin/python
"""
finite element FemModel class.
"""

import multiprocessing as mp
import os
import shutil
import copy
# -*- coding: utf-8 -*-
import sys
import time

import numpy as np
from lxml import etree as ETree
from scipy import sparse as scipySparse

import FEMelements
# import gmsh
import myGmsh
import tools


def extract_COO(elem):
    tempKlocal_coo = scipySparse.coo_matrix(elem.Kgeom)
    nComponents = tempKlocal_coo.nnz

    # multiply by 1/rho
    if elem.propertiesDict['isElectrode']:
        tempKlocal_coo *= (1.0 / elem.rhoT)
    else:
        tempKlocal_coo *= (1.0 / elem.rho)

    # re-write row and col in terms of global node numbers
    row = elem.connectivity[tempKlocal_coo.row]
    col = elem.connectivity[tempKlocal_coo.col]
    val = tempKlocal_coo.data

    return [row, col, val]


# gmsh.initialize()

# noinspection PyAttributeOutsideInit
class FemModel():
    """
    main finite element model class


    This class is responsible for loading the gmsh file and creating the global stiffness matrix.
    """

    def __init__(self, confFile, outputBaseDir):
        """

        Parameters
        ----------
        confFile: str
            configuration file .conf
        """
        self.confFile = confFile
        self.outputBaseDir = outputBaseDir
        [self.baseDir, self.filePrefix, _] = tools.splitPath(self.confFile)
        self._loadConf()

    def _loadConf(self):
        """
        load information for the FEM model from the .conf file. this function is automatically called by __init__
        """
        self.confFEM = ETree.parse(self.confFile).getroot().xpath('FEMmodel')[0]

        self.fileMSH = os.path.abspath(self.baseDir + tools.getElemValueXpath(self.confFEM, xpath='general/meshFile', valType='str'))
        self.nElectrodes = tools.getElemValueXpath(self.confFEM, xpath='general/nElectrodes', valType='int')
        self.dimension = tools.getElemValueXpath(self.confFEM, xpath='general/dimension', valType='int')

        if self.dimension == 2:
            self.height2D = tools.getElemValueXpath(self.confFEM, xpath='general/height2D', valType='float')
            unit = tools.getElemAttrXpath(self.confFEM, xpath='general/height2D', attrName='unit', attrType='str')
            self.height2D = tools.convToMetre(self.height2D, unit)
        else:
            self.height2D = None

    def loadGmsh(self):
        """
        load gmsh file into memory and create the local matrices of all elements and electrodes.
        """

        # original gmsh API is broken with multiprocessing...
        originalGmsh = False

        if originalGmsh:
            gmsh.initialize()
            time.sleep(1.0)

            gmsh.open(self.fileMSH)
            nids, coord, parametric_coord = gmsh.model.mesh.getNodes()
            self.nodeCoords = np.array(coord).reshape(len(coord) // 3, 3)

            # must reorder since gmsh function returns the nodes in strange order. I don't know why
            self.nodeCoords = self.nodeCoords[np.argsort(nids)][:]
        else:
            self.myGmshF = myGmsh.myGmsh()
            self.myGmshF.open(self.fileMSH)
            self.nodeCoords = self.myGmshF.getNodes()

            if self.dimension == 2:
                #with inclusion
                #self.myGmshF.plotMesh2D(title='Mesh', domainTags=[3, 4], electrodeTags=[1, 2], fileName=self.outputBaseDir + self.filePrefix +
                # '_mesh2D.svg')

                #without inclusion
                self.myGmshF.plotMesh2D(title='Mesh', domainTags=[3], electrodeTags=[1, 2], fileName=self.outputBaseDir + self.filePrefix +
                '_mesh2D.svg')

        # convert the mesh to metres
        meshUnit = tools.getElemAttrXpath(self.confFEM, xpath='general/meshFile', attrName='unit', attrType='str')
        self.nodeCoords = tools.convToMetre(self.nodeCoords, meshUnit)

        # rotate mesh if needed
        self.RotMat=[]
        if tools.isActiveElement(self.confFEM, xpath='rotations'):
            self.hasRotation=True
            rotations = self.confFEM.xpath('rotations')[0]
            for rot in rotations.iter('rotation'):
                axis=tools.getElemValueXpath(rot, xpath='axis', valType='str').lower()
                angleRad=tools.getElemValueXpath(rot, xpath='angle_deg', valType='float')*np.pi/180.0
                self.RotMat.append(tools.rotMatrix(axis,angleRad))

            self.nodeCoords = self.applyRotation( self.nodeCoords.T, isInverse=False).T
        else:
            self.hasRotation = False

        # domainRegions
        print('Building domain regions...')
        self.elements = []
        regionsXML = self.confFEM.xpath('regions')[0]
        for region in regionsXML.iter('region'):
            if tools.isActiveElement(region, xpath='.'):
                label = tools.getElemValueXpath(region, xpath='label', valType='str')
                tags = tools.getElemValueXpath(region, xpath='meshTag', valType='list_int')
                isGrouped = tools.getElemValueXpath(region, xpath='isGrouped', valType='bool')
                dim = tools.getElemValueXpath(region, xpath='dimension', valType='int')
                rho = tools.getElemValueXpath(region, xpath='rho0', valType='float')

                print('  -> Building %s region...' % label)
                for tag in tags:

                    if originalGmsh:
                        # read data from FemModel
                        etypes, elements, connectivities = gmsh.model.mesh.getElements(dim, tag)
                        elements = elements[0]

                        # minus 1 bc node number starts with 1 in gmsh and here starts in 0
                        connectivities = connectivities[0] - 1

                        # reshape into 2D array
                        if dim == 2:
                            connectivities = np.array(connectivities).reshape(len(connectivities) // 3, 3)
                        if dim == 3:
                            connectivities = np.array(connectivities).reshape(len(connectivities) // 4, 4)

                        # must reorder since gmsh function returns the nodes in strange order. I don't know why
                        idxSorted = np.argsort(elements)
                        connectivities = connectivities[idxSorted, :]
                        elements = elements[idxSorted]
                    else:
                        [elements, connectivities] = self.myGmshF.getElem(dim, tag)

                    # build local FEM matrices
                    if isGrouped:
                        elem = FEMelements.UniformRhoRegion(dimension=dim, elemNbr=len(self.elements), connectivities=connectivities,
                                                            coords=self.nodeCoords, rho=rho, height2D=self.height2D, isSparse=False,
                                                            propertiesDict={'isElectrode': False, 'regionTag': tag, 'gmshElemNbr': elements})
                        self.elements.append(elem)

                    else:
                        # Setup a list of processes that we want to run

                        if dim == 2:
                            args = [(i + len(self.elements), c, self.nodeCoords, rho, self.height2D,
                                     {'isElectrode': False, 'regionTag': tag, 'gmshElemNbr': elements[i]}) for i, c in enumerate(connectivities)]
                            with mp.Pool(processes=tools.nCores) as p:
                                self.elements += p.starmap(FEMelements.LinearTriangle, args)

                        if dim == 3:
                            args = [(
                                i + len(self.elements), c, self.nodeCoords, rho, {'isElectrode': False, 'regionTag': tag, 'gmshElemNbr': elements[i]})
                                for i, c in enumerate(connectivities)]
                            with mp.Pool(processes=tools.nCores) as p:
                                self.elements += p.starmap(FEMelements.LinearTetrahedron, args)

                        del args  # with open("lixo_Kgeom_%s_numpy.txt" % label, "a") as f:  #    np.savetxt(f, elem.Kgeom.reshape(1, 16,order='F'))

        self.nElements = len(self.elements)

        # also includes the virtual nodes of the electrodes
        self.nNodes = self.nodeCoords.shape[0] + self.nElectrodes

        # electrodes
        print('  -> Building electrode elements...')

        electrodesXML = self.confFEM.xpath('electrodes')[0]

        tags = tools.getElemValueXpath(electrodesXML, xpath='meshTag', valType='list_int')
        rhoT = tools.getElemValueXpath(electrodesXML, xpath='rhoT0', valType='float')
        self.electrodes = []
        self.electrodeNodes = []
        for tag in tags:

            if originalGmsh:                # read data from FemModel
                elements = []
                connectivities = []
                if self.dimension == 2:
                    # if dim=2D, electrodes are line elements, with 2 nodes
                    etypes, elements, connectivities = gmsh.model.mesh.getElements(1, tag)
                if self.dimension == 3:
                    # if dim=3D, electrodes are triangle elements, with 3 nodes
                    etypes, elements, connectivities = gmsh.model.mesh.getElements(2, tag)

                elements = elements[0]
                connectivities = connectivities[0] - 1  # minus 1 bc node starts in 1 in gmsh and here starts in 0

                # reshape into 2D array
                if self.dimension == 2:
                    # if dim=2D, electrodes are line elements, with 2 nodes
                    connectivities = np.array(connectivities).reshape(len(connectivities) // 2, 2)
                if self.dimension == 3:
                    # if dim=3D, electrodes are triangle elements, with 3 nodes
                    connectivities = np.array(connectivities).reshape(len(connectivities) // 3, 3)

                # must reorder since gmsh function returns the nodes in strange order. I don't know why.
                idxSorted = np.argsort(elements)
                connectivities = connectivities[idxSorted, :]
                elements = elements[idxSorted]
            else:
                if self.dimension == 2:
                    # if dim=2D, electrodes are line elements, with 2 nodes
                    [elements, connectivities] = self.myGmshF.getElem(1, tag)
                if self.dimension == 3:
                    # if dim=3D, electrodes are triangle elements, with 3 nodes
                    [elements, connectivities] = self.myGmshF.getElem(2, tag)

            # build local FEM matrices
            self.electrodeNodes.append(self.nodeCoords.shape[0] + len(self.electrodes))

            elem = FEMelements.CompleteElectrodeHua(dimension=self.dimension, elemNbr=len(self.elements), connectivities=connectivities,
                                                    coords=self.nodeCoords, rhoT=rhoT, height2D=self.height2D, virtualNodeNbr=self.electrodeNodes[-1],
                                                    isSparse=False, propertiesDict={'isElectrode': True, 'regionTag': tag, 'gmshElemNbr': elements})
            self.electrodes.append(elem)
            self.elements.append(elem)

        # converts to numpy array
        self.electrodeNodes = np.array(self.electrodeNodes)

        if originalGmsh:
            gmsh.finalize()

    def applyRotation(self,coords,isInverse=False):
        """
        apply stored rotations to a matrix of coordinates
            the first rotation is the first element of the list self.RotMat, and so on.
        Parameters
        ----------
        coords: numpy array
            coordinate matrix. This matrix must be 3xN .
        isInverse: bool
            if true, apply the inverse of the rotations .In this case, the transpose (=inverse) of the last element of self.RotMat is applied first.
        """
        temp=copy.copy(coords)
        if isInverse:
            for mat in  reversed(self.RotMat):
                temp = np.matmul(mat.T, temp)
        else:
            for mat in  self.RotMat:
                temp = np.matmul(mat, temp)
        return temp

    def getDomainElements(self):
        """
        Return the elements of the mesh that are not electrodes
        Returns
        -------
            elements: list of elements
        """
        return [elem for elem in self.elements if not elem.propertiesDict['isElectrode']]

    def getDomainElemQuality(self, fileName):
        """
        Return a file with some parameter of the elements

        Returns
        -------
            elements: list of elements
        """
        meshUnit = tools.getElemAttrXpath(self.confFEM, xpath='general/meshFile', attrName='unit', attrType='str')
        with open(self.outputBaseDir + fileName, 'w') as f:
            f.write('#Nbr; centroid X; centroid Y; centroid Z; volume; aspectRatio\n')
            for elem in self.elements:
                if not elem.propertiesDict['isElectrode']:
                    centroidMeshUnit = tools.convFromMetre(elem.centroid, meshUnit)
                    volumeMeshUnit = tools.convFromMetre(tools.convFromMetre(tools.convFromMetre(elem.volume, meshUnit), meshUnit), meshUnit)
                    f.write('%d %f %f %f %f %f \n' % (
                        elem.number, centroidMeshUnit[0], centroidMeshUnit[1], centroidMeshUnit[2], volumeMeshUnit, elem.getAspectRatio()))

    def getMeshLimits(self):
        """
        Retuns the limits of the mesh.

        Returns
        -------
        listLimits: list of np arrays
            list of limits in the form   [ [minX, minY, min Z] , [maxX, maxY, maxZ] ]

        """
        minimum = np.min(self.nodeCoords, axis=0)
        maximum = np.max(self.nodeCoords, axis=0)
        return [minimum, maximum]

    def setResistivities(self, elemNumbers, rhoVector):
        """
        set the resistivities of the elements

        Note: Remember that if a regionTag is configured to be grouped, then there is a single element that
        represent the entire region.

        Parameters
        ----------

        elemNumbers: iterable of ints
            elements to set the resitivity

        rhoVector: 1D numpy.array
            vector with resistivities, or electrode parameter (rho.t) in case of electrode elements
        """
        for (e, rho) in zip(elemNumbers, rhoVector):
            if self.elements[e].propertiesDict['isElectrode']:
                self.elements[e].setRhoT(rho)
            else:
                self.elements[e].setRho(rho)

    def setMeshTagResistivity(self, meshTagList, rho):
        """
        set the resistivities of all elements with a given meshTag value

        Parameters
        ----------

        meshTagList: iterable of ints
            meshTag list

        rho: double
            resistivity, or electrode parameter (rho.t) in case of electrode elements

        """
        for elem in self.elements:
            if elem.propertiesDict['regionTag'] in meshTagList:
                if elem.propertiesDict['isElectrode']:
                    elem.setRhoT(rho)
                else:
                    elem.setRho(rho)

    def getElementsByMeshTag(self, meshTagList):
        """
        return a list of elements with the given meshTag

        Parameters
        ----------
        meshTagList: list of int
            meshTags of the regions

        Returns
        -------
        listElem: list
            list of elements

        """
        listElem = []
        for elem in self.elements:
            if elem.propertiesDict['regionTag'] in meshTagList:
                listElem.append(elem)
        return listElem

    def getElementsByElemNbr(self, elementNbrList):
        """
        return a list of elements with the given meshTag

        Parameters
        ----------
        elementNbrList: list of int
            number of the elements

        Returns
        -------
        listElem: list
            list of elements

        """
        return [self.elements[e] for e in elementNbrList]

    def exportGmsh_RhoElements(self, elements, title='Solution', iterNbr=0, sufixName='_output', mode='append'):
        """
        create a msh file for post processing

        Parameters
        ----------

        sufixName: string
            file name sufix

        elements: list
            list of FEM elements to be included


        mode: string
            valid options: 'append', 'new', 'solution_only'
            - 'append': Appends the resistivities to an existing File. In this case, one must make sure the titles of each
            solution is unique. This option allow many solutions to be saved in a single .msh file. If the output file does not exist,
            this mode works exactly as 'new', i.e., a new file will be created and the solution will be included.
            - 'new': creates a mesh file with both geometry definition and solution in one $ElementData of the msh format.
                This is used to save a single solution to a .msh file. Obs: you can use later 'append' to the file created in this mode to add more solutions
            - 'solution_only': saves only the information about the solution, i.e., no information about the mesh is saved. In order to use this
            file in gmsh, you must concatenate a .msh file containing the geometry and the file generated with this funcions, e..,
            using 'cat' command


            See: https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format
        """

        # find number of elements and build the string with all resistivities
        elementData = ''
        nElements = 0
        for elem in elements:
            if elem.isRegion:
                nElements += elem.nElements
                for i in range(elem.nElements):
                    if elem.propertiesDict['isElectrode']:
                        elementData += '%d %1.15e\n' % (elem.propertiesDict['gmshElemNbr'][i], elem.rhoT)
                    else:
                        elementData += '%d %1.15e\n' % (elem.propertiesDict['gmshElemNbr'][i], elem.rho)
            else:
                elementData += '%d %1.15e\n' % (elem.propertiesDict['gmshElemNbr'], elem.rho)
                #print('%d %d' % (elem.propertiesDict['gmshElemNbr'],elem.number))
                nElements += 1

        if mode.lower() == 'append':
            outputfileName = self.outputBaseDir + self.filePrefix + sufixName + '.msh'
            if not os.path.exists(outputfileName):
                shutil.copy2(self.fileMSH, outputfileName)

        if mode.lower() == 'new':
            outputfileName = self.outputBaseDir + self.filePrefix + sufixName + '.msh'
            shutil.copy2(self.fileMSH, outputfileName)

        if mode.lower() == 'solution_only':
            outputfileName = self.outputBaseDir + self.filePrefix + sufixName + '.sol'
            with open(outputfileName, 'w') as file:
                file.write('// please concatenate this file with %s.msh to open in gmsh\n' % self.filePrefix)
                file.write('// example: cat [path/to/]%s %s > temp.msh; gmsh temp.msh\n' % (
                    os.path.basename(self.fileMSH), self.filePrefix + sufixName + '.sol'))

        with open(outputfileName, 'a') as file:
            stringTags = ['"%s"' % title]
            realTags = [0.0]
            intTags = [iterNbr,  # time step
                       1,  # 1: scalar value, 3: vector, 9: tensor
                       nElements]  # number of elements in the list

            file.write('$ElementData\n')

            file.write('%d\n' % len(stringTags))  # number-of-string-tags
            for string in stringTags:
                file.write(string + '\n')  # string tags

            file.write('%d\n' % len(realTags))  # number-of-real-tags
            for val in realTags:
                file.write('%e\n' % val)  # real tags

            file.write('%d\n' % len(intTags))  # number-of-integer-tags
            for val in intTags:
                file.write('%d\n' % val)  # real tags

            file.write(elementData)

            file.write('$EndElementData\n')

        if self.dimension == 2:
            rho_elements = []

            for elem in elements:
                if elem.isRegion:
                    for i in range(elem.nElements):
                        if not elem.propertiesDict['isElectrode']:
                            rho_elements.append(elem.rho)
                else:
                    rho_elements.append(elem.rho)

            #self.myGmshF.plotdata2D(domainTags=[3, 4], electrodeTags=[1, 2], nodeData=nodalVoltages[:, 0], elementData=np.array(rho_elements),
            #                        title=None, fileName=self.outputBaseDir + self.filePrefix + '_node.png', nIsopotentialLines=30,
            #                        drawStreamLines=True, drawElementEdges=False, drawBoundaries=True, drawElectrodes=True)
            self.myGmshF.plotdata2D(domainTags=[3], electrodeTags=[1, 2], nodeData=None, elementData=np.array(rho_elements), title=None,
                                    nIsopotentialLines=30, drawElementEdges=True, fileName=self.outputBaseDir + self.filePrefix + '_elem.png',
                                    drawBoundaries=True, drawElectrodes=True)

    def exportGmsh_NodalVoltages(self, nodalVoltages, title='Solution', iterNbr=0, sufixName='_output', mode='append'):
        """
        create a msh file for post processing

        Parameters
        ----------

        sufixName: string
            file name sufix

        nodalVoltages: list or np array
            nodal voltages

        mode: string
            valid options: 'append', 'new', 'solution_only'
            - 'append': Appends the resistivities to an existing File. In this case, one must make sure the titles of each
            solution is unique. This option allow many solutions to be saved in a single .msh file. If the output file does not exist,
            this mode works exactly as 'new', i.e., a new file will be created and the solution will be included.
            - 'new': creates a mesh file with both geometry definition and solution in one $ElementData of the msh format.
                This is used to save a single solution to a .msh file. Obs: you can use later 'append' to the file created in this mode to add more solutions
            - 'solution_only': saves only the information about the solution, i.e., no information about the mesh is saved. In order to use this
            file in gmsh, you must concatenate a .msh file containing the geometry and the file generated with this funcions, e..,
            using 'cat' command


            See: https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format
        """

        if mode.lower() == 'append':
            outputfileName = self.outputBaseDir + self.filePrefix + sufixName + '.msh'
            if not os.path.exists(outputfileName):
                shutil.copy2(self.fileMSH, outputfileName)

        if mode.lower() == 'new':
            outputfileName = self.outputBaseDir + self.filePrefix + sufixName + '.msh'
            shutil.copy2(self.fileMSH, outputfileName)

        if mode.lower() == 'solution_only':
            outputfileName = self.outputBaseDir + self.filePrefix + sufixName + '.sol'
            with open(outputfileName, 'w') as file:
                file.write('// please concatenate this file with %s.msh to open in gmsh\n' % self.filePrefix)
                file.write('// example: cat [path/to/]%s %s > temp.msh; gmsh temp.msh\n' % (
                    os.path.basename(self.fileMSH), self.filePrefix + sufixName + '.sol'))

        for i in range(nodalVoltages.shape[1]):
            with open(outputfileName, 'a') as file:
                stringTags = ['"%s"' % title]
                realTags = [0.0]
                intTags = [iterNbr,  # time step
                           1,  # 1: scalar value, 3: vector, 9: tensor
                           nodalVoltages.shape[0]]  # number of elements in the list

                file.write('$NodeData\n')
                file.write('%d\n' % len(stringTags))  # number-of-string-tags
                for string in stringTags:
                    file.write(string + '\n')  # string tags

                file.write('%d\n' % len(realTags))  # number-of-real-tags
                for val in realTags:
                    file.write('%e\n' % val)  # real tags

                file.write('%d\n' % len(intTags))  # number-of-integer-tags
                for val in intTags:
                    file.write('%d\n' % val)  # real tags

                for j, v in enumerate(nodalVoltages[:, i]):
                    file.write('%d %1.15e\n' % (j + 1, v))

                file.write('$EndNodeData\n')

        if self.dimension == 2:
            rho_elements = []

            for elem in self.elements:
                if elem.isRegion:
                    for i in range(elem.nElements):
                        if not elem.propertiesDict['isElectrode']:
                            rho_elements.append(elem.rho)
                else:
                    rho_elements.append(elem.rho)
            #self.myGmshF.plotdata2D(domainTags=[3, 4], electrodeTags=[1, 2], nodeData=nodalVoltages[:, 0], elementData=np.array(rho_elements),
            #                        title=None, fileName=self.outputBaseDir + self.filePrefix + '_node.png', nIsopotentialLines=30,
            #                        drawStreamLines=True, drawElementEdges=False, drawBoundaries=True, drawElectrodes=True)
            self.myGmshF.plotdata2D(domainTags=[3], electrodeTags=[1, 2], nodeData=nodalVoltages[:, 0], elementData=np.array(rho_elements),
                                    title=None, fileName=self.outputBaseDir + self.filePrefix + '_node.png', nIsopotentialLines=30,
                                    drawStreamLines=True, drawElementEdges=False, drawBoundaries=True, drawElectrodes=True)
            print('oi')

    def buildKglobal(self):
        """
        compute global matrix, add referecence node and build its sparse version.
        """

        self._compute_Kglobal()

        # compares solution
        confDebug = ETree.parse(self.confFile).getroot().xpath('debugMode')[0]
        if tools.isActiveElement(confDebug, xpath='kglobal'):
            fileName = self.baseDir + tools.getElemValueXpath(confDebug, xpath='kglobal', valType='str')

            data = np.genfromtxt(fileName, dtype=[('row', int), ('col', int), ('val', float)])
            rows = data['row'] - 1
            cols = data['col'] - 1
            vals = data['val']
            dataFile = scipySparse.coo_matrix((vals, (rows, cols)), shape=self.Kglobal.shape).todense()

            print("  -> largest difference (absolute value) of %s : %1.5e" % ('k_global', np.amax(np.absolute(self.Kglobal - dataFile))))

        self.setReferenceVoltageNode()

    def _compute_Kglobal(self):
        """
        compute global matrix.
        """

        try:
            del self.KglobalSp
        except AttributeError:
            pass

        print('Building FEM global matrix...')

        # extract COO information in parallel
        args = [(e,) for e in self.elements]
        with mp.Pool(processes=tools.nCores) as p:
            dataList = p.starmap(extract_COO, args)

        # find the total number of non zero elements in all local matrices
        size = 0
        for data in dataList:
            size += len(data[0])

        rows = np.empty(size, dtype=np.int)
        cols = np.empty(size, dtype=np.int)
        vals = np.empty(size)

        pos = 0
        for data in dataList:
            nComponents = len(data[0])

            # re-write row and col in terms of global node numbers
            rows[pos:pos + nComponents] = data[0]
            cols[pos:pos + nComponents] = data[1]
            vals[pos:pos + nComponents] = data[2]
            pos += nComponents

        self.KglobalSp = scipySparse.coo_matrix((vals, (rows, cols)), shape=(self.nNodes, self.nNodes)).tocsr()

    def saveKglobal(self, fileName,  # type: str
                    binary=False  # type: bool
                    ):
        """
        save global matrix to a text file.

        Parameters
        ----------
        fileName: str
            file path

        binary: bool, optional
            save in binary format. Default: False
        """
        if binary:
            np.save(fileName, self.Kglobal)
        else:
            np.savetxt(fileName, self.Kglobal)

    def setReferenceVoltageNode(self):
        """
        set the reference node for the measurements
        """
        confRefNode = ETree.parse(self.confFile).getroot().xpath('voltage/referenceVoltageNode')[0]

        method = tools.getElemValueXpath(confRefNode, xpath='method', valType='str').lower()

        if method == 'fixed_electrode':
            electrdeNbr = tools.getElemValueXpath(confRefNode, xpath='fixedElectrodeNbr', valType='int')
            self.voltageRefNode = self.electrodeNodes[electrdeNbr - 1]  # subtracts 1 because electrode numbers start with 0 in the code.

        if method == 'origin':
            nodeDists = np.sum(self.nodeCoords * self.nodeCoords, axis=1)
            self.voltageRefNode = np.argmin(nodeDists)

        if method == 'nodeNbr':
            customNode = tools.getElemValueXpath(confRefNode, xpath='nodeNbr', valType='int')
            self.voltageRefNode = customNode - 1  # subtracts 1 because node numbers start with 0 in the code.

        if method == 'coords':
            coords = np.array(tools.getElemValueXpath(confRefNode, xpath='coords', valType='list_float'))
            coordsUnit = tools.getElemAttrXpath(confRefNode, xpath='coords', attrName='unit', attrType='str')
            coords = tools.convToMetre(coords, coordsUnit)

            coords = self.applyRotation(coords,isInverse=False)

            nodeDists = np.sum((self.nodeCoords - coords) ** 2, axis=1)
            self.voltageRefNode = np.argmin(nodeDists)

        self.KglobalSp[self.voltageRefNode, :] = 0
        self.KglobalSp[:, self.voltageRefNode] = 0
        self.KglobalSp[self.voltageRefNode, self.voltageRefNode] = 1.0


if __name__ == '__main__':

    if sys.version_info.major == 2:
        sys.stdout.write("Sorry! This program requires Python 3.x\n")
        sys.exit(1)

    file = './malhasAnaliticas/2D/analitica_2D.conf'

    myMesh = FemModel(file)
    myMesh.loadGmsh()
    myMesh.compute_Kglobal()

    tools.compareMatrices(myMesh.Kglobal, './malhasAnaliticas/2D/k_global_analitica.txt', 'Kglobal')

    # import pdb; pdb.set_trace()  # myMesh.compute_Kglobal_Sparse()
