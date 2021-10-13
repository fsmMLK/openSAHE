#!/bin/python
"""
forward problem solver main class
"""
# -*- coding: utf-8 -*-

import os

import numpy as np
from lxml import etree as ETree
from scipy import linalg as scipyLinalg

import anatomicalAtlas
import EITobservationModel
import tools


class EITbaseProblemCore:
    """
    Base class for both forward and inverse problems
    """
    def __init__(self, confFile, femMesh):
        self.confFile = confFile
        [self.baseDir, self.filePrefix, _] = tools.splitPath(self.confFile)
        self.femMesh = femMesh
        self.outputBaseDir = self.femMesh.outputBaseDir

        # set number of cores
        generalConf = ETree.parse(self.confFile).getroot().xpath('general')[0]
        self.nCoresMKL = tools.getElemValueXpath(generalConf, xpath='nCoresMKL', valType='int')
        if self.nCoresMKL == 0:
            self.nCoresMKL = tools.nCores

        self.currentConfDict = {}
        self.voltageConfDict = {}
        self.observationModel = None

    def _setCurrPattern(self):
        """
        set the current injection pattern
        """

        currentConf = ETree.parse(self.confFile).getroot().xpath('current')[0]

        self.currentConfDict['method'] = tools.getElemValueXpath(currentConf, xpath='method', valType='str').lower()

        if self.currentConfDict['method'] not in ['bipolar_skip_full', 'bipolar_pairs']:
            print('ERROR: Current pattern type not recognized:  %s' % pattern)
            return

        self.currentConfDict['direction'] = tools.getElemValueXpath(currentConf, xpath='direction', valType='str')

        if self.currentConfDict['method'] == 'bipolar_skip_full':
            self.currentConfDict['skip'] = tools.getElemValueXpath(currentConf, xpath='bipolarSkipFullOpts/skip', valType='int')

        if self.currentConfDict['method'] == 'bipolar_pairs':
            # subtracts 1 because electrode numbers start from 0
            self.currentConfDict['injectionPairs'] = np.array(tools.getElemValueXpath(currentConf, xpath='bipolarPairsOpts/injectionPairs',
                                                                         valType='list_list_int'),dtype=int)-1
        # load current value and converts to Ampere if
        value = tools.getElemValueXpath(currentConf, xpath='value', valType='float')
        currUnit = tools.getElemAttrXpath(currentConf, xpath='value', attrName='unit', attrType='str')
        if currUnit == 'mA':
            value *= 0.001

        self.currentConfDict['value'] = value

        self.freq_Hz = tools.getElemValueXpath(currentConf, xpath='frequency_Hz', valType='float')


    def _setVoltPattern(self):
        """
        set the voltage pattern
        """
        voltageConf = ETree.parse(self.confFile).getroot().xpath('voltage')[0]

        self.voltageConfDict['method'] = tools.getElemValueXpath(voltageConf, xpath='method', valType='str').lower()

        if self.voltageConfDict['method'] not in ['single_ended', 'differential_skip']:
            print('ERROR: voltage pattern type not recognized:  %s' % pattern)
            return

        if self.voltageConfDict['method'] == 'differential_skip':
            self.voltageConfDict['direction'] = tools.getElemValueXpath(voltageConf, xpath='diffSkipOpts/direction', valType='str')
            self.voltageConfDict['skip'] = tools.getElemValueXpath(voltageConf, xpath='diffSkipOpts/skip', valType='int')

        self.voltageConfDict['removeInjectingPair'] = tools.getElemValueXpath(voltageConf, xpath='removeInjectingPair', valType='bool')



class ForwardProblemCore(EITbaseProblemCore):
    """
    forward problem solver core class.
    """

    def __init__(self, confFile, femMesh):
        super().__init__(confFile, femMesh)
        self.confForward = ETree.parse(self.confFile).getroot().xpath('forwardProblem')[0]

        self.f = 0
        self.nFrames = tools.getElemValueXpath(self.confForward, xpath='numFrames', valType='int')
        self.frameRate = tools.getElemValueXpath(self.confForward, xpath='frameRate_Hz', valType='float')
        self.framePeriod = 1.0/self.frameRate

        self.outputFile = None
        self.outputIsBinary = None

        self._setOutputFile()
        print('  -> Creating observation model...')
        self.createObservationModel()

        if tools.isActiveElement(self.confForward, xpath='nodalVoltages'):
            self.saveNodalVoltages = True
            self.exportGmshNodalVoltages = tools.getElemValueXpath(self.confForward, xpath='nodalVoltages/exportGmsh', valType='bool')
            self.fileNodalVoltages = os.path.abspath(
              self.outputBaseDir + tools.getElemValueXpath(self.confForward, xpath='nodalVoltages/file', valType='str'))
        else:
            self.saveNodalVoltages = False

        # load Atlas
        confAtlas = ETree.parse(self.confFile).getroot().xpath('AnatomicalAtlas')[0]
        if tools.isActiveElement(confAtlas, xpath='.'):
            self.atlasCore = anatomicalAtlas.AnatomicalAtlasCore(confAtlas,self.freq_Hz, self.femMesh)
        else:
            self.atlasCore = None

    def _setOutputFile(self):
        """
        Set output file
        """
        self.outputFile = os.path.abspath(
            self.outputBaseDir + tools.getElemValueXpath(self.confForward, xpath='measurementOutput/file', valType='str'))
        self.outputIsBinary = tools.getElemAttrXpath(self.confForward, xpath='measurementOutput/file', attrName='binary', attrType='bool')

    def createObservationModel(self):
        """
        build the observation model used in the solver
        """
        self._setCurrPattern()
        self._setVoltPattern()
        self.observationModel = EITobservationModel.ObservationModelCore(self.voltageConfDict, self.currentConfDict, self.femMesh, self.nCoresMKL)


    def exportGMSH(self, title='Forward Problem', iterNbr=0, mode='new', sufixName='_forwardProblem', addDomain=True, addElectrodes=False):
        """
        Export the solution to gmsh

        Parameters
        ----------
        title: string
            title of the view in Gmsh
        iterNbr: int
            Number of the iteration
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
        sufixName: string
            suffi name of the file
        addDomain: bool
            include elements of the domain
        addElectrodes: bool
            include electrodes

        """

        listElements = []

        if addDomain:
            listElements += [elem for elem in self.femMesh.elements if not elem.propertiesDict['isElectrode']]
        if addElectrodes:
            listElements += [elem for elem in self.femMesh.elements if elem.propertiesDict['isElectrode']]

        self.femMesh.exportGmsh_RhoElements(listElements, title, iterNbr, sufixName, mode)

    def setFrameResistivitiesManual(self, frameNbr, elementVector, rhoValues):
        """
        set the resistivity of the domain manually, i.e., it does not follow the .conf instructions. Preferebly use setFrameResistivities() method
        Parameters
        ----------
        frameNbr: number of the frame
        elementNumber: numpy array of mesh elements
            number of the elements. if an element is not present here, then the resistivity of the element is that present in the FEMmodel
            segment of this file. usually good to use for the electrodes
        rhoValues: numpy array
            resistivity values. The array must contain the resistivites of all elements of the mesh, except the electrodes
        """
        self.currentFrame = frameNbr
        destinationElementNumbers = [elem.number for elem in elementVector]
        self.femMesh.setResistivities(destinationElementNumbers, rhoValues)

    def setFrameResistivities(self, frameNbr):
        """
        set the resistivity of the domain
        Parameters
        ----------
        frameNbr: number of the frame
        """
        self.currentFrame = frameNbr
        self.currentTime = frameNbr*self.framePeriod

        regions = self.confForward.xpath('regionResistivities')[0]
        for region in regions.iter('region'):
            if tools.isActiveElement(region, xpath='.'):
                meshTagList = tools.getElemValueXpath(region, xpath='meshTag', valType='list_int')
                typeRegion = tools.getElemValueXpath(region, xpath='type', valType='str').lower()
                if typeRegion == 'uniform':
                    rhoList = tools.getElemValueXpath(region, xpath='uniformRho', valType='list_float')

                    if self.currentFrame >= len(rhoList):  # uses the last value of rhoList if frameNbr is larger and the length of rhoList
                        RhoValue = rhoList[-1]
                    else:
                        RhoValue = rhoList[self.currentFrame]

                    self.femMesh.setMeshTagResistivity(meshTagList, RhoValue)

                if typeRegion == 'file':
                    fileName = os.path.abspath(self.baseDir + tools.getElemValueXpath(region, xpath='file', valType='str'))
                    nLinesFile = tools.getNumberOfLines(fileName)

                    RhoValues = tools.readNthLineData(filePath=fileName, lineNbr=min(frameNbr, nLinesFile), separator=' ')

                    destinationElements = self.femMesh.getElementsByMeshTag(meshTagList)
                    destinationElementNumbers = [elem.number for elem in destinationElements]
                    self.femMesh.setResistivities(destinationElementNumbers, RhoValues)

                if typeRegion == 'anatomical_atlas':
                    if self.atlasCore is None:
                        print('ERROR: atlas core not loaded...')
                        quit()

                    if False:
                        nbr=[]
                        rho=[]
                        for p in self.atlasCore.vascularTerritories:
                            nbr.append(p['elemNbr'])
                            rho.append(p['territoryID'])

                        self.femMesh.setResistivities(nbr, rho)
                        return


                    destinationElements = self.femMesh.getElementsByMeshTag(meshTagList)
                    destinationElementNumbers = [elem.number for elem in destinationElements]

                    sampleType = tools.getElemValueXpath(region, xpath='sampleType', valType='str').lower()

                    # the atlas must have at least the static component. therefore I am checking for this component
                    if self.atlasCore.components['static']['avg'] is None:
                        if sampleType.lower() == 'average':
                            self.atlasCore.interpolateStatistics(destinationElements, interpAVG=True, interpCOVK=False, avg_FileName=None,covK_FileName=None)
                        if sampleType.lower() == 'sample':
                            self.atlasCore.interpolateStatistics(destinationElements, interpAVG=True, interpCOVK=True, avg_FileName=None,covK_FileName=None)

                    if self.atlasCore.useDynamicAtlas:
                        timeNormalized = self.currentTime * 1.0/self.atlasCore.vesselDynamicModel.cardiacRate
                    else:
                        timeNormalized = 0.0

                    self.atlasCore.consolidateStatistics(timeNormalized)

                    if sampleType == 'sample':
                        averageOnly = False
                    else:
                        averageOnly = True

                    rhoVals = self.atlasCore.generateSamples(nSamples=1, averageOnly=averageOnly, fileName=None, ensurePositiveRho=True, rhoMinLmit=0.0002)

                    self.femMesh.setResistivities(destinationElementNumbers, rhoVals)

        for obj in self.confForward.xpath('objects')[0]:
            if tools.isActiveElement(obj, xpath='.'):
                typeObj = tools.getElemValueXpath(obj, xpath='type', valType='str').lower()
                if typeObj == 'sphere':
                    meshTagList = tools.getElemValueXpath(obj, xpath='regionTags', valType='list_int')
                    rhoList = tools.getElemValueXpath(obj, xpath='rho', valType='list_float')

                    if self.currentFrame >= len(rhoList):  # uses the last value of rhoList if frameNbr is larger and the length of rhoList
                        RhoValue = rhoList[-1]
                    else:
                        RhoValue = rhoList[self.currentFrame]

                    if RhoValue > 0:
                        # load coordinates and convert to metres
                        center = np.array(tools.getElemValueXpath(obj, xpath='center', valType='list_float'))
                        radius = tools.getElemValueXpath(obj, xpath='radius', valType='float')
                        radius = tools.convToMetre(radius, tools.getElemAttrXpath(obj, xpath='center', attrName='unit', attrType='str'))
                        center = tools.convToMetre(center, tools.getElemAttrXpath(obj, xpath='radius', attrName='unit', attrType='str'))

                        for elem in self.femMesh.elements:
                            if elem.propertiesDict['regionTag'] in meshTagList:
                                if scipyLinalg.norm(elem.centroid - center) < radius:
                                    if elem.propertiesDict['isElectrode']:
                                        elem.setRhoT(RhoValue)
                                    else:
                                        elem.setRho(RhoValue)

    def solve(self):
        """
        solve the forward problem. This function will use the current mesh resistivity. See 'setFrameResistivities' method on how to
        update mesh resistivity.

        Returns
        -------
        measurements : 1d numpy array
            electrode voltages of all current patterns

        """

        if self.currentFrame == 0:
            appendFile = False
            mode = 'new'
        else:
            appendFile = True
            mode = 'append'

        if self.saveNodalVoltages:
            voltages = self.observationModel.forwardProblemSp_allNodes(fileName=self.fileNodalVoltages, append=appendFile)
            if self.exportGmshNodalVoltages:
                self.observationModel.femMesh.exportGmsh_NodalVoltages(voltages, title='Solution', iterNbr=0, sufixName='_outputNodalVoltages', mode=mode)

        if self.outputIsBinary:
            print('ERROR: binary format of measurement file is not implemented.')
            return None
        else:
            return self.observationModel.forwardProblemSp(self.outputFile, append=appendFile,
                                                          singleEnded=self.voltageConfDict['method'] == 'single_ended')
