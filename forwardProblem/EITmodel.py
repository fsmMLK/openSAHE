#!/bin/python
"""
Main EIT model
"""
from multiprocessing import set_start_method

#  https://pythonspeed.com/articles/python-multiprocessing/
if __name__ == '__main__':
    set_start_method("spawn")

import argparse
import os
# -*- coding: utf-8 -*-
import sys
import time

from lxml import etree as ETree

import EITforwardProblem
import FEMmodel
import tools

class EITcoreSolver:
    """
    Main EIT model class
    """

    def __init__(self, confFile  # type: str
                 ):
        """
        Parameters
        ----------
        confFile: str
            configuration file .conf

        """
        self.confFile = os.path.abspath(confFile)
        [self.baseDir, self.filePrefix, _] = tools.splitPath(self.confFile)

        # creates the base output dir
        generalConf = ETree.parse(self.confFile).getroot().xpath('general')[0]
        self.outputBaseDir = os.path.abspath(self.baseDir + tools.getElemValueXpath(generalConf, xpath='outputDir', valType='str')) + '/'
        tools.createDir(self.outputBaseDir)

        # load FEM mesh
        self.femMesh = FEMmodel.FemModel(self.confFile, self.outputBaseDir)
        self.femMesh.loadGmsh()
        # self.femMesh.getDomainElemQuality('lixo_elemQuality.txt')
        # self.femMesh.exportGmsh_RhoElements(self.femMesh.elements, title='Rho 1', iterNbr=0, sufixName='_rho0',mode='new')

        start = time.time()

        self.femMesh.buildKglobal()

        if False:
            tools.saveSparseCOO(self.femMesh.KglobalSp,'lixo_K.txt')

        print('  -> time Kglobal: %f s' % (time.time() - start))


class EITforwardSolver(EITcoreSolver):
    """
    Main EIT forward problem class
    """

    def __init__(self, confFile  # type: str
                 ):
        """
        Parameters
        ----------
        confFile: str
            configuration file .conf
        """
        super().__init__(confFile)

        # create forward problem
        self.forwardProblem = EITforwardProblem.ForwardProblemCore(self.confFile, self.femMesh)

    def solveFrame(self, frame):
        self.forwardProblem.setFrameResistivities(frame)
        self.femMesh.buildKglobal()
        self.forwardProblem.observationModel.update()

        if self.forwardProblem.currentFrame == 0:
            mode = 'new'
        else:
            mode = 'append'

        self.forwardProblem.exportGMSH(title='Forward Problem frame %03d' % frame, iterNbr=frame, sufixName='_forwardProblem', mode=mode,
                                       addDomain=True, addElectrodes=True)

        measurements = self.forwardProblem.solve()

        return measurements

    def solveAll(self):
        for frame in range(self.forwardProblem.nFrames):
            print('\n')
            print('-------------- Forward problem #%03d of %03d --------------' % (frame + 1, self.forwardProblem.nFrames))
            start = time.time()

            measurements = self.solveFrame(frame)

            print('  -> time: %f s' % (time.time() - start))


if __name__ == '__main__':

    if sys.version_info.major == 2:
        sys.stdout.write("Sorry! This program requires Python 3.x\n")
        sys.exit(1)

    parser = argparse.ArgumentParser(description='EIT toolbox.')
    optional = parser._action_groups.pop()  # Edited this line
    required = parser.add_argument_group('required arguments')

    required.add_argument('-i', action="store", dest='confFile', type=str, help='input .conf file. Absolute of relative paths are allowed')
    #optional.add_argument('--optional_arg')
    parser._action_groups.append(optional)  # added this line

    args = parser.parse_args()

    if args.confFile is None:
        print('ERROR: provide an input .conf file. Exiting...')
        sys.exit(1)

    tools.showModulesVersion()

    rootDir = os.getcwd() + '/'

    runForward = ETree.parse(args.confFile).getroot().xpath('forwardProblem')[0]
    if tools.isActiveElement(runForward, xpath='.'):
        print('\n')
        print('====== STARTING FORWARD PROBLEM SOLVER ======')
        EIT_forwardProblem = EITforwardSolver(args.confFile)
        # EIT_forwardProblem.solveFrame(1)
        EIT_forwardProblem.solveAll()

    print('End of EITmodel.py')
    # import pdb; pdb.set_trace()
