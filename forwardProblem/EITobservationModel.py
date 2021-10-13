#!/bin/python
# -*- coding: utf-8 -*-
"""
observation model classes
"""
import time

import numpy as np

from mySparse import myPardisoSparse


class ObservationModelCore:
    """
    core observation model. this class defines a few functions that should be common
    """

    def __init__(self, voltageConfDict, currentConfDict, femMesh, nCoresMKL):
        self.currentConfDict = currentConfDict
        self.voltageConfDict = voltageConfDict
        self.femMesh = femMesh
        self.nCoresMKL = nCoresMKL

        self.KglobalPardiso = None
        self.currMatrix = None
        self.se2diffMatrix = None

        self.update()
        self.calcCurrMatrix()

        if self.voltageConfDict['method'] == 'differential_skip':
            self.calcSEtoDiffMatrix()

        # build diagonal matrix containing 1 if the measurement must be used or not.
        self.calcMeasWeightMatrix()

    def update(self):
        """
        updates observation model. This is necessary when Kglobal change.
        """
        self.KglobalPardiso = myPardisoSparse(self.femMesh.KglobalSp, mtype=11, nCoresPardiso=self.nCoresMKL)

    def calcCurrMatrix(self):
        """
        computes the current matrix, following the information provided in the .conf file.
        """

        if self.currentConfDict['method'] == 'bipolar_skip_full':
            self.nCurrents = self.femMesh.nElectrodes
            self.currMatrix = np.zeros((self.femMesh.nNodes, self.nCurrents))
            self.injectionPairs = np.zeros((self.femMesh.nElectrodes, 2), dtype=int)

            for i in range(self.nCurrents):
                if self.currentConfDict['direction'] == '+-':
                    currPos = i
                    currNeg = (i + self.currentConfDict['skip'] + 1) % self.femMesh.nElectrodes
                else:
                    currPos = (i + self.currentConfDict['skip'] + 1) % self.femMesh.nElectrodes
                    currNeg = i

                self.injectionPairs[i] = np.array([currPos, currNeg])
                self.currMatrix[self.femMesh.electrodeNodes[currPos], i] = self.currentConfDict['value']
                self.currMatrix[self.femMesh.electrodeNodes[currNeg], i] = -self.currentConfDict['value']

        if self.currentConfDict['method'] == 'bipolar_pairs':
            self.nCurrents = len(self.currentConfDict['injectionPairs'])
            self.currMatrix = np.zeros((self.femMesh.nNodes, self.nCurrents))
            self.injectionPairs = self.currentConfDict['injectionPairs']

            for i, pair in enumerate(self.injectionPairs):
                if self.currentConfDict['direction'] == '+-':
                    currPos, currNeg = pair
                else:
                    currNeg, currPos = pair

                self.currMatrix[self.femMesh.electrodeNodes[currPos], i] = self.currentConfDict['value']
                self.currMatrix[self.femMesh.electrodeNodes[currNeg], i] = -self.currentConfDict['value']

        # set the elements of the reference voltage node to zero.
        self.currMatrix[self.femMesh.voltageRefNode, :] = 0

    def calcMeasWeightMatrix(self):
        """
        Find active measurements. defines the number of measurements per injection pair
        """
        matrix = np.ones([self.nCurrents, self.femMesh.nElectrodes ])

        if self.voltageConfDict['removeInjectingPair']:
            if self.voltageConfDict['method'] == 'single_ended':
                for i, pair in enumerate(self.injectionPairs):
                    matrix[i,pair] = 0.0

            if self.voltageConfDict['method'] == 'differential_skip':
                for i, currPair in enumerate(self.injectionPairs):
                    c1, c2 = currPair

                    rows2,_ = np.where(self.measurementPairs == c2)                    # find measurements that involve electrode c1
                    rows1,_ = np.where(self.measurementPairs == c1)

                    matrix[i,rows1] = 0.0
                    matrix[i,rows2] = 0.0

        self.measWeightMatrix = matrix.flatten('C')

        self.activeMeasurementPositions = np.where(self.measWeightMatrix>0)[0]

    def calcSEtoDiffMatrix(self):
        """
        computes the matrix that converts single-ended to differential measurements, following the
        information provided in the .conf file.
        """
        self.se2diffMatrix = np.zeros((self.femMesh.nElectrodes, self.femMesh.nElectrodes),dtype=float)
        self.measurementPairs = np.zeros((self.femMesh.nElectrodes, 2), dtype=int)

        if self.voltageConfDict['method'] == 'differential_skip':
            for i in range(self.femMesh.nElectrodes):
                if self.voltageConfDict['direction'] == '+-':
                    voltPos = i
                    voltNeg = (i + self.voltageConfDict['skip'] + 1) % self.femMesh.nElectrodes
                else:
                    voltPos = (i + self.voltageConfDict['skip'] + 1) % self.femMesh.nElectrodes
                    voltNeg = i


                # measurements
                self.measurementPairs[i] = np.array([voltPos, voltNeg])
                self.se2diffMatrix[i, voltPos] = 1
                self.se2diffMatrix[i, voltNeg] = -1

    def forwardProblemSp(self, fileName=None, append=False, singleEnded=False):
        """
        solves the forward problem using sparse version of the FEM matrix.

        Parameters
        ----------
        fileName: str, optional
            output file name. If None (default), then no file is saved

        append: bool
            append file. If 'False' the file is overwritten

        singleEnded : bool
            If true, the results will be single ended with respect to the reference node. if False, then the result
            will follow the configuration of the .conf file.

        Returns
        -------
        measurements : 1d numpy array
            electrode voltages of all current patterns. Thsi vector contains measurements of active electrodes only

        """

        nodalVoltages = self.KglobalPardiso.solve(self.currMatrix)

        # dense Kglobal matrix
        # nodalVoltages = scipyLinalg.solve(self.femMesh.Kglobal, self.currMatrix, assume_a='pos', check_finite=False)

        # extract measurements from electrodes virtual nodes
        electrodeVoltages = nodalVoltages[self.femMesh.electrodeNodes, :]

        if not singleEnded and self.voltageConfDict['method'] == 'differential_skip':
            electrodeVoltages = np.matmul(self.se2diffMatrix, electrodeVoltages)

        # vectorize array columnwise
        electrodeVoltages = electrodeVoltages.flatten('F')

        # extracts only valid measurement electrode
        electrodeVoltages = electrodeVoltages[self.activeMeasurementPositions]

        if fileName is not None:
            if append:
                with open(fileName, 'ab') as f:
                    np.savetxt(f, electrodeVoltages.reshape(1, electrodeVoltages.shape[0]))
            else:
                with open(fileName, 'w') as f:
                    f.write('# headerSize=%d\n' % 8)
                    f.write('# nElectrodes=%d\n' % self.femMesh.nElectrodes)
                    f.write('# nInjections=%d\n' % self.nCurrents)
                    f.write('# currentPattern=%s\n' % self.currentConfDict['method'])
                    f.write('# voltagePattern=%s\n' % self.voltageConfDict['method'])
                    f.write('# currentValue_A=%f\n' % self.currentConfDict['value'])
                    f.write('# ignoreInjectingElectrodes=%s\n' %self.voltageConfDict['removeInjectingPair'])
                    f.write('# measurementArray=')

                with open(fileName, 'ab') as f:
                    np.savetxt(f, self.measWeightMatrix > 0 ,fmt='%d',newline=' ')
                    f.write(bytes('\n','utf8'))
                    np.savetxt(f, electrodeVoltages.reshape(1, electrodeVoltages.shape[0]))

        return electrodeVoltages

    def forwardProblemSp_allNodes(self, fileName=None, append=False):
        """
        solves the forward problem using sparse version of the FEM matrix. Solves for all nodes. DOES NOT COMPUTE DIFFERENTIAL MEASUREMENTS!

        Parameters
        ----------
        fileName: str, optional
            output file name. If None (default), then no file is saved. The file has the following format:
                    cols 1,2,3: x,y,z coordinates of the node
                    cols 4,5,....  solutions of the forward problem

            The file will not contain the voltage of the virtual nodes of the electrodes.

        append: bool
            append file. If 'False' the file is overwritten

        Returns
        -------
        measurements : 1d numpy array
            voltages of all current patterns

        """

        nodalVoltages = self.KglobalPardiso.solve(self.currMatrix)

        # removes the virtual nodes of the electrodes
        nodalVoltages = nodalVoltages[:-self.femMesh.nElectrodes, :]

        coords = self.femMesh.nodeCoords

        data = np.concatenate([coords, nodalVoltages], axis=1)
        if fileName is not None:
            if append:
                with open(fileName, 'ab') as f:
                    np.savetxt(f, data)
            else:
                np.savetxt(fileName, data)

        return nodalVoltages


class LinearObservationModel(ObservationModelCore):
    """
    linearized observation model class
    """

    def __init__(self, voltageConfDict, currentConfDict, femMesh, stateElements, nCoresMKL):
        super().__init__(voltageConfDict, currentConfDict, femMesh, nCoresMKL)

        self.stateElements = stateElements
        self.jacobian = None

    def calcJacobian(self):
        """
        computes the jacobian matrix
        """
        start = time.time()
        print('Computing Jacobian...')

        # compute vCalc = [K]^-1 * C
        vCalc = self.KglobalPardiso.solve(self.currMatrix)

        # print('  - inverting global matrix...')
        invK = self.KglobalPardiso.invert()

        # print('  - computing columns of the Jacobian...')
        self.jacobian = np.zeros([self.femMesh.nElectrodes * self.nCurrents, len(self.stateElements)])

        for idx, e in enumerate(self.stateElements):
            if idx % 1000 == 999:
                print("  -> element %d of %d" % (idx + 1, len(self.stateElements)))
            elem = self.femMesh.elements[e]
            # compute temp = [del(K)/del(rho_k)] * [K]^-1 * C = [del(K)/del(rho_k)] * [vCalc]
            if elem.propertiesDict['isElectrode']:
                temp = -1.0 / (elem.rhoT ** 2) * np.matmul(elem.Kgeom, vCalc[elem.connectivity, :])
            else:
                temp = -1.0 / (elem.rho ** 2) * np.matmul(elem.Kgeom, vCalc[elem.connectivity, :])

            # compute tempJacobian = -[K]^-1 * [del(K)/del(rho_k)] * [K]^-1 * C = -[K]^-1 * [temp]
            tempJacobian = np.matmul(-invK[self.femMesh.electrodeNodes, :][:, elem.connectivity], temp)

            if self.voltageConfDict['method'] == 'differential_skip':
                tempJacobian = np.matmul(self.se2diffMatrix, tempJacobian)

            self.jacobian[:, idx] = np.reshape(tempJacobian, self.femMesh.nElectrodes * self.nCurrents, order='F')

        # extracts only valid measurement electrode
        self.jacobian = self.jacobian[self.activeMeasurementPositions]

        print('  -> time Jacobian: %f s' % (time.time() - start))
