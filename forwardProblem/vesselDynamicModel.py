#!/bin/python
"""
vessel's dynamic model solver
"""

# -*- coding: utf-8 -*-

import argparse
import glob
import os
import shutil
import sys
from distutils import dir_util

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as scipyInterp
import yaml
from lxml import etree as ETree

import tissuePropCalculator
import tools

# loads Qt5Agg only in my laptop
if tools.isFeLap():
    matplotlib.use('Qt5Agg')

class dynamicModel:

    def __init__(self, confDynamicModel, baseDir):

        print('Initiating openBF solver...')
        # load configurations
        self.confDynamicModel = confDynamicModel

        # creates the base output dir
        self.baseDir = baseDir
        self.outputBaseDir = os.path.abspath(
          self.baseDir + tools.getElemValueXpath(self.confDynamicModel, xpath='openBF/outputDir', valType='str')) + '/'
        tools.createDir(self.outputBaseDir)

        # openBF data
        inputFile = os.path.abspath(self.baseDir + tools.getElemValueXpath(self.confDynamicModel, xpath='openBF/inputfile', valType='str'))
        self.inputFileDir = os.path.dirname(inputFile) + '/'
        self.inputFile = os.path.basename(inputFile)

        # openBf inlet boundary conditions
        self.cardiacRate = tools.getElemValueXpath(self.confDynamicModel, xpath='openBF/inputFlow/cardiacRate', valType='float')
        unit = tools.getElemAttrXpath(self.confDynamicModel, xpath='openBF/inputFlow/cardiacRate', attrName='unit', attrType='str')
        if unit.lower() == 'bpm':
            self.heartRate *= 60
        self.peakFlow = tools.getElemValueXpath(self.confDynamicModel, xpath='openBF/inputFlow/peakFlow', valType='float')
        self.systolicEjectiontime = tools.getElemValueXpath(self.confDynamicModel, xpath='openBF/inputFlow/systolicEjectiontimePercentage',
                                                            valType='float') / self.cardiacRate
        self.inputSamplingPeriod = tools.getElemValueXpath(self.confDynamicModel, xpath='openBF/inputFlow/samplingPeriod', valType='float')

        # read input parameters
        self.hematocrit = tools.getElemValueXpath(self.confDynamicModel, xpath='visserModel/hematocrit', valType='float')

        if tools.isActiveElement(self.confDynamicModel, xpath='visserModel/maxChangePercentage'):
            self.maxChangePercentage = tools.getElemValueXpath(self.confDynamicModel, xpath='visserModel/maxChangePercentage', valType='float')
        else:
            self.maxChangePercentage = None

        self.electricalPropErrorPercentage = tools.getElemValueXpath(self.confDynamicModel, xpath='visserModel/electricalPropErrorPercentage', valType='float')

        # read input file contents
        with open(inputFile) as f:
            self.openBFinputData = yaml.load(f, Loader=yaml.FullLoader)

        self.outputBaseDir += self.openBFinputData['project name'] + '_results/'  # do not forget the last / !!!

        self.loadVesselData()

    def loadVesselData(self):
        """
        load information of the vessels
        """

        # list head arteries  (ID and label)
        brainArteryLeft = {27: ('PCA1', 'L'), 32: ('PCA2', 'L'), 23: ('MCA', 'L'), 25: ('ACA1', 'L'), 29: ('ACA2', 'L'), 10: ('ECA', 'L'),
                           36: ('SCA', 'L')}
        brainArteryRight = {28: ('PCA1', 'R'), 33: ('PCA2', 'R'), 24: ('MCA', 'R'), 26: ('ACA1', 'R'), 30: ('ACA2', 'R'), 13: ('ECA', 'R'),
                            37: ('SCA', 'R')}

        brainArteries = brainArteryLeft.copy()  # start with x's keys and values
        brainArteries.update(brainArteryRight)

        self.vesselData = {}

        for data in self.openBFinputData['network']:
            R0 = float(data['R0'])
            label = data['label']
            id = int(label.split('_')[1])

            if id in brainArteries:
                label = brainArteries[id][0]
                side = brainArteries[id][1]
                self.vesselData[id] = {'label': label.upper(), 'side': side.upper(), 'R0': R0}

    def createInputFlow(self):
        """
        creates inlet flow (sinusoidal) following
        'Modelling the circle of Willis to assess the effects of anatomical variations and occlusions on cerebral flows',
        J Alastruey et al
        2007

        Returns
        -------

        """
        showPlot = False
        inletFile = self.inputFileDir + 'heart_inlet.dat'

        cardiacPeriod = 1.0 / self.cardiacRate

        if cardiacPeriod > 1.0:
            print('ATENCAO! verificar o output cardiaco!! os valores de Qmax foram ajustados para frequencia cardiaca de 1bpm! Ver artigo '
                  '\'Modelling the circle of Willis to assess the effects of anatomical variations and occlusions on cerebral flows\'')
            sys.exit()

        time = np.linspace(0, cardiacPeriod, int(1 / self.inputSamplingPeriod) + 1)

        # creates the sine function
        input = self.peakFlow * np.sin(time * np.pi / self.systolicEjectiontime)

        # removes the signal to keep just the first positive bump
        # epsilon is imporatante bc input 0 causes numerical problems.
        epsilon = 1e-10
        input[time > self.systolicEjectiontime] = epsilon
        input[0] = epsilon

        if showPlot:
            fig, ax = plt.subplots(1, 1)
            ax.grid(True)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Flow rate (m^3/s)')
            ax.plot(time, input, '.-')
            plt.show()

        data = np.column_stack((time, input))
        np.savetxt(inletFile, data)

    def run_openBF(self,skipRun=False):
        # optional arguments:
        #  -v, --verbose    Print STDOUT - default false
        #  -f, --out_files  Save complete results story rather than only the
        #                   last cardiac cycle
        #  -h, --help       show this help message and exit

        mainJl = self.inputFileDir + 'main.jl'
        shutil.copy(tools.openBFDir + 'main.jl', mainJl)
        os.chmod(mainJl, 0o770)

        # julia must run from the input file directory
        currDir = os.getcwd()
        os.chdir(self.inputFileDir)

        if not skipRun:
            self.createInputFlow()

            options = '-v'
            command = [tools.juliaExe, mainJl, self.inputFile, options]
            os.system(' '.join(command))

            tempOutputDir = os.path.abspath(os.path.join(self.inputFileDir, self.openBFinputData['results folder'])) + '/'
            dir_util.copy_tree(tempOutputDir, self.outputBaseDir)
            dir_util.remove_tree(tempOutputDir)
        else:
            print('ATENTION: run_openBF SKIPPING THE SOLVER!!!!!!')
            print('ATENTION: run_openBF SKIPPING THE SOLVER!!!!!!')
            print('ATENTION: run_openBF SKIPPING THE SOLVER!!!!!!')
            print('ATENTION: run_openBF SKIPPING THE SOLVER!!!!!!')

        os.chdir(currDir)

    def loadSolution(self, resample=False, resapleFreq_Hz=10.0):

        for signalType in ['pressure','velocity','area','flow_rate','wave_speed']:

            for id, data in self.vesselData.items():
                time, signal = self._loadSolutionVessel(id, signal=signalType, vesselNode=3, resample=resample, resapleFreq_Hz=resapleFreq_Hz)
                data[signalType] = signal
                data['time'] = time

                if signalType.lower() == 'pressure':
                    # pressure in kPa
                    data[signalType] /= 1000.0

    def getPropVessel(self, vesselName, side, property, timeNormalized):
        """
        interpolates rho signal of a given vessel at specitif normalized time (time between 0 and 1.0)
        Parameters
        ----------
        vesselName: ACA, MCA, PCA, ECA, SCA
        side: L, R
        propertyName:
                'area', 'flow_rate', 'wave_speed'
                'resistivity', 'resistivity_error', 'delta_resistivity_percentage'
                'conductivity', 'conductivity_error', 'delta_conductivity_percentage'
                'relpermittivity', 'relpermittivity_error', 'delta_relpermittivity_percentage'

                units: A: m^2, flow_rate: m^3/s, waveSpeed: ?,
        timeNormalized: between 0.0 and 1.0
        """
        for id, data in self.vesselData.items():
            time = timeNormalized * (data['time'][-1] - data['time'][0])
            if (vesselName.upper() in data['label'].upper()) and (side.upper() in data['side'].upper()):
                # interpolates value
                resampler = scipyInterp.interp1d(data['time'], data[property.lower()], kind='cubic', copy=True, bounds_error=False,
                                                 fill_value='extrapolate')

                # https://stackoverflow.com/questions/37592643/scipy-interpolate-returns-a-dimensionless-array
                val = resampler(time)[()]
                return [time, val]

        # if not found
        print('WARNING: VESSEL NOT FOUND! RETURNING PROPERTY OF STILL BLOOD')
        if property.lower() in ['conductivity','resistivity','relpermittivity']:
            return [time, self.propV0]
        if property.lower() in ['conductivity_error','resistivity_error','relpermittivity_error']:
            return [time, self.propV0*self.electricalPropErrorPercentage]
        if property.lower() in ['flow_rate']:
            return [time, 0.0]
        if property.lower() in ['area', 'wave_speed']:
            return [time, None]


    def plotSignals(self, fileName=None, signalDictLabel='velocity', title='Velocity (m/s)',addLegend=False,scalingFactor=None):
        """
        plot the signals in 4 rows:ACA, MCA and PCA, ECA
        """
        fig, axList = plt.subplots(5, 1, dpi=300, figsize=(5, 10))
        # row 0
        axList[0].tick_params(labelbottom=False)
        axList[0].grid(True)
        axList[0].set_title(title)
        # row 1
        axList[1].tick_params(labelbottom=False)
        axList[1].grid(True)
        # row 2
        axList[2].tick_params(labelbottom=False)
        axList[2].grid(True)
        # row 3
        axList[3].tick_params(labelbottom=False)
        axList[3].grid(True)
        # row 4
        axList[4].set_xlabel('Time (s)')
        axList[4].grid(True)

        # col 0
        if False:
            axList[0].set_ylabel('ACA ' + yLabel)
            axList[1].set_ylabel('MCA ' + yLabel)
            axList[2].set_ylabel('PCA ' + yLabel)
            axList[3].set_ylabel('ECA ' + yLabel)
            axList[4].set_ylabel('SCA ' + yLabel)

        maxVal = -10000
        minVal = 10000
        for id, value in self.vesselData.items():
            timeVec = value['time']

            if scalingFactor is not None:
                signal = value[signalDictLabel]*scalingFactor
            else:
                signal = value[signalDictLabel]

            maxVal = max(maxVal, np.amax(signal))
            minVal = min(minVal, np.amin(signal))

            if value['side'].upper() == 'L':
                color = 'r'
                side = 'left'
            if value['side'].upper() == 'R':
                color = 'b'
                side = 'right'

            if 'ACA' in value['label']:
                row = 0
            if 'MCA' in value['label']:
                row = 1
            if 'PCA' in value['label']:
                row = 2
            if 'ECA' in value['label']:
                row = 3
            if 'SCA' in value['label']:
                row = 4

            axList[row].plot(timeVec, signal, label=value['label'] + ' ' + side, linewidth=1)

            if addLegend:
                axList[row].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # set ylim
        for ax in axList.flatten():
            ax.set_xlim(0, 1)
            if signalDictLabel == 'velocity':
                ax.set_ylim(tools.roundDown(minVal, digits=1), tools.roundUp(maxVal, digits=1))
                ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f'))
            if signalDictLabel == 'resistivity':
                ax.set_ylim(tools.roundDown(minVal, digits=2), tools.roundUp(maxVal, digits=2))
                ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
            if signalDictLabel == 'pressure':
                ax.set_ylim(tools.roundDown(minVal, digits=0), tools.roundUp(maxVal, digits=0))
                ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f'))
            if signalDictLabel == 'flow_rate':
                ax.set_ylim(tools.roundDown(minVal, digits=0), tools.roundUp(maxVal, digits=0))
                ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f'))
            if 'delta' in signalDictLabel:
                ax.set_ylim(tools.roundDown(minVal, digits=0), tools.roundUp(maxVal, digits=0))
                ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f'))

        if fileName is not None:
            plt.rcParams.update({'font.size': 15})
            fig.set_figheight(10)
            fig.set_figwidth(4)
            plt.savefig(fileName, dpi=300, facecolor='w', edgecolor='w', format=None, transparent=False, bbox_inches='tight', pad_inches=0.05,
                        metadata=None)
        else:
            plt.show()

        plt.close(fig)

    def _loadSolutionVessel(self, arteryID, signal='velocity', vesselNode=3, resample=False, resapleFreq_Hz=10.0):
        """
        load the solution of one vessel, given its ID number.
        Parameters
        ----------
        arteryID: int
            number ID of the vessel
        signal: string
            type of signal: valid values: 'area','velocity','pressure','flow_rate','wave_speed'
        vesselNode

        Returns
        -------
        [time,signal]

        """

        showPlot = False

        if showPlot:
            fig, ax = plt.subplots(1, 1)
            ax.grid(True)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Velocity (m/s)')

        if signal.lower() == 'area':
            sigType = 'A'
        if signal.lower() == 'wave_speed':
            sigType = 'c'
        if signal.lower() == 'pressure':
            sigType = 'P'
        if signal.lower() == 'flow_rate':
            sigType = 'Q'
        if signal.lower() == 'velocity':
            sigType = 'u'

        file = glob.glob(self.outputBaseDir + '*_%0.2d_%s.last' % (arteryID, sigType))[0]

        data = np.loadtxt(file, delimiter=' ')
        time = data[:, 0]
        time -= time[0]
        signal = data[:, vesselNode]

        if showPlot:
            ax.plot(time, signal, '-r.')

        if resample:
            resampler = scipyInterp.interp1d(time, signal, kind='cubic', copy=True, bounds_error=False, fill_value='extrapolate')

            nPoints = int(round(time[-1] - time[0]) * resapleFreq_Hz) + 1
            time = np.arange(time[0], 1 / resapleFreq_Hz * nPoints, 1 / resapleFreq_Hz)
            signal = resampler(time)

        if showPlot:
            ax.plot(time, signal, '-k.')
            plt.show()
            plt.close(fig)

        return [time, signal]

    def getStillBloodProp(self, freq_Hz, property='resistivity'):
        """
        return the electrical property of still blood (v=0)
        freq_Hz: frequency in hertz to determine electrical property of still blood
        propertyName: 'resistivity', 'conductivity', 'relPermittivity'
        """
        tissueProp = tissuePropCalculator.TissueCalculator()
        # property of still blood
        if property.lower() == 'conductivity':
            propV0, _ = tissueProp.getConductivity(tissueName='Blood', frequency_Hz=freq_Hz, uncertainty_Perc=self.electricalPropErrorPercentage)
        if property.lower() == 'resistivity':
            propV0, _ = tissueProp.getResistivity(tissueName='Blood', frequency_Hz=freq_Hz, uncertainty_Perc=self.electricalPropErrorPercentage)
        if property.lower() == 'relpermittivity':
            propV0, _ = tissueProp.getRelPermittivity(tissueName='Blood', frequency_Hz=freq_Hz, uncertainty_Perc=self.electricalPropErrorPercentage)
        return propV0

    def convert_Vel2electricalProp(self, freq_Hz, property='resistivity'):
        """
        freq_Hz: frequency in hertz to determine electrical property of still blood
        propertyName: 'resistivity', 'conductivity', 'relPermittivity'
        """
        self.propV0 = self.getStillBloodProp(freq_Hz, property='resistivity')

        for id, data in self.vesselData.items():
            # equation 4, visser's Thesis
            # visser's model <v> is the average velocity!

            # openBF's velocity u is also the average velocity!

            reducedVel = np.abs(data['velocity'] / data['R0'])

            if property.lower() == 'resistivity':
                deltaProp_normalized = -0.45 * self.hematocrit * (1.0 - np.exp(-0.26 * np.power(reducedVel, 0.39)))
            if property.lower() == 'conductivity' or property.lower() == 'relpermittivity':
                deltaProp_normalized = 0.58 * self.hematocrit * (1.0 - np.exp(-0.20 * np.power(reducedVel, 0.41)))

            if self.maxChangePercentage is not None:
                # this accomodates experimental results (0.15) smaller than predicted by Vissel's model (0.25). See anatomical atlas article
                deltaProp_normalized *= self.maxChangePercentage/0.25

            data['delta_' + property.lower()+'_percentage'] = deltaProp_normalized*100
            data[property.lower()] = self.propV0 + deltaProp_normalized * self.propV0
            data[property.lower() + '_error'] = data[property.lower()] * self.electricalPropErrorPercentage

if __name__ == '__main__':

    if sys.version_info.major == 2:
        sys.stdout.write("Sorry! This program requires Python 3.x\n")
        sys.exit(1)

    parser = argparse.ArgumentParser(description='openBF solver')
    optional = parser._action_groups.pop()  # Edited this line
    required = parser.add_argument_group('required arguments')

    required.add_argument('-i', action="store", dest='confFile', type=str, help='input .conf file. Absolute of relative paths are allowed')
    # optional.add_argument('--optional_arg')
    parser._action_groups.append(optional)

    args = parser.parse_args()

    if args.confFile is None:
        print('ERROR: provide an input .conf file. Exiting...')
        sys.exit(1)

    tools.showModulesVersion()

    rootDir = os.getcwd() + '/'

    confVessels = ETree.parse(args.confFile).getroot().xpath('AnatomicalAtlas/dynamicAtlas')[0]
    [baseDir, _, _] = tools.splitPath(args.confFile)
    solver = dynamicModel(confVessels, baseDir)
    solver.run_openBF(skipRun=False)
    solver.loadSolution(resample=False, resapleFreq_Hz=100.0)

    freq_Hz = 1000

    solver.convert_Vel2electricalProp(freq_Hz, property='resistivity')

    tVec = np.linspace(0.0, 1.0, 1000)
    x = [solver.getPropVessel(vesselName='ACA', side='L', property='resistivity', timeNormalized=t)[1] for t in tVec]

    # fig = plt.figure()
    # plt.plot(tVec, x)
    # plt.show()
    # plt.close()
    solver.plotSignals(solver.outputBaseDir + 'lixo.png', signalDictLabel='resistivity', title=r'Resistivity ($\Omega.m$)', addLegend=False)

    solver.plotSignals(solver.outputBaseDir + 'result_resistivity.png', signalDictLabel='resistivity', title=r'Resistivity ($\Omega.m$)', addLegend=False)
    solver.plotSignals(solver.outputBaseDir + 'result_delta_resistivity_perc.png', signalDictLabel='delta_resistivity_percentage',
                       title=r'Resitivity change $\frac{\Delta \rho_{\ell}}{\rho_0}$ (%)', addLegend=True)

    solver.plotSignals(solver.outputBaseDir + 'result_flowRate.png', signalDictLabel='flow_rate', title=r'Flow rate ($ml/s$)',
                       addLegend=False,scalingFactor=1e6)
    solver.plotSignals(solver.outputBaseDir + 'result_velocity.png', signalDictLabel='velocity', title='Average velocity (cm/s)',addLegend=False,
                       scalingFactor=100)
    solver.plotSignals(solver.outputBaseDir + 'result_pressure.png', signalDictLabel='pressure', title='Pressure (kPa)',addLegend=False)

    print('End of vesselDynamicModel.py')
