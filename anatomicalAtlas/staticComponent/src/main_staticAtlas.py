# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 07:14:36 2020

@author: guys_
"""

import glob
import multiprocessing as mp
import os
import re
import shutil

import matplotlib
import numpy as np

import atlasStatic
import neuroImgStaticAtlas
import plotUtils
import utils

#---------------------------------------------------------------------------
# Configuration

rFActList = [ 4 ]  # resampling factor. use integer values
propList = [ 'resistivity' ] # electrical property list. Valid values: 'resistivity', 'conductivity' , 'relPermittivity'
freqHzList = [ 100 ]  # list of frequencies in Hz  [100,1000,10000,100000,1000000]

#---------------------------------------------------------------------------

currPath = os.getcwd() + '/'
rootDir = os.path.abspath(currPath + '../') + '/'

runNormalization = True
runImgProcessing = True

createTissueMasks = False

if not os.path.exists(rootDir + utils.paths['atlas']):
    os.mkdir(rootDir + utils.paths['atlas'])
if not os.path.exists(rootDir + utils.paths['output']):
    os.mkdir(rootDir + utils.paths['output'])

if runNormalization:
    shutil.copyfile(rootDir + utils.paths['input'] + 'atlas_reference_shape.nii', rootDir + utils.paths['atlas'] + 'atlas_reference_shape.nii')
    neuroImgStaticAtlas.imgRegistration(rootDir)
    neuroImgStaticAtlas.imgSegmentation(rootDir)

for rFact in rFAct_list:
    resampleFactor = [rFact, ] * 3

    if runImgProcessing:
        listFiles = sorted(glob.glob(rootDir + utils.paths['output'] + '*00_T1_aligned.nii'))


        def runParallelSeg(file):
            # this function is used to run in parallel using multiprocessing
            [_, filePrefixName, _] = utils.splitPath(file)
            filePrefixName = re.sub('_00.*', '', filePrefixName)
            print('Processing file %s in parallel ...' % os.path.basename(filePrefixName))
            sample = neuroImgStaticAtlas.neuroImg(file, filePrefixName)
            sample.unifySegmentation(nSegments=5)
            sample.fixHoles()
            sample.cleanHeadSurface()
            sample.downsample(factor=resampleFactor)


        args = [(c,) for c in listFiles]
        with mp.Pool(processes=utils.nCores) as p:
            output = p.starmap(runParallelSeg, args)

    for propType in propList:
        for freq_Hz in freqHzList:
            print('frequency: %d property type: %s' % (freq_Hz, propType))

            list_Imgs = sorted(glob.glob(rootDir + utils.paths['output'] + '*_resampled_factor_%d%d%d.nii' % tuple(resampleFactor)))

            if len(list_Imgs) == 0:
                print('Sampled files not found. Make sure you have  *_resampled_factor_XXX.nii files. You might need to run again with runImgProcessing=True ')
                sys.exit()

            AtlasOutputPrefix = 'Atlas_%s_freq_%d_RFact_%d%d%d' % tuple([propType, int(freq_Hz)] + resampleFactor)
            Atlas = atlasStatic.Atlas(list_Imgs, rootDir + utils.paths['atlas'], outputNamePrefix=AtlasOutputPrefix, freq_Hz=freq_Hz,
                                      propType=propType)

            if createTissueMasks:
                for tissue in ['GM', 'WM', 'CSF', 'BONE', 'SCALP']:
                    [_, boundary, _, _] = Atlas.findBoundaryTissues(tissueCodeList=[tissue], boundaryMinPercentage=0.0, fileNamePrefix=None)
                    plotUtils.saveArraysliceSetMontage(boundary / 50, rootDir + utils.paths['atlas'] + 'mask_%s' % tissue,
                                                       colormap=matplotlib.cm.copper, corSlice=int(100 / resampleFactor[0]),
                                                       traSlice=int(94 / resampleFactor[1]), sagSlice=int(93 / resampleFactor[2]), showColorBar=True,
                                                       cbarFontSize=15, cbarLabel='', valueLimits=[0, 1.0], orientation='vert')

            # find the boundaries of the domain, based on a minimum percentage of occurrences. considers all tissues
            Atlas.findMask(boundaryMinPercentage=0.75, forceRecalculate=True)

            # find the boundaries of the brain (GM+WM+CSF)
            Atlas.findBoundaryTissues(tissueCodeList=['SCALP'], boundaryMinPercentage=0.75,
                                      fileNamePrefix=rootDir + utils.paths['atlas'] + AtlasOutputPrefix + '_SCALP_Mask')

            Atlas.findBoundaryTissues(tissueCodeList=['GM', 'WM', 'CSF'], boundaryMinPercentage=0.75,
                                      fileNamePrefix=rootDir + utils.paths['atlas'] + AtlasOutputPrefix + '_BRAIN_Mask')

            inputSegmentedImage_Nii = rootDir + utils.paths['input'] + 'atlas_reference_shape_vascular_territories.nii'
            Atlas.createVascularTerritories(inputSegmentedImage_Nii, resampleFactor,
                                            fileNamePrefix=rootDir + utils.paths['atlas'] + AtlasOutputPrefix)
            # average
            Atlas.calcMean(forceRecalculate=True)
            img = Atlas.unvectorizeImg(Atlas.averageVector)
            Atlas.saveNII(img, rootDir + utils.paths['atlas'] + AtlasOutputPrefix + '_Avg.nii', dtype='float64', headerInfo=Atlas.imgHeader_float)

            # covariance
            Atlas.calcCov(forceRecalculate=True)
