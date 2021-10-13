# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 07:14:36 2020

@author: guys_
"""

import glob
import multiprocessing as mp
import os
import re
import sys
import shutil

import matplotlib
import nibabel as nib
import numpy as np

import atlasDynamic
import plotUtils
import utils

sys.path.append('../../staticComponent/src')

import neuroImgCore
import neuroImgDynamicAtlas

#---------------------------------------------------------------------------
# Configuration

rFActList = [ 4 ]  # resampling factor. use integer values

#---------------------------------------------------------------------------

currPath = os.getcwd() + '/'
rootDir = os.path.abspath(currPath + '../') + '/'

# normalize and segment sampes
runNormalization = True
runSegmentation = True

if not os.path.exists(rootDir + utils.paths['atlas']):
    os.mkdir(rootDir + utils.paths['atlas'])
if not os.path.exists(rootDir + utils.paths['output']):
    os.mkdir(rootDir + utils.paths['output'])

if runNormalization:
    # aligns the samples with the reference  Normal001-MRA.nii
    #
    # in this function (registration.py) I have to use Normal001-MRA.nii located in the /input/ directory as fixed image because
    # the atlas_reference_shape.nii I used in the static atlas is only (~190x233x189 pixels) and the angio images are (~450x450x130).
    # Remember that imgRegistration will make each moving image the same size as the fixed one. This would result in low res normalized images.
    # The next step will have to normalize in relation to atlas_reference_shape.nii, but I will do it on the final average image only and not the
    # samples.
    neuroImgDynamicAtlas.imgRegistration(rootDir)
    #plotUtils.comp_3Dplot_NII(rootDir + utils.paths['output'] + 'Normal003-MRA_00_aligned.nii', rootDir + utils.paths['input'] + 'Normal001-MRA.nii')

    shutil.copyfile(rootDir + utils.paths['input'] + 'atlas_reference_shape.nii', rootDir + utils.paths['atlas'] + 'atlas_reference_shape.nii')

    # I am aligning Normal001-MRA.nii because all the other images were aligned to this image. Therefore I can use the same
    # transformation later when I transform the coordinates of the voxels of the dynamic atlas to the reference geometry of the static atlas
    outputfile = rootDir + utils.paths['atlas'] + 'Normal001-MRA_aligned_0GenericAffine.mat'
    if not os.path.exists(outputfile):
        fixed = os.path.abspath(rootDir + utils.paths['atlas'] + 'atlas_reference_shape.nii')
        moving = os.path.abspath(rootDir + utils.paths['input'] + 'Normal001-MRA.nii')
        neuroImgCore.antsRegistrationSynQuick(movingImagePath=moving, fixedImagePath=fixed, outputFileDir=rootDir + utils.paths['atlas'],
                                              outputFilePrefix='Normal001-MRA_aligned')

for rFact in rFActList:
    resampleFactor = np.array([rFact, ] * 3, dtype=int)

    if runSegmentation:
        # segment the images. No affine transformation is performed.
        print('Finding masks')
        listFiles = sorted(glob.glob(rootDir + utils.paths['output'] + '*_aligned.nii'))

        def runParallelSeg(file):
            # this function is used to run in parallel using multiprocessing
            [_, filePrefixName, _] = utils.splitPath(file)
            filePrefixName = re.sub('_00.*', '', filePrefixName)
            print('Processing file %s in parallel ...' % os.path.basename(filePrefixName))
            neuroImg = neuroImgDynamicAtlas.neuroImg(file,filePrefixName)

            if True:
                neuroImg.segmentImage(file, thresholdLevelVessel=0.3)
                neuroImg.unifySegmentation(nSegments=2)
            else:
                print('ATENTION: skipping segmentImage and unifySegmentation!')
                print('ATENTION: skipping segmentImage and unifySegmentation!')

            neuroImg.fixHoles()
            neuroImg.downsample(factor=resampleFactor)

        args = [(c,) for c in listFiles]
        with mp.Pool(processes=utils.nCores) as p:
            output = p.starmap(runParallelSeg, args)


    listFiles = sorted(glob.glob(rootDir + utils.paths['output'] + '*_resampled_factor_%d%d%d.nii' % tuple(resampleFactor)))

    if len(listFiles) == 0:
        print('Sampled files not found. Make sure you have  *_resampled_factor_XXX.nii files. You might need to run again with runImgProcessing=True ')
        sys.exit()

    AtlasOutputPrefix = 'Atlas_normalized_Rfactor_%d%d%d' % tuple(resampleFactor.tolist())
    Atlas = atlasDynamic.atlasDynamic(listFiles, rootDir + utils.paths['atlas'], outputNamePrefix=AtlasOutputPrefix)

    # find the boundaries of the domain, based on a minimum percentage of occurances. considers all tissues
    Atlas.findMask(boundaryMinPercentage=0.75, openCloseIterations=0,forceRecalculate=True)

    # find the boundaries of blood
    Atlas.findBoundaryTissues(tissueCodeList=['BLOOD'], boundaryMinPercentage=0.01,
                                  fileNamePrefix=rootDir + utils.paths['atlas'] + AtlasOutputPrefix + '_BLOOD_Mask', openCloseIterations=2)

    #Disconsider 'OTHER' tissue before the calculation.
    # OTHER was added to facilitate defining the mask
    Atlas.calcMean()
    img = Atlas.unvectorizeImg(Atlas.averageVector)
    Atlas.saveNII(img, rootDir + utils.paths['atlas'] + AtlasOutputPrefix + '_Avg.nii', dtype='float64', headerInfo=Atlas.imgHeader_float)

    Atlas.calcCov()

    print('oi')
