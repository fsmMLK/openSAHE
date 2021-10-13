# -*- coding: utf-8 -*-

import glob
import gzip
import os
import re

import nibabel as nib
import numpy as np

import neuroImgCore
import utils

### SEGMENTATION AND REGISTRATION ###
def imgRegistration(rootPath):
    """
    resgister images from input folder

    Input:
    - rootPath: root folder of the project. The folder with atlas, inputData, outputData, src folders
    """

    print('registering images...')
    rootPath = os.path.abspath(rootPath) + '/'

    # find the prefixes of the files
    list_inputImgs = sorted(glob.glob(rootPath + utils.paths['input'] + 'Normal*.nii'))
    fixedReferenceImg = os.path.abspath(rootPath + utils.paths['atlas'] + 'atlas_reference_shape.nii')

    for movingImg in list_inputImgs:

        outputFileDir = rootPath + utils.paths['output']
        [_, filenamePrefix, _] = utils.splitPath(movingImg)

        if '-T1' in filenamePrefix:
            outFileSufix = filenamePrefix.replace('-T1', '_00_T1_aligned')
        if '-T2' in filenamePrefix:
            outFileSufix = filenamePrefix.replace('-T2', '_00_T2_aligned')

        neuroImgCore.antsRegistrationSynQuick(movingImagePath=os.path.abspath(movingImg), fixedImagePath=fixedReferenceImg,
                                              outputFileDir=outputFileDir, outputFilePrefix=outFileSufix)

def imgSegmentation(rootPath):
    """
    segment images of the utils.paths['output'] folder

    input:
    - rootPath: root folder of the project. The folder with atlas, inputData, outputData, src folders
    """

    print('segmentig images...')
    rootPath = os.path.abspath(rootPath) + '/'
    # explore  docker:  docker run -it nipype/nipype sh

    # find the prefixes of the files
    # listing only images T1 because we use T1 as reference to register t2 into t1
    list_inputImgs = set(glob.glob(rootPath + utils.paths['output'] + '*_00_T1_aligned.nii')) - set(
        glob.glob(rootPath + utils.paths['output'] + "c[0-9]*"))
    list_inputImgs = sorted(list_inputImgs)

    print('Attention. SPM Segmentation does not show any output during execution. It takes quite some time to segment. Wait....')

    for fileT1 in list_inputImgs:
        fileT2 = fileT1.replace('_T1_aligned', '_T2_aligned')
        neuroImgCore.spm12SegmentationT1T2(fileT1, fileT2)


class neuroImg(neuroImgCore.neuroImgCore):
    def __init__(self, Niifile, filePrefixName):
        super(neuroImg, self).__init__(Niifile, filePrefixName)


