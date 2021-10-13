# -*- coding: utf-8 -*-
"""
codigo usado para pré-processar as imagens de ressonancia para usar no atlas
"""
import copy
import glob
import gzip
import os
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy
# noinspection PyUnresolvedReferences
from mpl_toolkits import mplot3d
from nipype.interfaces.image import Reorient
from skimage import measure, morphology, restoration

sys.path.append('../../staticComponent/src')

import neuroImgCore
import plotUtils
import utils

def alignAtlas(fixedImg_NII, movingImg_NII):
    [inputDir, filePrefixName, _] = utils.splitPath(movingImg_NII)
    [outputDir, _, _] = utils.splitPath(fixedImg_NII)

    # plotUtils.comp_3Dplot_NII(movingImg_NII, fixedImg_NII)

    baseDir = os.path.abspath(inputDir + '../../../').replace("C:\\", "/c/").replace("\\", "/") + '/'

    singularityImgFile = os.path.abspath('../sigularityImages/nipype_ANTS_SPM12_SingularityImg.simg')
    os.system('singularity exec -B %s:/data/ %s bash -c \' cd /data && python src/atlas_dynamicTissue/registrationMalha01Atlas.py\'' % (
        baseDir, singularityImgFile))

    # extracts the .gz file into a temp.nii file
    tempFile = baseDir + 'temp.nii'
    with gzip.GzipFile(baseDir + 'Alinhamento_Warped.nii.gz', 'rb') as fzip:
        with open(tempFile, 'wb') as fout:
            fout.write(fzip.read())

    # convert temp.nii to int16 and saves it with the final name
    neuroImgCore.changeNII_dataType(tempFile, outputDir + filePrefixName + '_aligned.nii', dtype=np.uint16)

    # rename the transformation matrix
    os.rename(baseDir + 'Alinhamento_0GenericAffine.mat', outputDir + filePrefixName + '_aligned.mat')

    # removes unnecessary files
    for f in glob.glob(baseDir + 'Alinhamento_*.gz'):
        os.remove(f)

    os.remove(tempFile)

    #plotUtils.comp_3Dplot_NII(outputDir + filePrefixName + '_aligned.nii', fixedImg_NII)
    print('oi')


### REGISTRATION ###
# here I have to use Normal001-MRA.nii located in the /inputImages/ directory as fixed image because
# the atlas.nii image is smaller due to the reduction factor I used in the static atlas.
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
    fixedReferenceImg = os.path.abspath(rootPath + utils.paths['input'] + 'Normal001-MRA.nii')

    for movingImg in list_inputImgs:

        outputFileDir = rootPath + utils.paths['output']
        [_, filenamePrefix, _] = utils.splitPath(movingImg)

        neuroImgCore.antsRegistrationSynQuick(movingImagePath=os.path.abspath(movingImg), fixedImagePath=fixedReferenceImg,
                                              outputFileDir=outputFileDir, outputFilePrefix=filenamePrefix + '_00_aligned')


def imgRegistrationOld(path):
    """
    Função criada para alinhar as imagens de um diretório.

    Entrada:
    - path: diretório com as imagens a serem normalizadas.
    """

    print('registering images...')
    path = os.path.abspath(path).replace("C:\\", "/c/").replace("\\", "/") + '/'
    # explore  docker:  docker run -it nipype/nipype sh

    if False:
        os.system('docker run -v ' + path + ':/data/ -w /data nipype/nipype python registration.py')
    else:
        singularityImgFile = os.path.abspath('../sigularityImages/nipype_ANTS_SPM12_SingularityImg.simg')
        os.system('singularity exec -B %s:/data/ %s bash -c \' cd /data && python '
                  'src/atlas_dynamicTissue/registration.py\'' % (path + '../../', singularityImgFile))

    # find the prefixes of the files
    list_inputImgs = sorted(glob.glob(path + utils.paths['output'] + '*_00_aligned_0GenericAffine.mat'))
    prefixList = [os.path.basename(f).replace('_00_aligned_0GenericAffine.mat', '') for f in list_inputImgs]

    # extracts the .zp files and give them new names
    for prefix in prefixList:

        outFile = os.path.abspath(path + utils.paths['output'] + prefix + '_00_aligned.nii')
        if os.path.exists(outFile):
            print('File %s is already aligned. Skipping...' % prefix)
        else:
            # gzip file of interest
            file = os.path.abspath(path + utils.paths['output'] + prefix + '_00_aligned_Warped.nii.gz')

            # extracts the .gz file into a temp.nii file
            tempFile = path + utils.paths['output'] + 'temp.nii'
            with gzip.GzipFile(file, 'rb') as fzip:
                with open(tempFile, 'wb') as fout:
                    fout.write(fzip.read())

            # convert temp.nii to int16 and saves it with the final name
            img = nib.load(tempFile)
            imgHeader = img.header
            imgHeader.set_data_dtype(np.uint16)
            nib.save(img, outFile)

            # removes unnecessary files
            os.remove(tempFile)
            os.remove(file)
            os.remove(path + utils.paths['output'] + prefix + '_00_aligned_InverseWarped.nii.gz')


def plotVessels(imgArray):
    fractionOfPoints = 50
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    indices = np.argwhere(imgArray)
    indices = indices[::fractionOfPoints, :]

    ax.scatter(indices[:, 0], indices[:, 1], indices[:, 2], c='r')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def plotVessels2(arrayCoords1, arrayCoords2):
    fractionOfPoints1 = 50
    fractionOfPoints2 = 5
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(arrayCoords1[::fractionOfPoints1, 0], arrayCoords1[::fractionOfPoints1, 1], arrayCoords1[::fractionOfPoints1, 2], c='r')

    ax.scatter(arrayCoords2[::fractionOfPoints2, 0], arrayCoords2[::fractionOfPoints2, 1], arrayCoords2[::fractionOfPoints2, 2], c='b')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


class neuroImg(neuroImgCore.neuroImgCore):
    def __init__(self, Niifile, filePrefixName):
        super(neuroImg, self).__init__(Niifile, filePrefixName)

    def segmentImage(self, file, thresholdLevelVessel=0.2):
        # plotUtils.comp_3Dplot_NII(imgOut, os.getcwd() + '/' + utils.paths['atlas'] + 'atlas.nii')

        outFile = self.outputDir + self.filePrefixName + '_01_segmented_c2.nii'
        if os.path.exists(outFile):
            print('File %s already segmented. Skipping...' % self.filePrefixName)
        else:
            imgArrayOriginal = self.loadImgArray(file)

            # imgArray = ndimage.median_filter(imgArray, 4)
            imgArray = restoration.denoise_tv_chambolle(imgArrayOriginal, weight=10, eps=0.002, n_iter_max=200)

            # plotUtils.comp_3Dplot(imgArrayOriginal, imgArray)
            # image normalization so that the pixels are within 0.0 and 1.0
            maxIntensity = np.max(imgArray)
            minIntensity = np.min(imgArray)
            # normalizes image between 0.0 and 1.0
            imgArray = (imgArray - minIntensity) / (maxIntensity - minIntensity)

            # segments based on threshold

            # 1- vessel
            maskVessel = imgArray > thresholdLevelVessel
            outputFile = self.outputDir + self.filePrefixName + '_01_segmented_c1.nii'
            # multiply by 255 to converto to 8bit gray scale
            self.saveNII(255 * maskVessel, outputFile, dtype=np.uint8, headerInfo=self.imgHeader_uint8)

            # 2- head
            # find treshould based on noise in the boundary
            imgSlice = imgArray[:50, :, :]
            percentile = np.percentile(imgSlice, 90)
            maskHead = imgArray > (percentile * 1.1)

            # fill holes inthe first and last slides
            maskHead[:, :, 0] = scipy.ndimage.binary_fill_holes(maskHead[:, :, 0])
            maskHead[:, :, -1] = scipy.ndimage.binary_fill_holes(maskHead[:, :, -1])

            # compute open operation to fix the mesh
            maskOriginal = copy.copy(maskHead)

            def create_circular_mask(size=np.array([5, 5, 5]), center=None, radius=None):
                # soruce: https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
                if center is None:  # use the middle of the image
                    center = (size / 2.0).astype(int)
                if radius is None:  # use the smallest distance between the center and image walls
                    radius = min(center[0], center[1], center[2], size[0] - center[0], size[1] - center[1], size[2] - center[2])

                X, Y, Z = np.ogrid[:size[0], :size[1], :size[2]]
                dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2)

                mask = dist_from_center <= radius
                return mask

            maskHead = scipy.ndimage.binary_fill_holes(maskHead)

            # --- remove everything smaller than largest object
            # label and calculate parameters for every cluster in mask
            labels = measure.label(maskHead)
            rp = measure.regionprops(labels)
            # get size of largest cluster
            size = max([i.area for i in rp])

            maskHead = morphology.remove_small_objects(maskHead, min_size=size * 0.9)

            operatorMask = create_circular_mask(np.array([7, ] * 3))
            openCloseIterations = 2
            # remove small white dots
            # good reference: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
            maskHead = scipy.ndimage.binary_opening(maskHead, structure=operatorMask, iterations=openCloseIterations)
            # remove small black dots
            # good reference: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
            # maskHead = scipy.ndimage.binary_closing(maskHead, structure=operatorMask, iterations=openCloseIterations)

            # plotUtils.comp_3Dplot(maskOriginal.astype(int), maskHead.astype(int), scale='linear', colormap='jet', colorLimits='full')

            outputFile = self.outputDir + self.filePrefixName + '_01_segmented_c2.nii'
            self.saveNII(255 * maskHead, outputFile, dtype=np.uint8, headerInfo=self.imgHeader_uint8)
