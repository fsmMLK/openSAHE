# -*- coding: utf-8 -*-

import multiprocessing as mp
import os
import sys
import copy

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.ndimage
import scipy.spatial

import utils

sys.path.append('../../staticComponent/src')

from atlasStatic import Atlas as staticAtlas

import neuroImgCore
import tissuePropCalculator


def chunkfy(lst, chuckLength):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), chuckLength):
        yield lst[i:i + chuckLength]


class atlasDynamic(staticAtlas):
    def __init__(self, imgFileList, outputDir, outputNamePrefix='Atlas'):
        propDummy = 'conductivity'
        freqDummy=1.0
        super(atlasDynamic, self).__init__(imgFileList, outputDir, outputNamePrefix, freq_Hz=freqDummy, propType=propDummy)
        self.buildTissuePropDictNormalized()

    def buildTissuePropDictNormalized(self):

        BLOOD, stdBLOOD = [1.0, 1.0] #normalized
        OTHER, stdOTHER = [0.005, 0.005] #normalized

        # name: [ANTS_codeNbr, mean conductivity, stdDev conductivity]
        self.tissueDataDict = {'BLOOD': [1, BLOOD, stdBLOOD], 'OTHER': [2, OTHER, stdOTHER]}

        self.Nt = len(self.tissueDataDict)
        self.backgroundData = [OTHER, stdOTHER]

    def buildTissuePropDict(self):

        tissueProp = tissuePropCalculator.TissueCalculator()

        # desvios padrao: use 20% do valor medio, como apresentado como estimativa de incerteza no artigo da gabriel.
        uncertainty_perc = 0.2

        if self.propType.lower() == 'conductivity':
            function = tissueProp.getConductivity
        if self.propType.lower() == 'resistivity':
            function = tissueProp.getResistivity
        if self.propType.lower() == 'relpermittivity':
            function = tissueProp.getRelPermittivity

        BLOOD, stdBLOOD = function(tissueName='Blood', frequency_Hz=self.freq_Hz, uncertainty_Perc=uncertainty_perc)
        OTHER, stdOTHER = [0.005, 0.005] #normalized

        # name: [ANTS_codeNbr, mean conductivity, stdDev conductivity]

        self.tissueDataDict = {'BLOOD': [1, BLOOD, stdBLOOD], 'OTHER': [2, OTHER, stdOTHER]}

        self.Nt = len(self.tissueDataDict)
        self.backgroundData = [OTHER, stdOTHER]

    def findMask(self, boundaryMinPercentage=0.75, openCloseIterations=0,forceRecalculate=False):
        """
        find the boundaries of the blood, based on a minimum percentage of occurances.
        Parameters
        ----------
        boundaryMinPercentage: float
            porcentagem mínima de imagens que devem conter o voxel  para que ele seja mantido no contorno medio.
        openCloseIterations: int
            numero de repeticoes da operacao morfologica de abrir e fechar

        """
        fileNamePrefix = self.outputDir + self.outputNamePrefix
        outputFile = fileNamePrefix + '_validPixels.npy'

        if os.path.exists(outputFile) and not forceRecalculate:
            self.validPixels = np.load(outputFile)
            self.NvalidPixels = len(self.validPixels)
            self.mask = np.load(fileNamePrefix + '_Mask.npy')
            self.imgShape = self.mask.shape
        else:
            tissuesCodes = ['BLOOD','OTHER']

            [self.mask, _, _, _] = self.findBoundaryTissues(tissuesCodes, boundaryMinPercentage, fileNamePrefix + '_Mask', openCloseIterations)
            self.imgShape = self.mask.shape

            # indices of nonzero pixels
            self.validPixels = np.nonzero(self.mask.flatten('C'))[0]
            self.NvalidPixels = len(self.validPixels)

            np.save(outputFile, self.validPixels)

    def findBoundaryTissues(self, tissueCodeList=['BLOOD'], boundaryMinPercentage=0.05, fileNamePrefix=None, openCloseIterations=0):
        """
        find the boundaries of the tissues in the list, based on a minimum percentage of occurances. This function also computes the coordnates of
        the valid voxels if calcCoords=True
        Parameters
        ----------
        tissueCodeList: list of strings
            valid values: 'BLOOD', 'OTHER'

        boundaryMinPercentage: float
            porcentagem mínima de imagens que devem conter o voxel  para que ele seja mantido no contorno medio. Valores entre 0.0 e 1.0

        fileNamePrefix: string
            file name prefix. This string must contain also the path to the file.
                if None: no files are salved.
                if valid path: then 4 files are salved:
                    fileNamePrefix_mask.npy: file with the mask array, of type bool
                    fileNamePrefix_mask.nii: file with the mask image, in .nii format
                    fileNamePrefix_mask_indices.csv: file with the indices of the valid voxels. Each line is in the form
                                i,j,k where the voxel is located at mask[i,j,k]
                    fileNamePrefix_mask_coords.csv: file with the coords of the valid voxels. Each line is in the form
                                x,y,z where the i-th line is associated with the voxel with indices at the same i-th line in
                                fileNamePrefix_mask_indices.csv

        openCloseIterations: int>=0
            number of iterations of the morphological OPEN operation.
        """
        boundarySum = np.zeros(self.loadImg(self.listFiles[0]).shape)

        ids = [self.tissueDataDict[t][0] for t in tissueCodeList]

        for i, file in enumerate(self.listFiles):
            if i % 10 == 0:
                print('  processing file %d of %d' % (i + 1, len(self.listFiles)))
            mask = np.isin(self.loadImg(file), ids)
            boundarySum += mask.astype('float')

        mask = boundarySum >= (boundaryMinPercentage * self.Np)

        # plotUtils.single_3Dplot(mask)

        # compute open operation to fix the mesh
        if openCloseIterations > 0:
            maskOriginal = mask
            # remove small white dots
            # good reference: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
            mask = scipy.ndimage.binary_opening(mask, iterations=openCloseIterations)
            # remove small black dots
            # good reference: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
            mask = scipy.ndimage.binary_closing(mask, iterations=openCloseIterations)  # plotUtils.showMask3D(maskOriginal, mask, showDiff=True)

        # label features in the mask and eliminate skull
        # self.labelStructures(mask)
        # plotUtils.showMask3D(mask)

        # compute the coordinate of the points.
        indices = np.argwhere(mask)
        indices = np.column_stack((indices, np.ones((indices.shape[0], 1))))  # add one extra element to use with the affinematrix
        coords = np.matmul(self.affineMatrix, indices.T).T
        coords = coords[:, :-1]  # removes the last column
        indices = indices[:, :-1]  # removes the last column

        # change X and Y signs
        """
        Resolução do problema de eixos invertidos:
        https://sourceforge.net/p/advants/discussion/840261/thread/2a1e9307/

        The input and output point coordinates are in "physical-LPS" coordinates, 
        that is, they are the coordinates encoded in the nifti header 
        (which are in RAS orientation), but with X and Y's sign flipped 
        (so it becomes LPS orientation). To use the function, I first do -X and -Y 
        for the points I want to use, feed them to the function, and again take
        -X and -Y of what comes out to recover the physical coordinates I expected.
        """
        coords[:, 0] *= -1.0
        coords[:, 1] *= -1.0

        # transform coordinates do match the static atlas
        transformationMATfile = self.outputDir + 'Normal001-MRA_aligned_0GenericAffine.mat'
        coordsTransformed = self.transformCoords(coords,transformationMATfile)

        if fileNamePrefix is not None:
            # save mask in npy and in nii
            np.save(fileNamePrefix + '.npy', mask)

            img = nib.Nifti1Image(mask.astype('uint16'), affine=self.affine, header=self.imgHeader)
            nib.save(img, fileNamePrefix + '.nii')

            np.savetxt(fileNamePrefix + '_indices.csv', indices, header='i,j,k', fmt='%i', delimiter=',')
            np.savetxt(fileNamePrefix + '_coords_aligned.csv', coordsTransformed, delimiter=',',
                       header='-x,-y,z (ATTENTION: negative x and y! See https://sourceforge.net/p/advants/discussion/840261/thread/2a1e9307/)')

        return [mask, boundarySum, indices[:, :-1], coords[:, :-1]]


    def transformCoords(self, coords,transformationMATfile):
        """
        Apply the affine transform to the coords of the point to match the atlas/atlas_reference_shape.nii.
        """
        print('    -> Transforming Atlas coordinates...')

        # save coords to a CSV file and transform into the atlas coordinates
        np.savetxt(self.outputDir + 'temp_coords.csv', coords, delimiter=',',
                   header='-x,-y,z (ATTENTION: negative x and y! See https://sourceforge.net/p/advants/discussion/840261/thread/2a1e9307/)')

        neuroImgCore.antsApplyTransformation(self.outputDir + 'temp_coords.csv', transformationMATfile,self.outputDir,'temp_coords_transformed')

        coordsNew = np.loadtxt(self.outputDir + 'temp_coords_transformed.csv', skiprows=1, delimiter=',')

        os.remove(self.outputDir + 'temp_coords.csv')
        os.remove(self.outputDir + 'temp_coords_transformed.csv')

        return coordsNew

    def labelStructures(self, imgArray):
        labeled_array, num_features = scipy.ndimage.label(imgArray)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.grid(True)

        ax.voxels(imgArray == 1, edgecolors='gray', shade=False)
        plt.show()
