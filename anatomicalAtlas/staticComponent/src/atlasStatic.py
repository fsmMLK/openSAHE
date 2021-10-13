# -*- coding: utf-8 -*-

import copy
import os
import sys

import matplotlib
import nibabel as nib
import numpy as np

import plotUtils

import tissuePropCalculator

class Atlas:
    def __init__(self, imgFileList, outputDir, outputNamePrefix='Atlas', freq_Hz=125e3, propType='conductivity'):
        self.listFiles = imgFileList
        self.freq_Hz = freq_Hz
        self.propType = propType
        self.outputNamePrefix = outputNamePrefix
        self.outputDir = outputDir
        self.Np = len(self.listFiles)
        self.imgHeader = nib.load(self.listFiles[0]).header
        self.imgHeader.set_data_dtype(np.uint16)
        self.imgHeader_float = copy.deepcopy(self.imgHeader)
        self.imgHeader_float.set_data_dtype(np.float)
        self.affine = nib.load(self.listFiles[0]).affine
        self.affineMatrix = self.imgHeader.get_sform()  # Afine Matrix para transformar voxel em coordenadas de mundo

        self.buildTissuePropDict()

    def buildTissuePropDict(self):

        tissueProp = tissuePropCalculator.TissueCalculator()

        # desvios padrao: use 20% do valor medio, como apresentado como estimativa de incerteza no artigo da gabriel.
        uncertainty_perc = 0.2

        if self.propType.lower() == 'conductivity':
            myFunction = tissueProp.getConductivity

        if self.propType.lower() == 'resistivity':
            myFunction = tissueProp.getResistivity

        if self.propType.lower() == 'relpermittivity':
            myFunction = tissueProp.getRelPermittivity

        GM, stdGM = myFunction(tissueName='Brain_(Grey_Matter)', frequency_Hz=self.freq_Hz, uncertainty_Perc=uncertainty_perc)
        WM, stdWM = myFunction(tissueName='Brain_(White_Matter)', frequency_Hz=self.freq_Hz, uncertainty_Perc=uncertainty_perc)
        CSF, stdCSF = myFunction(tissueName='Cerebro_Spinal_Fluid', frequency_Hz=self.freq_Hz, uncertainty_Perc=uncertainty_perc)

        # SCALP Uses muscle values.  resultado parecido com a tabela mencionada pelo roberto no mestrado dele.
        SCALP, stdSCALP = myFunction(tissueName='Muscle', frequency_Hz=self.freq_Hz, uncertainty_Perc=uncertainty_perc)

        # bones:  average between cortical 0.020861 and cancellous bone 0.084086
        B1, stdB1 = myFunction(tissueName='Bone_(Cancellous)', frequency_Hz=self.freq_Hz, uncertainty_Perc=uncertainty_perc)
        B2, stdB2 = myFunction(tissueName='Bone_(Cortical)', frequency_Hz=self.freq_Hz, uncertainty_Perc=uncertainty_perc)
        BONE = (B1 + B2) / 2.0
        stdBONE = np.sqrt(stdB1 ** 2 + stdB2 ** 2)

        # name: [ANTS_codeNbr, mean conductivity, stdDev conductivity]

        self.tissueDataDict = {'GM': [1, GM, stdGM], 'WM': [2, WM, stdWM], 'CSF': [3, CSF, stdCSF], 'BONE': [4, BONE, stdBONE],
                               'SCALP': [5, SCALP, stdSCALP]}

        self.Nt = len(self.tissueDataDict)
        self.backgroundData = [SCALP, stdSCALP]

    def preview_NII(self, NIIfile):
        array = nib.load(NIIfile).get_fdata()
        plotUtils.single_3Dplot(array, scale='linear', colormap='jet', colorLimits='slice')

    def loadImg(self, fileName):
        img = nib.load(fileName)
        return img.get_fdata().astype('uint16')

    def getMasks(self, imgArray):
        maskTissues = [imgArray == (i + 1) for i in range(self.Nt)]
        return maskTissues

    def saveNII(self, imgArray, fileName, dtype='float64', headerInfo=None):
        """
        Função criada para salvar um array em uma imagem .nii

        Entrada:
        - fileName: caminho e nome do arquivo .nii a ser salvo.
        - dtype: variavel para definir o tipo de dado (int, float, etc)
        - header_info: header obtido da estrurua do nibabel.
        """
        img = nib.Nifti1Image(imgArray.astype(dtype), affine=self.affine, header=headerInfo)
        img.header.set_data_dtype(dtype)
        nib.save(img, fileName)

    def findMask(self, boundaryMinPercentage=0.75, forceRecalculate=False):
        """
        find the boundaries of the domain, based on a minimum percentage of occurances. considers all tissues
        Parameters
        ----------
        boundaryMinPercentage: float
            porcentagem mínima de imagens que devem conter o voxel  para que ele seja mantido no contorno medio.

        """
        fileNamePrefix = self.outputDir + self.outputNamePrefix
        outputFile = fileNamePrefix + '_validPixels.npy'

        if os.path.exists(outputFile) and not forceRecalculate:
            self.validPixels = np.load(outputFile)
            self.NvalidPixels = len(self.validPixels)
            self.mask = np.load(fileNamePrefix + '_Mask.npy')
            self.imgShape = self.mask.shape

        else:
            tissuesCodes = self.tissueDataDict.keys()

            [self.mask, _, _, _] = self.findBoundaryTissues(tissuesCodes, boundaryMinPercentage, fileNamePrefix + '_Mask')
            self.imgShape = self.mask.shape

            # indices of nonzero pixels
            self.validPixels = np.nonzero(self.mask.flatten('C'))[0]
            self.NvalidPixels = len(self.validPixels)

            np.save(outputFile, self.validPixels)

    def findBoundaryTissues(self, tissueCodeList=['GM', 'WM', 'CSF'], boundaryMinPercentage=0.75, fileNamePrefix=None):
        """
        find the boundaries of the tissues in the list, based on a minimum percentage of occurances. This function also computes the coordnates of
        the valid voxels if calcCoords=True
        Parameters
        ----------
        tissueCodeList: list of strings
            valid values: 'GM', 'WM', 'CSF', 'BONE', 'SCALP'

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

        if fileNamePrefix is not None:
            # save mask in npy and in nii
            np.save(fileNamePrefix + '.npy', mask)

            img = nib.Nifti1Image(mask.astype('uint16'), affine=self.affine, header=self.imgHeader)
            nib.save(img, fileNamePrefix + '.nii')

            np.savetxt(fileNamePrefix + '_indices.csv', indices, header='i,j,k', fmt='%i', delimiter=',')
            np.savetxt(fileNamePrefix + '_coords.csv', coords, delimiter=',',
                       header='-x,-y,z (ATTENTION: negative x and y! See https://sourceforge.net/p/advants/discussion/840261/thread/2a1e9307/)')

        return [mask, boundarySum, indices[:, :-1], coords[:, :-1]]

    def createVascularTerritories(self,  inputSegmentedImage_Nii, resampleFactor, fileNamePrefix):
        """
        Função utilizada para  criar os vascular territories.

        Entrada:
        inputSegmentedImage_Nii: image with segmented vascular territories
        resample factor: fator de reducao usado no atlas
        fileNamePrefix: prefixo do nome do arquivo de saida
        """
        outputfile = fileNamePrefix + '_vascularTerritories'

        if os.path.exists(outputfile + '.npy'):
            pass
        else:

            # load reference image for visualization

            if False:
                averageImageFile = self.outputDir + 'atlas_reference_shape.nii'
                img = nib.load(averageImageFile)
                referenceImage = np.around(nib.processing.resample_to_output(img, voxel_sizes=np.multiply(resampleFactor, img.header.get_zooms()), order=0,
                                                                   mode='nearest').get_fdata(), decimals=0)

                plotUtils.comp_3Dplot(referenceImage, segmentsAll)

            # load vascular Territories segments
            img = nib.load(inputSegmentedImage_Nii)
            segmentsAll = np.around(nib.processing.resample_to_output(img, voxel_sizes=np.multiply(resampleFactor, img.header.get_zooms()), order=0,
                                                               mode='nearest').get_fdata(), decimals=0)

            MCA_R_ID=1
            ACA_R_ID=2
            PCA_R_ID=3
            SCA_R_ID=4
            STEM_ID=5
            ACA_L_ID=6
            PCA_L_ID =7
            MCA_L_ID=8
            SCA_L_ID=9
            ECA_R_ID=10
            ECA_L_ID=11

            # masks

            maskACA_L = (segmentsAll == ACA_L_ID)
            maskACA_R = (segmentsAll == ACA_R_ID)

            maskMCA_L = (segmentsAll == MCA_L_ID)
            maskMCA_R = (segmentsAll == MCA_R_ID)

            maskPCA_L = (segmentsAll == PCA_L_ID)
            maskPCA_R = (segmentsAll == PCA_R_ID)

            maskSCA_L = (segmentsAll == SCA_L_ID)
            maskSCA_R = (segmentsAll == SCA_R_ID)

            maskSTEM = (segmentsAll == STEM_ID)

            maskECA_R = (segmentsAll == ECA_R_ID)
            maskECA_L = (segmentsAll == ECA_L_ID)

            vascularTerritories=maskACA_L.astype(int) * 1 + maskACA_R.astype(int) * 2 + \
                                maskMCA_L.astype(int) * 3 + maskMCA_R.astype(int) * 4 + \
                                maskPCA_L.astype(int) * 5 + maskPCA_R.astype(int) * 6 + \
                                maskSCA_L.astype(int) * 7 + maskSCA_R.astype(int) * 8 + \
                                maskSTEM.astype(int) * 9 + \
                                maskECA_L.astype(int) * 10 + maskECA_R.astype(int) * 11

            #plotUtils.single_3Dplot(vascularTerritories,scale='linear', colormap='jet', colorLimits='slice')

            if fileNamePrefix is not None:
                img = nib.Nifti1Image(vascularTerritories.astype('uint16'), affine=self.affine, header=self.imgHeader)
                nib.save(img, outputfile + '.nii')

                np.save(outputfile + '.npy', self.vectorizeImg(vascularTerritories))

    def disconsiderTissue(self,tisueDicKeyName):
        """
        disconsiders a tissue in the calculations.
        tisueDicKeyName: key name in the dictionary: Ex 'GM', 'WM', 'CSF', 'BONE','SCALP','BLOOD','OTHER'
        """
        del self.tissueDataDict[tisueDicKeyName]
        self.Nt -= 1

    def setElectricalProperty(self, imgVector):
        """
        Função criada para substituir labels por valores de resistividade.

        backgroundVal: value to be used for pixels outside the segmented image but within the boundary of the average shape
        """
        electricPropertyVector = np.zeros(imgVector.shape, dtype='float')
        for idx, value in self.tissueDataDict.items():
            prop = value[1]
            code = int(value[0])
            electricPropertyVector += prop * (imgVector == code)

        # add scalp property to the pixels outside the segmented image but within the boundary of the average shape
        backgroundVal = self.backgroundData[0]
        electricPropertyVector += (imgVector == 0) * backgroundVal

        return electricPropertyVector

    def vectorizeImg(self, imgArray):
        """
        Transforms an 3D image into the vectorized form, that is, removes all pixels without information

        Parameters
        ----------
        imgArray: 3D numpy array

        Returns
        -------
        img: 1D numpy array
            vectorized image
        """
        temp = imgArray.flatten('C')
        return temp[self.validPixels]

    def unvectorizeImg(self, imgVector):
        """
        transforms a vectorized image into a 3D image array
        Parameters
        ----------
        imgVector: 1D numpy array

        Returns
        -------
        img: 3D numpy array
            3D array

        """
        img = np.zeros(self.imgShape[0] * self.imgShape[1] * self.imgShape[2], dtype=imgVector.dtype) * np.nan
        img[self.validPixels] = imgVector
        return np.reshape(img, self.imgShape, 'C')

    def calcMean(self,forceRecalculate=True):
        """
        computes the average of the atlas.
        """
        outputFile = self.outputDir + self.outputNamePrefix + '_Avg.npy'

        if os.path.exists(outputFile) and not forceRecalculate:
            print('average  already computed. loading it...')
            self.averageVector = np.load(outputFile)
        else:
            print('computing atlas AVERAGE')
            self.averageVector = np.zeros(self.NvalidPixels)

            for n, file in enumerate(self.listFiles):
                if n % 10 == 0:
                    print('  processing file %d of %d' % (n + 1, len(self.listFiles)))

                imgArray = self.loadImg(file)

                # plotUtils.comp_plot(imgArray,self.mask)
                imgVector = self.vectorizeImg(imgArray)
                imgVectorElectricProperty = self.setElectricalProperty(imgVector)
                # np.save(self.outputDir +  '_lixo_img_%d.npy' % i, self.unvectorizeImg(imgVectorElectricProperty))
                if False:
                    if 'Normal010' in file:
                        plotUtils.saveArraysliceSet(self.unvectorizeImg(imgVectorElectricProperty), file.replace('.nii', '_resistivity'),
                                                    colormap=matplotlib.cm.jet, corSlice=int(65 / 2), traSlice=int(93 / 2), sagSlice=int(114 / 2))

                self.averageVector += imgVectorElectricProperty

            self.averageVector /= self.Np

            np.save(outputFile, self.averageVector)

    def calcCov(self,forceRecalculate=True):
        """
        computes the covariance of the atlas. See notebook 3 page 19
        """
        outputFile = self.outputDir + self.outputNamePrefix + '_CovK.npy'

        if os.path.exists(outputFile) and not forceRecalculate:
            print('CovK already computed. loading it...')
            self.covK = np.load(outputFile)
        else:
            variances = []
            for idx, value in self.tissueDataDict.items():
                variances.append(value[2] ** 2)
                print('mean: %f std=%f' % (value[1], value[2]))
            variances.append(1.0)
            Y = np.diag(variances)
            Ysqrt = np.sqrt(Y)

            # np.savetxt(self.outputDir + 'lixo_Y.txt', Ysqrt)

            self.covK = np.zeros([self.NvalidPixels, self.Np * (self.Nt + 1)])

            for n, file in enumerate(self.listFiles):
                if n % 10 == 0:
                    print('  processing file %d of %d' % (n + 1, len(self.listFiles)))
                imgArray = self.loadImg(file)
                imgVector = self.vectorizeImg(imgArray)
                maskTissues = self.getMasks(imgVector)
                x_mean = self.setElectricalProperty(imgVector)
                W_j = np.column_stack(tuple(maskTissues) + (x_mean - self.averageVector,))
                K_j = W_j.dot(Ysqrt) / np.sqrt(self.Np)

                self.covK[:, n * (self.Nt + 1):(n + 1) * (self.Nt + 1)] = K_j

            # [self.U, self.S, self.Vt] = np.linalg.svd(self.covK, full_matrices=False)

            np.save(outputFile, self.covK)
            # np.save(self.outputDir + self.outputNamePrefix + '_CovK_U.npy', self.U)
            # np.save(self.outputDir + self.outputNamePrefix + '_CovK_S.npy', self.S)
            # np.save(self.outputDir + self.outputNamePrefix + '_CovK_Vt.npy', self.Vt)

    def getVarianceImg(self):
        """
        extracts the variance image (main diagonal of the covariance matrix).

        Returns
        -------
        img: 1D numpy array
            vectorized image
        """
        return np.sum(np.square(self.covK), axis=1)
