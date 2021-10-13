# -*- coding: utf-8 -*-

import copy
import gzip
import os
import re
import sys
import glob

import nibabel as nib
import nibabel.processing as nibp
import numpy as np
import scipy.ndimage
import skimage.io as io

import plotUtils
import utils

from random import choice
from string import ascii_uppercase

def generateRandomString(length=10):
    return ''.join(choice(ascii_uppercase) for i in range(length))

def convertMHA2NII(input_MHA, output_NII, formatType=np.uint16, voxelSize=[1.0, 1.0, 1.0]):
    """
    converts .mha files to .nii
    Parameters
    ----------
    input_MHA
    formatType

    Returns
    -------

    """

    print('ATENCAO: NAO USAR ESTA ROTINA PARA CONVERTER MHA2NII!!! USAR O 3D SLICER. ESTA FUNCAO OU A DE ROTACAO ESTA COM PROBLEMAS')
    sys.exit()

    print('Converting %s to .nii' % os.path.basename(input_MHA))
    img = io.imread(input_MHA, plugin='simpleitk')

    header = nib.Nifti1Header()
    header.set_data_shape(img.shape)
    header.set_data_dtype(formatType)
    header['pixdim'][1:4] = voxelSize
    header.set_xyzt_units(xyz=2)  # codigo 2: milimetros
    header.set_sform([[voxelSize[0], 0, 0, 0], [0, voxelSize[1], 0, 0], [0, 0, voxelSize[2], 0], [0, 0, 0, 1]])
    t_image = nib.Nifti1Image(img, affine=None, header=header)
    nib.save(t_image, output_NII)

    if False:
        img = nib.load(output_NII)
        array = img.get_fdata()
        plotUtils.single_3Dplot(array)


def rotateImg(fileIn_NII, fileOut_NII=None):
    # if fileOut_NII=None, the rotated image is saved in the input file.

    print('ATENCAO: NAO USAR ESTA ROTINA PARA CONVERTER MHA2NII!!! USAR O 3D SLICER. ESTA FUNCAO OU A DE ROTACAO ESTA COM PROBLEMAS')
    sys.exit()

    img = nib.load(fileIn_NII)
    [inputDir, filePrefixName, fileExtension] = utils.splitPath(fileIn_NII)

    if True:
        # rotates the image 180 degrees and reorient the image because ANTS cannot properly align without this.
        shapeImg = img.shape
        header = copy.deepcopy(img.header)

        imgNew = np.swapaxes(img.get_fdata(), 2, 0)
        imgNew = np.rot90(imgNew, k=2, axes=(0, 1))

        header.set_data_shape(imgNew.shape)
        header['pixdim'][1:4] = [header['pixdim'][3], header['pixdim'][2], header['pixdim'][1]]

        t_image = nib.Nifti1Image(imgNew, affine=None, header=header)
        if fileOut_NII is not None:
            nib.save(t_image, fileOut_NII)
        else:
            nib.save(t_image, fileIn_NII)

    else:
        """ this way seems to work and orient correctly, but the registration step after that seems to ignore the changes in orientation. For 
        corret results use the method above.
        """
        reorient = Reorient(orientation='SPR')
        reorient.inputs.in_file = fileIn_NII
        res = reorient.run()
        imgOut = inputDir + 'temp.nii'
        if fileOut_NII is not None:
            os.rename(res.outputs.out_file, fileOut_NII)
        else:
            os.rename(res.outputs.out_file, fileIn_NII)
        os.remove(res.outputs.transform)


def changeNII_dataType(fileIn, fileOut, dtype=np.uint16):
    """
    loadas a nii file and changes its data type
    Parameters
    ----------
    fileIn: string
        input nii file
    fileOut: string
        output nii file
    dtype: numpy datatype

    """
    img = nib.load(fileIn)
    img.header.set_data_dtype(dtype)
    nib.save(img, fileOut)

def antsRegistrationSynQuick(movingImagePath, fixedImagePath, outputFileDir, outputFilePrefix):
    """
    register images using Ants Syn quick

    https://nipype.readthedocs.io/en/latest/api/generated/nipype.interfaces.ants.html

    movingImagePath. full path to the moving image
    fixedImagePath. full path to the reference image
    outputFileDir. full path to the output folder
    outputFilePrefix. output file name prefix
    """

    outFileMat = os.path.abspath(outputFileDir) + '/' + outputFilePrefix + '_0GenericAffine.mat'
    rootDataDir = os.path.abspath(os.getcwd() + '/..') + '/'

    # directories must be relative to rootDataDir
    fixedImagePathRel = os.path.relpath(fixedImagePath, rootDataDir)
    movingImagePathRel = os.path.relpath(movingImagePath, rootDataDir)

    if os.path.exists(outFileMat):
        print('File %s is already registered. Skipping...' % os.path.basename(outFileMat))
        return

    print('Processing ' + os.path.basename(movingImagePath))

    randomSuffix = generateRandomString(10)
    outputFilePrefix = outputFilePrefix + randomSuffix
    pythonfile = os.getcwd() + '/' + 'registrationANTS_temp_%s.py' % randomSuffix
    with open(pythonfile, 'w') as f:
        f.write('from nipype.interfaces.ants import RegistrationSynQuick\n')
        f.write('# import os \n\n')
        f.write('# print(os.getcwd())\n')
        f.write('# print(os.path.exists(\'%s\'))\n' % fixedImagePathRel)
        f.write('# print(os.path.exists(\'%s\'))\n' % movingImagePathRel)
        f.write('reg = RegistrationSynQuick()\n')
        f.write('reg.inputs.fixed_image = \'%s\'\n' % fixedImagePathRel)
        f.write('reg.inputs.moving_image = \'%s\'\n' % movingImagePathRel)
        f.write('reg.inputs.dimension = 3\n')
        f.write('reg.inputs.num_threads = %d\n' % utils.nCores)
        f.write('reg.inputs.transform_type = \'a\'\n')
        f.write('reg.inputs.output_prefix = \'%s_\'\n' % outputFilePrefix)
        f.write('reg.run()')

    singularityImgFile = os.path.abspath(rootDataDir + utils.singularityImgPath)
    os.system('singularity exec -B %s:/data/ %s bash -c \' pwd && cd /data && python %s\'' % (
        rootDataDir, singularityImgFile, os.path.relpath(pythonfile, rootDataDir)))
    os.remove(pythonfile)

    # gzip file of interest
    fileZip = os.path.abspath(rootDataDir + outputFilePrefix + '_Warped.nii.gz')
    print('Unpacking file %s...' % (os.path.basename(fileZip)))
    # extracts the .gz file into a temp.nii file
    tempFile = rootDataDir + utils.paths['output'] + 'temp_%s.nii' % randomSuffix
    with gzip.GzipFile(fileZip, 'rb') as fzip:
        with open(tempFile, 'wb') as fout:
            fout.write(fzip.read())
    os.remove(fileZip)
    os.remove(os.path.abspath(rootDataDir + outputFilePrefix + '_InverseWarped.nii.gz'))

    # convert temp.nii to int16 and saves it with the final name
    changeNII_dataType(fileIn=tempFile, fileOut= outputFileDir + outputFilePrefix.replace(randomSuffix,'') + '.nii', dtype=np.uint16)
    os.remove(tempFile)

    # renames affine.mat file
    inFileMat = os.path.abspath(rootDataDir + outputFilePrefix + '_0GenericAffine.mat')
    os.rename(inFileMat, outFileMat)

def antsApplyTransformation(input_CSV, transform_MAT,outputFileDir,outputFilePrefix):
    """
    register images using Ants Syn quick

    https://nipype.readthedocs.io/en/latest/api/generated/nipype.interfaces.ants.html

    input_CSV. full path to the input file with coordinates. the file must be a CSV file
    transform_MAT. full path to the transformation. the file is a .mat file, created by ANTS. see antsRegistrationSynQuick
    outputFileDir. full path to the output folder
    outputFilePrefix. output file name prefix
    """

    rootDataDir = os.path.abspath(os.getcwd() + '/..') + '/'

    # directories must be relative to rootDataDir
    fixedImagePathRel = os.path.relpath(input_CSV, rootDataDir)
    movingImagePathRel = os.path.relpath(transform_MAT, rootDataDir)

    print('Processing ' + os.path.basename(input_CSV))

    randomSuffix = generateRandomString(10)
    pythonfile = os.getcwd() + '/' + 'transformationANTS_temp_%s.py' % randomSuffix
    tempFile = 'temp_%s.csv' % randomSuffix
    with open(pythonfile, 'w') as f:
        f.write('from nipype.interfaces.ants import ApplyTransformsToPoints\n')
        f.write('# import os \n\n')
        f.write('# print(os.getcwd())\n')
        f.write('# print(os.path.exists(\'%s\'))\n' % fixedImagePathRel)
        f.write('# print(os.path.exists(\'%s\'))\n' % movingImagePathRel)
        f.write('at = ApplyTransformsToPoints()\n')
        f.write('at.inputs.dimension = 3\n')
        f.write('at.inputs.input_file = \'%s\'\n' % fixedImagePathRel)
        f.write('at.inputs.transforms = [\'%s\']\n' % movingImagePathRel)
        f.write('at.inputs.invert_transform_flags = [True]\n')
        f.write('at.inputs.output_file = \'%s\'\n' % tempFile)
        f.write('at.run()')

    singularityImgFile = os.path.abspath(rootDataDir + utils.singularityImgPath)
    os.system('singularity exec -B %s:/data/ %s bash -c \' pwd && cd /data && python %s\'' % (
        rootDataDir, singularityImgFile, os.path.relpath(pythonfile, rootDataDir)))
    os.remove(pythonfile)

    # renames affine.mat file
    os.rename(rootDataDir + tempFile, outputFileDir + outputFilePrefix + '.csv')


def spm12SegmentationT1T2(T1imagePath, T2imagePath):
    """
    head segmentation using spm12

    https://nipype.readthedocs.io/en/latest/api/generated/nipype.interfaces.ants.html

    movingImagePath. full path to the moving image
    fixedImagePath. full path to the reference image
    outputFileDir. full path to the output folder
    outputFilePrefix. output file name prefix
    """

    rootDataDir = os.path.abspath(os.getcwd() + '/..') + '/'

    # directories must be relative to rootDataDir
    T1fileRelative = os.path.relpath(T1imagePath, rootDataDir)
    T2fileRelative = os.path.relpath(T2imagePath, rootDataDir)

    # check if the file starting with c1 already existis.
    [outputDir, filePrefixName, fileExtension] = utils.splitPath(T1imagePath)
    fileNameWithNewNumber = re.sub('_00.*', '_01', filePrefixName)
    outputFile = outputDir + fileNameWithNewNumber + '_segmented_c1.nii'

    if os.path.exists(outputFile):
        print('File %s is already segmented. Skipping...' % os.path.basename(outputFile))
        return

    print('Processing ' + os.path.basename(T1fileRelative))

    randomSuffix = generateRandomString(10)
    pythonfile = os.getcwd() + '/' + 'segmentationSPM12_temp%s.py' % randomSuffix
    with open(pythonfile, 'w') as f:
        f.write('import nipype.interfaces.spm as spm\n')
        f.write('import os \n')
        f.write('import utils \n\n')

        f.write('matlab_cmd = \'/opt/spm12-r7771/run_spm12.sh /opt/matlabmcr-2019b/v97/ script\'\n')
        f.write('spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)\n\n')

        f.write('# print(os.getcwd())\n')
        f.write('# print(os.path.exists(\'%s\'))\n' % T1fileRelative)
        f.write('# print(os.path.exists(\'%s\'))\n' % T2fileRelative)
        f.write('fileT1 = \'%s\'\n' % T1fileRelative)
        f.write('fileT2 = \'%s\'\n' % T2fileRelative)
        f.write('print(\'Coregistering T2->T1...\')\n')
        f.write('coreg = spm.Coregister()\n\n')

        f.write('coreg.inputs.target = fileT1\n')
        f.write('coreg.inputs.source = fileT2\n')
        f.write('coreg.run()\n\n')

        f.write('# rename registered image T2\n')
        f.write('[dirName, filePrefixName, fileExtension] = utils.splitPath(fileT2)\n')
        f.write('fileT2_coregistered = dirName + \'r\' + filePrefixName + fileExtension\n\n')

        f.write('#removes the first character that is an \'r\' (default output prefix from SPM) and  add prefix \'_coregistered\'\n')
        f.write('fileT2_newName = dirName + filePrefixName + \'_coregistered\' + fileExtension\n')
        f.write('os.rename(fileT2_coregistered, fileT2_newName)\n\n')

        f.write('print(\'Segmenting image...\')\n\n')

        f.write('seg = spm.MultiChannelNewSegment()\n\n')

        f.write('saveBiasFieldImg=False\n')
        f.write('saveCorrectedImage=False\n')
        f.write('channel1= (fileT1, (0.0001, 60, (saveBiasFieldImg, saveCorrectedImage) ) )\n')
        f.write('channel2= (fileT2_newName, (0.0001, 60, (saveBiasFieldImg, saveCorrectedImage) ) )\n')
        f.write('seg.inputs.channels = [channel1, channel2]\n')
        f.write('seg.run()\n')

    singularityImgFile = os.path.abspath(rootDataDir + utils.singularityImgPath)
    os.system('singularity exec -B %s:/data/ %s bash -c \' cd /data && python %s\'' % (
         rootDataDir, singularityImgFile, os.path.relpath(pythonfile, rootDataDir)))
    os.remove(pythonfile)

    # renames the output
    fileSeg8old = filePrefixName + '_seg8.mat'
    fileSeg8new = fileNameWithNewNumber + '_segmented_seg8.mat'
    os.rename(outputDir + fileSeg8old, outputDir + fileSeg8new)

    # fix tissue file names
    for file in sorted(glob.glob(rootDataDir + utils.paths['output'] + 'c[0-9]*_00_T1_aligned.nii')):
        [_, filePrefixName, fileExtension] = utils.splitPath(file)
        tissueString = filePrefixName[0:2]
        newName = outputDir + fileNameWithNewNumber + '_segmented_' + tissueString + fileExtension
        os.rename(file, newName)

class neuroImgCore():
    def __init__(self, Niifile, filePrefixName):
        """

        Parameters
        ----------
        Niifile: an nii image. this file will be used to extract the header and directory only. Any segmented or original image will do
        filePrefixName: prefix file name
        """
        [self.outputDir, _, self.fileExtension] = utils.splitPath(Niifile)
        self.filePrefixName = filePrefixName
        self.loadHeader(Niifile)

    def loadHeader(self, fileName):
        img = nib.load(fileName)
        # self.dataType = img.header['datatype'].dtype
        self.data = img.get_fdata()
        self.shape = img.shape
        # plotUtils.single_3Dplot(self.data)

        self.affine = img.affine
        self.voxelSizes_mm = img.header.get_zooms()
        self.affineMatrix = img.header.get_sform()  # Afine Matrix para transformar voxel em coordenadas de mundo
        self._buildHeaders(img.header)


    def loadImgArray(self, fileName, dtype=None):
        """
        load a .nii file and converts to dtype array

        dtype: one of https://docs.scipy.org/doc/numpy-1.10.0/user/basics.types.html
        some values:  bool_,int[8,16,32,64],uint[8,16,32,64],float[16,32,64],comples[64,128]. can also be a nd.dtype

        """
        if dtype is None:
            return nib.load(fileName).get_fdata()
        else:
            return nib.load(fileName).get_fdata().astype(dtype)

    def _buildHeaders(self, mainHeader):
        """
        create NII headers for different data types
        Parameters
        ----------
        mainHeader: main header to be replicated

        """
        self.imgHeader_uint8 = copy.deepcopy(mainHeader)
        self.imgHeader_uint8.set_data_dtype(np.uint8)

        self.imgHeader_uint16 = copy.deepcopy(mainHeader)
        self.imgHeader_uint16.set_data_dtype(np.uint16)

        self.imgHeader_float = copy.deepcopy(mainHeader)
        self.imgHeader_float.set_data_dtype(np.float)

    def saveNII(self, imgArray, fileName, dtype='float64', headerInfo=None):
        """
        Saves an array into a nii file

        Entrada:
        - fileName: full path to the file
        - dtype: dtype of the image (int, float, etc)
        - header_info: header from nibabel
        """
        img = nib.Nifti1Image(imgArray.astype(dtype), affine=self.affine, header=headerInfo)
        img.header.set_data_dtype(dtype)
        nib.save(img, fileName)

    def saveNumpy(self, imgArray, fileName, dtype='float64'):
        """
        Função criada para salvar um array em formato numpy

        Entrada:
        - fileName: caminho e nome do arquivo .nii a ser salvo.
        - dtype: variavel para definir o tipo de dado (int, float, etc)
        """
        np.save(fileName, imgArray.astype(dtype))

    def unifySegmentation(self, nSegments=5):
        """
        Função utilizada para juntar as máscaras segmentadas pelo SPM em
        uma única imagem, considerando que cada píxel receba a label de maior
        probabilidade. Em caso de empate, a label é mantida como 0.

        Entrada:
        segmentImages list with the paths of the segment files.
        """
        outputfile = self.outputDir + self.filePrefixName + '_01_segmented.nii'

        if os.path.exists(outputfile):
            self.imgSegments = nib.load(outputfile).get_fdata()  # self.saveNumpy(self.imgSegments, outputfile.replace('.nii','.npy'), dtype=np.uint8)
        else:
            listTissues = []
            for n in range(nSegments):
                # Estas imagens apresentam a probabilidade de cada pixel ser de um determinado tecido. pixels fora da cabeca tem valor 0
                fileName = self.outputDir + self.filePrefixName + '_01_segmented_c%d.nii' % (n + 1)
                img = nib.load(fileName).get_fdata()
                listTissues.append(img)

            maxProbabilityImg = np.maximum.reduce(listTissues)

            maskTissues = [x == maxProbabilityImg for x in listTissues]

            # ensures the marks compose a partition of the image
            for i in range(nSegments):
                for j in range(i + 1, nSegments):
                    maskTissues[j] = maskTissues[j] * np.logical_not(maskTissues[i])

            # imagem caracteristica de todos os tecidos.
            self.imgSegments = np.zeros(listTissues[0].shape)

            for i in range(nSegments):
                self.imgSegments += maskTissues[i] * (i + 1)

            mask = np.zeros(listTissues[0].shape)
            for img in listTissues:
                mask += img

            mask = mask > 0.49

            self.imgSegments *= mask
            self.saveNII(self.imgSegments, outputfile, dtype=np.uint8, headerInfo=self.imgHeader_uint8)

    def fixHoles(self):
        """
        Função utilizada para preencher buracos com a label de menor distancia
        dele que estiver em maior quantidade.
        """
        outputfile = self.outputDir + self.filePrefixName + '_02_filledHoles.nii'

        if os.path.exists(outputfile):
            self.imgSegments = nib.load(
                outputfile).get_fdata()  # self.saveNII(self.imgSegments, outputfile, dtype=np.uint8,headerInfo=self.imgHeader_uint8)  # self.saveNumpy(self.imgSegments, outputfile.replace('.nii','.npy'), dtype=intFormat)
        else:

            mask = self.imgSegments > 0

            # mask = scipy.ndimage.morphology.binary_fill_holes(mask)

            for j in range(np.size(mask, 2)):
                mask[:, :, j] = scipy.ndimage.morphology.binary_fill_holes(mask[:, :, j])

            for j in range(np.size(mask, 1)):
                mask[:, j, :] = scipy.ndimage.morphology.binary_fill_holes(mask[:, j, :])

            for j in range(np.size(mask, 0)):
                mask[j, :, :] = scipy.ndimage.morphology.binary_fill_holes(mask[j, :, :])

            bg = mask == 0
            holes = np.logical_xor(self.imgSegments == 0, bg)
            ind_fill = np.where(holes)
            molde = bg * (-1) + self.imgSegments + holes * (-2)

            final = self.imgSegments

            coord = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [-1, 0, -1], [-1, -1, 0],
                     [0, -1, -1], [1, 0, -1], [1, -1, 0], [0, 1, -1], [-1, 0, 1], [-1, 1, 0], [0, -1, 1], [1, 1, 1], [-1, -1, -1], [1, 1, -1],
                     [1, -1, 1], [-1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]

            for num_hole in range(np.size(ind_fill[0])):
                if num_hole % 10000 == 0:
                    print("  -> pixel %d of %d" % (num_hole + 1, np.size(ind_fill[0])))
                directions = np.zeros((26, 2))

                for ic in range(len(coord)):
                    neighbour = -2
                    inc = 1

                    while neighbour == -2:
                        idxX = ind_fill[0][num_hole] + inc * coord[ic][0]
                        idxY = ind_fill[1][num_hole] + inc * coord[ic][1]
                        idxZ = ind_fill[2][num_hole] + inc * coord[ic][2]
                        if ((0 <= idxX < np.size(self.imgSegments, 0)) and (0 <= idxY < np.size(self.imgSegments, 1)) and (
                          0 <= idxZ < np.size(self.imgSegments, 2))):
                            neighbour = molde[idxX, idxY, idxZ]
                        else:
                            neighbour = -1

                        if neighbour != -2:
                            directions[ic] = [neighbour, inc]

                        inc = inc + 1

                directions[6:18, 1] *= np.sqrt(2)
                directions[18:, 1] *= np.sqrt(3)
                directions = directions[np.where(directions[:, 0] != -1)]
                min_dis = np.argmin(directions[:, 1])
                directions = directions[np.where(directions[:, 1] == directions[min_dis, 1])]

                uni = np.unique(directions[:, 0])
                rep = np.zeros(len(uni))

                for ir in range(len(rep)):
                    rep[ir] = np.size(np.where(directions[:, 0] == uni[ir]))

                value_s = uni[np.argmax(rep)]

                final[ind_fill[0][num_hole], ind_fill[1][num_hole], ind_fill[2][num_hole]] = value_s

            self.imgSegments = final
            self.saveNII(self.imgSegments, outputfile, dtype=np.uint8, headerInfo=self.imgHeader_uint8)

    def cleanHeadSurface(self, openIterations=4):
        """
        Função utilizada para eliminar grupos desconectados menores e manter
        apenas o maior do escalpo

        Entrada:
        - image_path: nome base do arquivo.
        """

        outputfile = self.outputDir + self.filePrefixName + '_03_cleaned.nii'

        if os.path.exists(outputfile):
            self.imgSegments = nib.load(outputfile).get_fdata()  # self.saveNumpy(self.imgSegments, outputfile.replace('.nii','.npy'), dtype=np.uint8)
        else:

            labeled_array, num_features = scipy.ndimage.measurements.label(self.imgSegments)

            listCont = []
            for i in range(num_features):
                listCont.append(np.count_nonzero(labeled_array == i + 1))

            largestLabel = np.argmax(listCont)

            self.imgSegments = (labeled_array == largestLabel + 1) * self.imgSegments

            # open scalp surface to remove artifacts
            headMask = self.imgSegments > 0
            headMask = scipy.ndimage.binary_opening(headMask, iterations=openIterations)
            self.imgSegments = headMask * self.imgSegments

            self.saveNII(self.imgSegments, outputfile, dtype=np.uint8, headerInfo=self.imgHeader_uint8)

    def downsample(self, factor=[4, 4, 4], min_tol=0.01):
        """
        Função criada para diminuir a resolução da imagem.
        """

        outputfile = self.outputDir + self.filePrefixName + '_04_resampled_factor_%d%d%d.nii' % tuple(factor)

        if os.path.exists(outputfile):
            print('resampled image already computed. skipping...')
            img = nib.load(outputfile)
            self.imgSegments = img.get_fdata()
        else:
            imgOrig = nib.Nifti1Image(self.imgSegments, affine=self.affine, header=self.imgHeader_uint8)
            img = nibp.resample_to_output(imgOrig, voxel_sizes=np.multiply(factor,self.voxelSizes_mm),order = 0,mode='nearest' )
            self.imgSegments = np.around(img.get_fdata(), decimals=0)

        self._buildHeaders(img.header)
        self.affine = img.affine
        self.voxelSizes_mm = img.header.get_zooms()

        if not os.path.exists(outputfile):
            self.saveNII(self.imgSegments, outputfile, dtype=np.uint8, headerInfo=self.imgHeader_uint8)
