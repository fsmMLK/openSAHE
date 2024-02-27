#!/bin/python
"""
regularizations module
"""

# -*- coding: utf-8 -*-

import copy
import glob
import os
import sys
import time

import nibabel as nib
import numpy as np
from scipy import io as scipyIo
from scipy import sparse as scipySparse

import FEMelements
import tissuePropCalculator
import vesselDynamicModel

sys.path.append('./atlas_staticTissue')

import tools
import multiprocessing as mp
import matplotlib

# loads Qt5Agg only in my laptop
if tools.isFeLap():
    matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
from mpl_toolkits import mplot3d


def chunkfy(lst, chuckLength):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), chuckLength):
        yield lst[i:i + chuckLength]


# function for parallel processing
def computeDistVessel(elementCentroid, vesselVoxelsCoords, norm='L2'):
    # compute distance vectors to the reference element
    deltas = vesselVoxelsCoords - elementCentroid

    if norm.upper() == 'L2':
        dist = np.sum(deltas ** 2, axis=1)  # distance squared here!
        idxMin = np.argmin(dist)
        distMin = np.sqrt(dist[idxMin])
    if norm.upper() == 'L1':
        dist = np.sum(np.absolute(deltas), axis=1)
        idxMin = np.argmin(dist)
        distMin = dist[idxMin]
    return [distMin, idxMin]


def computeInterpolationRow(centroidDestination, centroidOrigin, interpolationStdDev):
    # compute distance vectors to the reference element
    deltas = centroidOrigin - centroidDestination

    dist_squared = np.sum(deltas ** 2, axis=1)  # distance squared here!

    method = 'gaussian'
    if method == 'gaussian':
        # find origin elements  not too far from the destination element
        validElements = np.argwhere(dist_squared < 36.0 * (interpolationStdDev ** 2)).flatten()  # 36*var results in  weights~= 1e-15

        # if no element is close, then find the closest
        if len(validElements) > 0:
            weights = np.exp(-dist_squared[validElements] / (2 * interpolationStdDev ** 2))
        else:
            # validElements = np.argwhere(dist_squared == dist_squared.min()).flatten()
            # if the standard deviation is too small, then increases it unit finds nearby elements
            factor = 1.1
            while len(validElements) == 0:
                newSTD = interpolationStdDev * factor
                validElements = np.argwhere(dist_squared < 36.0 * (newSTD ** 2)).flatten()  # 36*var results in  weights~= 1e-15
                weights = np.exp(-dist_squared[validElements] / (2 * newSTD ** 2))
                factor *= factor

    # normalizes
    if weights.sum() > 0.0:
        weights /= weights.sum()
    else:
        print('ERROR: interpolator matrix row has no elements != zero!. ROW: %d' % i)
    return [weights, validElements]


class AnatomicalAtlasCore:
    def __init__(self, atlasConf, freq_Hz, femMesh):

        print('Building the anatomical Atlas')
        self.atlasConf = atlasConf
        self.freq_Hz = freq_Hz
        self.staticAtlasConf = self.atlasConf.xpath('staticAtlas')[0]
        self.dynamicAtlasConf = self.atlasConf.xpath('dynamicAtlas')[0]
        self.femMesh = femMesh
        self.filePrefix = self.femMesh.filePrefix
        self.baseDir = self.femMesh.baseDir
        self.outputBaseDir = self.femMesh.outputBaseDir
        self.staticAtlasDir = os.path.abspath(self.baseDir + tools.getElemValueXpath(self.staticAtlasConf, xpath='atlasDir', valType='str')) + '/'
        # self.baseFile = self.outputBaseDir + self.filePrefix
        self.avg = None
        self.covK = None
        self.components = {'static': {'label': 'static'}}

        # --> alignment
        self.voxelSize_mm = tools.getElemValueXpath(self.staticAtlasConf, xpath='alignment/voxelSize_mm', valType='list_float')
        self.alignmentMeshTag = tools.getElemValueXpath(self.staticAtlasConf, xpath='alignment/meshTag', valType='list_int')
        self.NIImaskFile = self.staticAtlasDir + tools.getElemValueXpath(self.staticAtlasConf, xpath='alignment/NIImaskFile', valType='str')
        self.pixelCoordsCSVfile = self.staticAtlasDir + tools.getElemValueXpath(self.staticAtlasConf, xpath='alignment/pixelCoordsCSVfile',
                                                                                valType='str')

        # --> atlas interpolation
        # load and convert stdDev to metres
        stdDev = tools.getElemValueXpath(self.staticAtlasConf, xpath='interpolationStdDev', valType='float')
        stdDevUnit = tools.getElemAttrXpath(self.staticAtlasConf, xpath='interpolationStdDev', attrName='unit', attrType='str')
        self.interpolationStdDev = tools.convToMetre(stdDev, stdDevUnit)
        self.interpolationChunkSize = tools.getElemValueXpath(self.staticAtlasConf, xpath='interpolationChunkSize', valType='int')

        self.components['static']['avgFile'] = os.path.abspath(
          self.staticAtlasDir + tools.getElemValueXpath(self.staticAtlasConf, xpath='avgFile', valType='str'))
        self.components['static']['covKFile'] = os.path.abspath(
          self.staticAtlasDir + tools.getElemValueXpath(self.staticAtlasConf, xpath='covKFile', valType='str'))
        self.components['static']['avg'] = None
        self.components['static']['covK'] = None

        self.voxelizeGeometry()
        self.alignAtlas()

        transformationMATfile = self.outputBaseDir + self.filePrefix + '_01_aligned_affine.mat'
        outputFile = self.outputBaseDir + self.filePrefix + '_02_coordinates_aligned.csv'
        self.components['static']['voxelCoords'] = self.transformAtlasPoints(self.pixelCoordsCSVfile, outputFile, transformationMATfile)

        self.components['static']['interpChunkFileNames'] = self.createInterpAtlas2Mesh(self.components['static']['voxelCoords'],
                                                            filePrefix=self.components['static']['label'])

        # --> blood vessels
        if tools.isActiveElement(self.dynamicAtlasConf, xpath='.'):
            self.dynamicAtlasDir = os.path.abspath(self.baseDir + tools.getElemValueXpath(self.dynamicAtlasConf, xpath='atlasDir', valType='str')) + '/'
            # read blood Vessel information from conf file.
            self.useDynamicAtlas = True
            self.components['dynamic'] = {'label': 'dynamic'}
            self.components['dynamic']['avgFile'] = os.path.abspath(
              self.dynamicAtlasDir + tools.getElemValueXpath(self.dynamicAtlasConf, xpath='avgFile', valType='str'))
            self.components['dynamic']['covKFile'] = os.path.abspath(
              self.dynamicAtlasDir + tools.getElemValueXpath(self.dynamicAtlasConf, xpath='covKFile', valType='str'))
            self.ElectricalProperty =  tools.getElemValueXpath(self.dynamicAtlasConf, xpath='property', valType='str')

            self.components['dynamic']['avg'] = None
            self.components['dynamic']['covK'] = None
            fileName = tools.getElemValueXpath(self.dynamicAtlasConf, xpath='pixelCoordsCSVfile', valType='str')
            self.vesselPixelCoordsCSVfile = os.path.abspath(self.dynamicAtlasDir + fileName)

            # fileName = tools.getElemValueXpath(self.dynamicAtlasConf, xpath='pixelFrequencyCSVfile', valType='str')
            # self.vesselPixelFrequencyCSVfile = os.path.abspath(self.dynamicAtlasDir + fileName)

            transformationMATfile = self.outputBaseDir + self.filePrefix + '_01_aligned_affine.mat'
            outputFile = self.outputBaseDir + self.filePrefix + '_02_vessel_coordinates_aligned.csv'
            self.components['dynamic']['voxelCoords'] = self.transformAtlasPoints(self.vesselPixelCoordsCSVfile, outputFile, transformationMATfile)

            self.calcVascularTerritories()

            #compute the volume of each vascular territory
            self.components['dynamic']['interpChunkFileNames'] = self.createInterpAtlas2Mesh(self.components['dynamic']['voxelCoords'],
                                                                          filePrefix=self.components['dynamic']['label'])

            # openBF
            self.vesselDynamicModel = vesselDynamicModel.dynamicModel(self.dynamicAtlasConf, self.baseDir)

            # run openBF if the directory does not exist
            if not os.path.exists(self.vesselDynamicModel.outputBaseDir):
                self.vesselDynamicModel.run_openBF(skipRun=False)

            self.vesselDynamicModel.loadSolution(resample=False, resapleFreq_Hz=None)
            self.vesselDynamicModel.convert_Vel2electricalProp(self.freq_Hz, property=self.ElectricalProperty)

        else:
            self.useDynamicAtlas = False

        #self.plotPoints()


    def voxelizeGeometry(self):
        """
        Create a gridded version of the mesh.

        Header format of  a .nii file:
        https://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h

        """
        outputfile = self.outputBaseDir + self.filePrefix + '_00_brain_voxelized.nii'

        if os.path.exists(outputfile):
            return
        else:
            print('    -> Creating pixelated mesh geometry...')
            [minBboxMesh, maxBboxMesh] = self.femMesh.getMeshLimits()

            # rounds limits and convert to millimetre
            minBboxMesh = np.floor(minBboxMesh * 1000)
            maxBboxMesh = np.ceil(maxBboxMesh * 1000)

            # voxel coords along x, y and Z
            xValues = np.arange(minBboxMesh[0], maxBboxMesh[0], self.voxelSize_mm[0])
            yValues = np.arange(minBboxMesh[1], maxBboxMesh[1], self.voxelSize_mm[1])
            zValues = np.arange(minBboxMesh[2], maxBboxMesh[2], self.voxelSize_mm[2])

            img = np.zeros((len(xValues), len(yValues), len(zValues)), dtype=bool)

            for elem in self.femMesh.getElementsByMeshTag(self.alignmentMeshTag):
                if elem.isRegion:
                    print('ERROR: elem type is UniformRhoRegion. This case was not implemented in the function AnatomicalAtlas_voxelizeGeometry')
                    quit()

                [minBbox, maxBbox] = elem.getBbox()
                # convert to millimetre
                minBbox *= 1000
                maxBbox *= 1000

                # find all pixel coordinates inside the bounding box of the element
                XidxInsideBbox = np.nonzero((xValues >= minBbox[0]) & (xValues <= maxBbox[0]))[0]
                YidxInsideBbox = np.nonzero((yValues >= minBbox[1]) & (yValues <= maxBbox[1]))[0]
                ZidxInsideBbox = np.nonzero((zValues >= minBbox[2]) & (zValues <= maxBbox[2]))[0]

                for i in XidxInsideBbox:
                    for j in YidxInsideBbox:
                        for k in ZidxInsideBbox:
                            if not img[i, j, k]:
                                # x1000 to convert m -> mm
                                img[i, j, k] = tools.isPointInsideTetra(np.array([xValues[i], yValues[j], zValues[k]]), elem.coords * 1000.0)

            # Criar header para salvar a imagem como .nii
            header = nib.Nifti1Header()
            header.set_data_shape(img.shape)
            header.set_data_dtype(np.float)  # must be float!
            header['pixdim'][1:4] = self.voxelSize_mm
            header.set_xyzt_units(xyz=2)  # codigo 2: milimetros
            header.set_sform([[self.voxelSize_mm[0], 0, 0, minBboxMesh[0]], [0, self.voxelSize_mm[1], 0, minBboxMesh[1]],
                              [0, 0, self.voxelSize_mm[2], minBboxMesh[2]], [0, 0, 0, 1]])
            t_image = nib.Nifti1Image(img, affine=None, header=header)
            # plotUtils.single_3Dplot(img*1)

            nib.save(t_image, outputfile)

    def alignAtlas(self):
        """
        Aligns the atlas to the pixelated version of the mesh, from voxelizeGeometry function. This function generates a .mat file with the affine
        transform needed to transform pixels from atlas space to femMesh space.
        """
        outputfile = self.outputBaseDir + self.filePrefix + '_01_aligned_affine.mat'

        if os.path.exists(outputfile):
            return
        else:
            print('    -> Aligning atlas with mesh...')

            rootDataDir = os.path.abspath(os.getcwd() + '/..') + '/'

            # directories must be relative to rootDataDir
            dirAtlasRelative = os.path.relpath(self.staticAtlasDir, rootDataDir) + '/'
            dirMeshRelative = os.path.relpath(os.path.abspath(self.outputBaseDir), rootDataDir) + '/'
            atlasIMG_relPath = dirAtlasRelative + os.path.basename(self.NIImaskFile)
            meshIMG_relPath = dirMeshRelative + self.filePrefix + '_00_brain_voxelized.nii'
            code = 'from nipype.interfaces.ants import RegistrationSynQuick\n' \
                   '# import os \n\n' \
                   '# print(os.getcwd())\n' \
                   '# print(os.path.exists(\'%s\'))\n' \
                   '# print(os.path.exists(\'%s\'))\n' \
                   'reg = RegistrationSynQuick()\n' \
                   'reg.inputs.fixed_image = \'%s\'\n' \
                   'reg.inputs.moving_image = \'%s\'\n' \
                   'reg.inputs.dimension = 3\n' \
                   'reg.inputs.num_threads = %d\n' \
                   'reg.inputs.transform_type = \'a\'\n' \
                   'reg.inputs.output_prefix = \'%s_01_aligned_\'\n' \
                   'reg.run()' % (meshIMG_relPath, atlasIMG_relPath, meshIMG_relPath, atlasIMG_relPath, tools.nCores, self.filePrefix)

            pythonfile = self.filePrefix + '_01_alignAtlas2Mesh.py'
            with open(self.outputBaseDir + pythonfile, 'w') as f:
                f.write(code)

            if False:
                os.system('docker run -v ' + rootDataDir + ':/data/ -w /data nipype/nipype python %s' % (dirMeshRelative + '/' + pythonfile))
            else:

                singularityImgFile = os.path.abspath(tools.singularityImgPath)
                os.system('singularity exec -B %s:/data/ %s bash -c \' cd /data && python %s\'' % (
                    rootDataDir, singularityImgFile, dirMeshRelative + pythonfile))

            for f in glob.glob(rootDataDir + self.filePrefix + '_01_aligned*.gz'):
                os.remove(f)

            os.rename(rootDataDir + self.filePrefix + '_01_aligned_0GenericAffine.mat', outputfile)

    def transformAtlasPoints(self, inputCSV, outputCSV, transformationMATfile):
        """
        Apply the affine transform in all atlas pixels
        """
        outputfile = outputCSV

        if not os.path.exists(outputfile):
            print('    -> Transforming Atlas coordinates...')

            # directories must be relative to rootDataDir
            rootDataDir = os.path.abspath(os.getcwd() + '/..') + '/'

            dirAtlasRelative = os.path.relpath(self.staticAtlasDir, rootDataDir) + '/'
            dirMeshRelative = os.path.relpath(os.path.abspath(self.outputBaseDir), rootDataDir) + '/'

            # atlasCOORDS_relPath = dirAtlasRelative + 'atlas_BRAIN_Mask_coords.csv'
            atlasCOORDS_relPath = os.path.relpath(inputCSV, rootDataDir)
            affineTransformFile_relPath = os.path.relpath(transformationMATfile, rootDataDir)

            code = 'from nipype.interfaces.ants import ApplyTransformsToPoints\n' \
                   '# import os \n\n' \
                   '# print(os.getcwd())\n' \
                   '# print(os.path.exists(\'%s\'))\n' \
                   '# print(os.path.exists(\'%s\'))\n' \
                   'at = ApplyTransformsToPoints()\n' \
                   'at.inputs.dimension = 3\n' \
                   'at.inputs.input_file = \'%s\'\n' \
                   'at.inputs.transforms = [\'%s\']\n' \
                   'at.inputs.invert_transform_flags = [True]\n' \
                   'at.inputs.output_file = \'%s_02_coordinates_aligned.csv\'\n' \
                   'at.run()\n' % (
                       atlasCOORDS_relPath, affineTransformFile_relPath, atlasCOORDS_relPath, affineTransformFile_relPath, self.filePrefix)

            pythonfile = self.filePrefix + '_02_transformAtlas2Mesh.py'
            with open(self.outputBaseDir + pythonfile, 'w') as f:
                f.write(code)

            if False:
                os.system('docker run -v ' + rootDataDir + ':/data/ -w /data nipype/nipype python %s' % (dirMeshRelative + '/' + pythonfile))
            else:
                singularityImgFile = os.path.abspath(tools.singularityImgPath)
                os.system('singularity exec -B %s:/data/ %s bash -c \' cd /data && python %s\'' % (
                    rootDataDir, singularityImgFile, dirMeshRelative + pythonfile))

            os.remove(self.outputBaseDir + pythonfile)
            os.rename(rootDataDir + self.filePrefix + '_02_coordinates_aligned.csv', outputfile)

        voxelCoords = np.loadtxt(outputfile, delimiter=',', skiprows=1)

        voxelCoords /= 1000  # converts to metre

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
        voxelCoords[:, 0] *= -1.0
        voxelCoords[:, 1] *= -1.0

        return voxelCoords

    def calcVascularTerritories(self):
        """
        find the vascular territories
        """
        outputfile = self.outputBaseDir + self.filePrefix + '_02_vascular_territories.npy'

        # vascular territories
        self.components['dynamic']['territories'] = []
        for t in self.dynamicAtlasConf.xpath('vascularTerritories/territories')[0]:
            territoryName = tools.getElemValueXpath(t, xpath='name', valType='str')
            side = tools.getElemValueXpath(t, xpath='side', valType='str')
            meshTags = tools.getElemValueXpath(t, xpath='meshTag', valType='list_int')
            segmentId = tools.getElemValueXpath(t, xpath='segmentationID', valType='int')

            self.components['dynamic']['territories'].append({'name': territoryName, 'side': side, 'meshTags': meshTags, 'segmentId': segmentId})

        if os.path.exists(outputfile):
            print('    -> Blood vessel elements already found. Skipping...')
            self.vascularTerritories = np.load(outputfile)
        else:

            # ELEMENTS OF THE BRAIN
            print('    -> Defining vascular territories...')

            meshTagsBrain = []
            segmentIdsBrain = []
            meshTagsScalp = []
            segmentIdsScalp = []
            for t in self.components['dynamic']['territories']:
                if t['name'] in ['ACA2', 'PCA2', 'MCA', 'SCA']:
                    meshTagsBrain += t['meshTags']
                    segmentIdsBrain.append(t['segmentId'])
                if t['name'] in ['ECA']:
                    meshTagsScalp += t['meshTags']
                    segmentIdsScalp.append(t['segmentId'])

            # remove repeated elements of the list
            meshTagsBrain = list(set(meshTagsBrain))
            segmentIdsBrain = list(set(segmentIdsBrain))
            meshTagsScalp = list(set(meshTagsScalp))
            segmentIdsScalp = list(set(segmentIdsScalp))

            self.vascularTerritories = []
            for meshTag,segmentIds in zip([meshTagsBrain, meshTagsScalp],[segmentIdsBrain,segmentIdsScalp]):
                selectedElements = self.femMesh.getElementsByMeshTag(meshTag)

                # extract the coords of the territories in consideration (only brain, only scalp, etc)
                vascularTerritoryFile = tools.getElemValueXpath(self.dynamicAtlasConf, xpath='vascularTerritories/segmentationFile', valType='str')
                vascularTerritorySegments = np.load(self.staticAtlasDir + vascularTerritoryFile)

                usedIds=[]
                for id in segmentIds:
                    usedIds.append(np.where(vascularTerritorySegments==id)[0])

                usedIds = np.concatenate(usedIds)
                usedIds.sort()

                # prepare data to run in parallel
                norm = 'L1'
                usedCoords  = self.components['static']['voxelCoords'][usedIds]
                args = [(elem.centroid, usedCoords, norm) for elem in selectedElements]
                with mp.Pool(processes=tools.nCores) as p:
                    listElem = p.starmap(computeDistVessel, args)

                distMin = np.array([x[0] for x in listElem], dtype=np.float32)
                idxMin = np.array([usedIds[x[1]] for x in listElem], dtype=np.int32)

                # set element frequency the same as the closest pixel n
                elemNbr = np.array([e.number for e in selectedElements], dtype='i8')

                elemTerritory = [vascularTerritorySegments[idx] for idx in idxMin]

                # crate a table of characteristics
                property = np.empty(idxMin.shape[0], dtype=[('elemNbr', 'i8'), ('minDist', 'f8'), ('territoryID', 'i8')])

                property['elemNbr'] = elemNbr
                property['minDist'] = distMin
                property['territoryID'] = elemTerritory

                self.vascularTerritories.append(property)

            self.vascularTerritories = np.concatenate(tuple(self.vascularTerritories))
            np.save(outputfile,self.vascularTerritories )

        # find the total volume in the territories
        if False:
            for vessel in self.components['dynamic']['territories']:
                name = vessel['name']
                side = vessel['side']
                segmentID = vessel['segmentId']

                # find all elements in current territory
                elementsFullListIDs = np.where(self.vascularTerritories['territoryID'] == segmentID)[0]
                elementsList = self.vascularTerritories['elemNbr'][elementsFullListIDs]

                vessel['volume'] = 0.0
                for elemNbr in elementsList:
                    vessel['volume'] += self.femMesh.elements[elemNbr].volume
                    #print('oi')

    def calcVascularTerritoriesPoints(self,pointCoords):
        """
        find the vascular territories
        """
        outputfile = self.outputBaseDir + self.filePrefix + '_02_vascular_territories_points.npy'

        # vascular territories
        self.components['dynamic']['territories'] = []
        for t in self.dynamicAtlasConf.xpath('vascularTerritories/territories')[0]:
            territoryName = tools.getElemValueXpath(t, xpath='name', valType='str')
            side = tools.getElemValueXpath(t, xpath='side', valType='str')
            segmentId = tools.getElemValueXpath(t, xpath='segmentationID', valType='int')

            self.components['dynamic']['territories'].append({'name': territoryName, 'side': side, 'segmentId': segmentId})

        if os.path.exists(outputfile):
            print('    -> Blood vessel elements already found. Skipping...')
            self.vascularTerritories = np.load(outputfile)
        else:

            # ELEMENTS OF THE BRAIN
            print('    -> Defining vascular territories...')

            segmentIds = [t['segmentId'] for t in self.components['dynamic']['territories']]

            # remove repeated elements of the list
            segmentIds = list(set(segmentIds))


            # extract the coords of the territories in consideration (only brain, only scalp, etc)
            vascularTerritoryFile = tools.getElemValueXpath(self.dynamicAtlasConf, xpath='vascularTerritories/segmentationFile', valType='str')
            vascularTerritorySegments = np.load(self.staticAtlasDir + vascularTerritoryFile)

            usedIds=[]
            for id in segmentIds:
                usedIds.append(np.where(vascularTerritorySegments==id)[0])

            usedIds = np.concatenate(usedIds)
            usedIds.sort()

            # prepare data to run in parallel
            norm = 'L1'
            usedCoords  = self.components['static']['voxelCoords'][usedIds]
            args = [(elem, usedCoords, norm) for elem in pointCoords]
            with mp.Pool(processes=tools.nCores) as p:
                listElem = p.starmap(computeDistVessel, args)

            distMin = np.array([x[0] for x in listElem], dtype=np.float32)
            idxMin = np.array([usedIds[x[1]] for x in listElem], dtype=np.int32)

            # set element frequency the same as the closest pixel n
            elemNbr = np.arange(pointCoords.shape[0], dtype='i8')

            elemTerritory = [vascularTerritorySegments[idx] for idx in idxMin]

            # crate a table of characteristics
            property = np.empty(idxMin.shape[0], dtype=[('elemNbr', 'i8'), ('minDist', 'f8'), ('territoryID', 'i8')])

            property['elemNbr'] = elemNbr
            property['minDist'] = distMin
            property['territoryID'] = elemTerritory

            self.vascularTerritories = []
            self.vascularTerritories.append(property)

            self.vascularTerritories = np.concatenate(tuple(self.vascularTerritories))
            np.save(outputfile,self.vascularTerritories )

    def plotPoints(self, additionalPoints=None):
        """
        plot points for debugging reasons
        """

        # plot centroids
        allDomain = True
        if allDomain:
            centroids = np.zeros((len(self.femMesh.getDomainElements()), 3))
            for i, elem in enumerate(self.femMesh.getDomainElements()):
                centroids[i, :] = elem.centroid
        else:
            centroids = np.zeros((len(self.femMesh.getElementsByMeshTag([39, 40])), 3))
            for i, elem in enumerate(self.femMesh.getElementsByMeshTag([39, 40])):
                centroids[i, :] = elem.centroid

        nSamples = 4000

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        if nSamples < centroids.shape[0]:
            samples = centroids[np.random.choice(centroids.shape[0], nSamples, replace=False), :]
        else:
            samples = centroids

        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c='k', label='mesh')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # plot pixels of the atlas
        if nSamples < self.components['static']['voxelCoords'].shape[0]:
            samples = self.components['static']['voxelCoords'][
                      np.random.choice(self.components['static']['voxelCoords'].shape[0], nSamples, replace=False), :]
        else:
            samples = self.components['static']['voxelCoords']

        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c='b', label='atlas static')

        if self.useDynamicAtlas:
            # plot pixels of the atlas
            if nSamples < self.components['dynamic']['voxelCoords'].shape[0]:
                samples = self.components['dynamic']['voxelCoords'][
                          np.random.choice(self.components['dynamic']['voxelCoords'].shape[0], nSamples, replace=False), :]
            else:
                samples = self.components['dynamic']['voxelCoords']
            ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c='r', label='atlas vessels')

        if additionalPoints is not None:
            if nSamples < additionalPoints.shape[0]:
                samples = additionalPoints[np.random.choice(additionalPoints.shape[0], nSamples, replace=False), :]
            else:
                samples = additionalPoints
            ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c='g', label='additional points')

        ax.legend()
        plt.show()

    def createInterpAtlas2points(self, atlasVoxelCoords, points, filePrefix=''):
        """
        This version is using sparse form and multiprocessing
        create the interpolation matrix. It loads the matrix from a file if it exists. Otherwise will save a file after the computation
        Parameters
        ----------

        atlasVoxelCoords: numpy array (Nx3)
            coordenadas dos voxels do atlas

        filePrefix:string
            suffix added to the output file. Files to be created:
                - [outputDir][mesh_name_prefix][chunkFileNameSufix]_[filePrefix]_allChunks.txt:  text file containing a list of all chunk files
                - [outputDir][mesh_name_prefix][chunkFileNameSufix]_[filePrefix]_chunk[XXX]_of_[YYY].npz:  binary numpy file with the interpolation matrix of a chunk

        points: numpy array (Nx3)
        """

        print('    -> Creating interpolation matrix...')

        destinationElements = points

        chunkList = list(chunkfy(destinationElements, self.interpolationChunkSize))

        chunkFiles = self.outputBaseDir + self.filePrefix + '_03_interpMat_%s_allChunks.txt' % filePrefix

        calcMatrix = True
        if os.path.exists(chunkFiles):
            with open(chunkFiles, 'r') as f:
                if 'COMPLETED' in f.read():
                    calcMatrix = False

        if not calcMatrix:
            print('    -> Interpolation matrix already computed. Skipping... ')
            with open(chunkFiles, 'r') as f:
                chunkFileNames = [self.outputBaseDir + line.rstrip('\n') for line in f]
            # removes last line that contains 'COMPLETED'
            chunkFileNames.pop()

            # extract the first line that contains
            self.interpolationChunkSize = int(chunkFileNames.pop(0).split('=')[-1])
        else:
            chunkFileNames = []

            with open(chunkFiles, 'w') as f:
                f.write('CHUNK_SIZE=%d\n' % self.interpolationChunkSize)

            for i, chunk in enumerate(chunkList):

                chunkFileName = self.filePrefix + '_03_interpMat_%s_chunk%03d_of_%03d.npz' % (filePrefix, i + 1, len(chunkList))
                chunkFileNames.append(self.outputBaseDir + chunkFileName)

                if os.path.exists(self.outputBaseDir + chunkFileName):
                    print('Chunk %d of %d already computed. Skipping...' % (i + 1, len(chunkList)))
                else:
                    print('Processing chunk %d of %d' % (i + 1, len(chunkList)))
                    start = time.time()

                    # prepare data to run in parallel
                    args = [(elem, atlasVoxelCoords, self.interpolationStdDev) for elem in chunk]
                    with mp.Pool(processes=tools.nCores) as p:
                        rowList = p.starmap(computeInterpolationRow, args)

                    rows = []
                    cols = []
                    vals = []
                    for r, data in enumerate(rowList):
                        rows += [r, ] * len(data[0])  # already join lines of the same region if needed
                        cols += data[1].tolist()
                        vals += data[0].tolist()

                    interpMatrix = scipySparse.coo_matrix((vals, (rows, cols)), shape=(chunk.shape[0], atlasVoxelCoords.shape[0])).tocsr()
                    del rows, cols, vals

                    scipySparse.save_npz(self.outputBaseDir + chunkFileName, interpMatrix)

                    del interpMatrix
                    print('  -> time: %f s' % (time.time() - start))

                with open(chunkFiles, 'a') as f:
                    f.write(chunkFileName + '\n')

            with open(chunkFiles, 'a') as f:
                f.write('COMPLETED\n')

        return chunkFileNames

    def createInterpAtlas2Mesh(self, atlasVoxelCoords, filePrefix=''):
        """
        This version is using sparse form and multiprocessing
        create the interpolation matrix. It loads the matrix from a file if it exists. Otherwise will save a file after the computation
        Parameters
        ----------

        atlasVoxelCoords: numpy array (Nx3)
            coordenadas dos voxels do atlas

        filePrefix:string
            suffix added to the output file. Files to be created:
                - [outputDir][mesh_name_prefix][chunkFileNameSufix]_[filePrefix]_allChunks.txt:  text file containing a list of all chunk files
                - [outputDir][mesh_name_prefix][chunkFileNameSufix]_[filePrefix]_chunk[XXX]_of_[YYY].npz:  binary numpy file with the interpolation matrix of a chunk
        """

        print('    -> Creating interpolation matrix...')

        destinationElements = self.femMesh.getDomainElements()

        chunkList = list(chunkfy(destinationElements, self.interpolationChunkSize))

        chunkFiles = self.outputBaseDir + self.filePrefix + '_03_interpMat_%s_allChunks.txt' % filePrefix

        calcMatrix = True
        if os.path.exists(chunkFiles):
            with open(chunkFiles, 'r') as f:
                if 'COMPLETED' in f.read():
                    calcMatrix = False

        if not calcMatrix:
            print('    -> Interpolation matrix already computed. Skipping... ')
            with open(chunkFiles, 'r') as f:
                chunkFileNames = [self.outputBaseDir + line.rstrip('\n') for line in f]
            # removes last line that contains 'COMPLETED'
            chunkFileNames.pop()

            # extract the first line that contains
            self.interpolationChunkSize = int(chunkFileNames.pop(0).split('=')[-1])
        else:
            chunkFileNames = []

            with open(chunkFiles, 'w') as f:
                f.write('CHUNK_SIZE=%d\n' % self.interpolationChunkSize)

            for i, chunk in enumerate(chunkList):

                chunkFileName = self.filePrefix + '_03_interpMat_%s_chunk%03d_of_%03d.npz' % (filePrefix, i + 1, len(chunkList))
                chunkFileNames.append(self.outputBaseDir + chunkFileName)

                if os.path.exists(self.outputBaseDir + chunkFileName):
                    print('Chunk %d of %d already computed. Skipping...' % (i + 1, len(chunkList)))
                else:
                    print('Processing chunk %d of %d' % (i + 1, len(chunkList)))
                    start = time.time()

                    elementsTotal = []
                    stateVectorPosition = []
                    for j, elem in enumerate(chunk):
                        if elem.isRegion:
                            elementsTotal += elem.elements
                            stateVectorPosition += [j] * elem.nElements
                        else:
                            elementsTotal.append(elem)
                            stateVectorPosition.append(j)

                    sizeTotal = len(elementsTotal)
                    print("  ->  chunk size: %d" % sizeTotal)

                    stateVectorPosition = np.array(stateVectorPosition)

                    # prepare data to run in parallel
                    args = [(elem.centroid, atlasVoxelCoords, self.interpolationStdDev) for elem in elementsTotal]
                    with mp.Pool(processes=tools.nCores) as p:
                        rowList = p.starmap(computeInterpolationRow, args)

                    rows = []
                    cols = []
                    vals = []
                    for r, data in enumerate(rowList):
                        rows += [stateVectorPosition[r], ] * len(data[0])  # already join lines of the same region if needed
                        cols += data[1].tolist()
                        vals += data[0].tolist()

                    interpMatrix = scipySparse.coo_matrix((vals, (rows, cols)), shape=(sizeTotal, atlasVoxelCoords.shape[0])).tocsr()
                    del rows, cols, vals

                    scipySparse.save_npz(self.outputBaseDir + chunkFileName, interpMatrix)

                    del interpMatrix
                    print('  -> time: %f s' % (time.time() - start))

                with open(chunkFiles, 'a') as f:
                    f.write(chunkFileName + '\n')

            with open(chunkFiles, 'a') as f:
                f.write('COMPLETED\n')

        return chunkFileNames

    def _selectInterpMatrixRows(self, destinationElements):
        """
        find the rows of the interpolation matrix that must be used.
        Parameters
        ----------
        destinationElements: list of femElements to be considered.
        """

        self.interpolatedElementNumbers = np.array([elem.number for elem in destinationElements])

        for atlasComponent in self.components.values():
            atlasComponent['interpMatrixChunkRows'] = []
            destinationElementNumbers = np.array([elem.number for elem in destinationElements])
            for file in atlasComponent['interpChunkFileNames']:

                # open the file and read only the shape. Source:
                # https://github.com/scipy/scipy/blob/v1.5.2/scipy/sparse/_matrix_io.py#L75-L149
                with np.load(file) as f:
                    shape = f['shape']
                    nRows = shape[0]

                # find elements within the current chunk
                rowIdx = np.argwhere(destinationElementNumbers < nRows).flatten()

                if len(rowIdx) > 0:
                    # split array into two parts, one within the chunk and the rest
                    [chunkRows, destinationElementNumbers] = np.split(destinationElementNumbers, [rowIdx[-1] + 1])
                    atlasComponent['interpMatrixChunkRows'].append((file, chunkRows))

                destinationElementNumbers -= nRows

    def consolidateStatistics(self, normalizedTimeInstant=0.0):
        """
        builds the resulting average and covariance matrices, based on the interpolated statistics of its components.

        You have to run interpolateStatistics before this function

        Parameters
        ----------
        normalizedTimeInstant: float
            normalized time instant. Value between 0.0 and 1.0

        """
        if self.components['static']['avg'] is not None:
            self.avg = copy.copy(self.components['static']['avg'])

        if self.components['static']['covK'] is not None:
            self.covK = copy.copy(self.components['static']['covK'])
            ncols_Static = self.components['static']['covK'].shape[1]


        if self.useDynamicAtlas:
            # augment self.covK = [covK_static covK_dynamic]
            if self.covK is not None:
                self.covK = np.hstack((self.covK,np.zeros(self.components['dynamic']['covK'].shape)))

            tissueProp = tissuePropCalculator.TissueCalculator()
            rho_0,_ = tissueProp.getResistivity(tissueName='Blood', frequency_Hz=self.freq_Hz, uncertainty_Perc=0.2)

            #  apply electrical property of the blood in each vessel of the model
            for vessel in self.components['dynamic']['territories']:
                name = vessel['name']
                side = vessel['side']
                segmentID = vessel['segmentId']
                #volume = vessel['volume']

                _, rho = self.vesselDynamicModel.getPropVessel(vesselName=name, side=side, property=self.ElectricalProperty,timeNormalized=normalizedTimeInstant)
                _, rho_Std = self.vesselDynamicModel.getPropVessel(vesselName=name, side=side, property=self.ElectricalProperty + '_error',
                                                                   timeNormalized=normalizedTimeInstant)
                _, flowRate = self.vesselDynamicModel.getPropVessel(vesselName=name, side=side, property='flow_rate',timeNormalized=normalizedTimeInstant)

                # find all elements in current territory
                elementsFullListIDs = np.where(self.vascularTerritories['territoryID'] == segmentID)[0]
                elementsFullList= self.vascularTerritories['elemNbr'][elementsFullListIDs]

                # find the elements that are currently sectected for interpolation
                for i, elemNbr in enumerate(self.interpolatedElementNumbers):
                    if elemNbr in elementsFullList:

                        if self.avg is not None:
                            if self.components['dynamic']['avg'][i]>0.005:
                                tempAvg = rho * self.components['dynamic']['avg'][i]
                            else:
                                tempAvg = rho * 0.005
                                #tempAvg = rho_0 * (flowRate/volume)

                            self.avg[i] += tempAvg

                        if self.covK is not None:
                            tempCov = copy.copy(self.components['dynamic']['covK'][i])
                            tempCov[::3] *= rho_Std
                            tempCov[1::3] *= rho_Std
                            tempCov[2::3] *= rho

                            self.covK[i][ncols_Static:] += tempCov

    def interpolateStatistics(self, destinationElements, interpAVG=True, interpCOVK=True, avg_FileName=None, covK_FileName=None):
        """
        interpolates the statistics of the atlas to the mesh.

        Parameters
        ----------
        destinationElements: list of femElements to be considered.
        interpAVG: bool
            if True, interpolates the average
        interpCOVK: bool
            if True, interpolates the covK matrix
        avg_FileName: string
            file name of the resulting interpolated avg vector without extension. Used only if interpAVG=True.
                - if points to an existing file, then the vector is loaded instead of computed
                - if points to a non existing file, then the vector is computed and stored in the file
                - If None, then the vector is computed and not stored in a file
        covK_FileName: string
            file name of the resulting interpolated covK matrix without extension. Used only if interpCOVK=True.
                - if points to an existing file, then the matrix is loaded instead of computed
                - if points to a non existing file, then the matrix is computed and stored in the file
                - If None, then the matrix is computed and not stored in a file

        Returns
        -------
        out: list
            format: [interpolated_avg, interpolated_covK]. If one is not interpolated, None is returned in its position

        """

        self._selectInterpMatrixRows(destinationElements)

        out = [None, None]

        for atlasComponent in self.components.values():
            # saves covK and AvG in a file to speed up re-runs.
            if interpAVG:
                if avg_FileName is None:
                    out[0] = self._interpolateAVG(atlasComponent, fileName=None)
                else:
                    fileName = avg_FileName + '_' + atlasComponent['label'] + '.npy'
                    if os.path.exists(fileName):
                        out[0] = self._loadInterpolatedAVG(atlasComponent, fileName)
                    else:
                        out[0] = self._interpolateAVG(atlasComponent, fileName)

            if interpCOVK:
                if covK_FileName is None:
                    out[1] = self._interpolateCOVK(atlasComponent, fileName=None)
                else:
                    fileName = covK_FileName + '_' + atlasComponent['label'] + '.npy'
                    if os.path.exists(fileName):
                        out[1] = self._loadInterpolatedCOVK(atlasComponent, fileName)
                    else:
                        out[1] = self._interpolateCOVK(atlasComponent, fileName)

        return out

    def _interpolateAVG(self, componentDict, fileName=None):
        """
        interpolate the average to the mesh
        Parameters
        ----------
        fileName: string (optional)
            output file name containing the interpolated vector. if None (default), no file is saved
        componentDict: dict
            dictionary of the static or dynamic part of the atlas
        Returns
        -------
        avg : numpy array
            average
        """
        print('    -> Interpolating AVG...')
        start = time.time()
        if componentDict['avg'] is None:
            vectorPixel = np.load(componentDict['avgFile'])
            avgChunks = []

            for i, chunkInfo in enumerate(componentDict['interpMatrixChunkRows']):
                file, rows = chunkInfo
                print('Processing chunk %d of %d' % (i + 1, len(componentDict['interpMatrixChunkRows'])))
                interpMatrixChunk = scipySparse.load_npz(file)
                avgChunks.append(interpMatrixChunk[rows, :].dot(vectorPixel))
            componentDict['avg'] = np.concatenate(avgChunks)

        if fileName is not None:
            np.save(fileName, componentDict['avg'])
        print('        -> time interpolate AVG: %f s' % (time.time() - start))
        return componentDict['avg']

    def _loadInterpolatedAVG(self, componentDict, fileName):
        """
        load avg vector.
        Parameters
        ----------
        fileName: string
            avg file name
        componentDict: dict
            dictionary of the static or dynamic part of the atlas
        Returns
        -------
        avg : numpy array
            average
        """
        print('    -> Loading AVG...')
        start = time.time()
        componentDict['avg'] = np.load(fileName)
        print('        -> time load AVG: %f s' % (time.time() - start))
        return componentDict['avg']

    def _interpolateCOVK(self, componentDict, fileName=None):
        """
        interpolate the Kcov to the mesh
        Parameters
        ----------
        fileName: string (optional)
            output file name containing the interpolated vector. if None (default), no file is saved
        componentDict: dict
            dictionary of the static or dynamic part of the atlas
        Returns
        -------
        covK : numpy array
            covariance K matrix (  cov = K . K^T )
        """
        print('    -> Interpolating COVK...')
        start = time.time()
        if componentDict['covK'] is None:
            matrixPixel = np.load(componentDict['covKFile'])
            covChunks = []

            for i, chunkInfo in enumerate(componentDict['interpMatrixChunkRows']):
                file, rows = chunkInfo
                print('Processing chunk %d of %d' % (i + 1, len(componentDict['interpMatrixChunkRows'])))
                interpMatrixChunk = scipySparse.load_npz(file)
                covChunks.append(interpMatrixChunk[rows, :].dot(matrixPixel))
            componentDict['covK'] = np.concatenate(covChunks, axis=0)

        if fileName is not None:
            np.save(fileName, componentDict['covK'])
        print('        -> time interpolate COVK: %f s' % (time.time() - start))
        return componentDict['covK']

    def _loadInterpolatedCOVK(self, componentDict, fileName):
        """
        load covK matrix.
        Parameters
        ----------
        fileName: string
            avg file name
        componentDict: dict
            dictionary of the static or dynamic part of the atlas
        Returns
        -------
        covK : numpy array
        covK : numpy array
            covariance K matrix (  cov = K . K^T )
        """
        print('    -> Loading COVK...')
        start = time.time()
        componentDict['covK'] = np.load(fileName)
        print('        -> time load COVK: %f s' % (time.time() - start))
        return componentDict['covK']

    def generateSamples(self, nSamples, averageOnly=False, fileName=None, ensurePositiveRho=True, rhoMinLmit=0.0002):
        """
        generates samples of the atlas.
        Algorithm based on https://math.stackexchange.com/questions/446093/generate-correlated-normal-random-variables

        Parameters
        ----------
        nSamples: int
            number of samples. This value has no effect it averageOnly=True. in this case, nSamples=1

        averageOnly: bool
            if true, then the sample is just the average. In this case nSamples is set to 1

        fileName: string (optional)
            name of the file. In this file, each line is a sample. If None, no file is saved (default)

        Returns
        -------
        samples: 1D or 2D numpy array
            1D array:  if  averageOnly==True
            2D array:  if averageOnly==False.  Each column is a sample.
        """

        # target sample:     z=N(mean_z,Cov_z),   Cov_z=K.K^T
        # for that, first sample  x_sample = N(u=0.0,I)
        # The size of x_sample is the number of columns of K
        #  z_sample = mean_z + K * x_sample, where K is the factorization of Cov_z.
        # Algorithm based on https://math.stackexchange.com/questions/446093/generate-correlated-normal-random-variables

        if averageOnly:
            rhoSamples = self.avg
        else:
            # each columns is a sample.
            x_samples = np.random.normal(0.0, 1.0, (self.covK.shape[1], nSamples))
            rhoSamples = self.covK.dot(x_samples)

            # add the mean
            for i in range(nSamples):
                rhoSamples[:, i] += self.avg

        # ensure all resistivities are positive
        if ensurePositiveRho:
            rhoSamples[rhoSamples < rhoMinLmit] = rhoMinLmit

        if fileName is not None:
            np.savetxt(fileName, rhoSamples.T)

        if averageOnly:
            return rhoSamples
        elif nSamples == 1:
            return rhoSamples.flatten()
        else:
            return rhoSamples
