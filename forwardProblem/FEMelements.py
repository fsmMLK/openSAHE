#!/bin/python
# -*- coding: utf-8 -*-
"""
Finite elements method related classes. This file defines the element types used in the code

"""
import multiprocessing as mp

import numpy as np

try:
    from nptyping import Array
except ImportError:
    pass

from scipy import linalg as scipyLinalg
from scipy import sparse as scipySparse

import tools


def localRefSystem2D(v1,  # type: Array[float]
                     v2  # type: Array[float]
                     ):
    """
    given 2 non parallel vectors in R^3, returns a orthonormal local base in R^2 of the space spanned by
    the 2 vectors. The base is defines so that v1_local is parallel to v1 and v2_local has positive second component

    Parameters
    ----------

    v1,v2 : numpy 1D array
        vectors in R^3

    Returns
    -------

    v1local,v2local : numpy 1D array
        local vectors

    """
    e1 = v1 / scipyLinalg.norm(v1)
    e2 = v2 - v2.dot(e1) * e1
    e2 = e2 / scipyLinalg.norm(e2)
    base = np.hstack((e1[:, None], e2[:, None]))
    v1local = base.T.dot(v1)
    v2local = base.T.dot(v2)
    return [v1local, v2local]


def areaTriangle(node1, node2, node3):
    """
    computes the area of a triangle, given the coordinates in R3 of its nodes
    Parameters
    ----------
        node1,2,3: numpy array in R3

    Returns
    -------
        area: float
            area of the triangle
    """
    v1 = node2 - node1
    v2 = node3 - node1
    return 0.5 * scipyLinalg.norm(np.cross(v1, v2))


class SimplexUniformRho():
    """
    base class of uniform resistivity fem elements.
    """

    def __init__(self, elemNbr,  # type: int
                 dimension,  # type: int
                 connectivity,  # type: Array[int]
                 coords,  # type: Array[float]
                 rho,  # type float
                 isSparse=False,  # type: bool
                 propertiesDict=None  # type: dict
                 ):
        """
        
        Parameters
        ----------
        elemNbr: int
                number of the element
        
        dimension: int {1,2,3}
                dimension of the simplex
        
        connectivity: 1D numpy array
                nodes of the element, in global terms. the local order of the nodes will be
                the same of connectivity input
                
        coords: 2D numpy array
                each line is composed by 3 columns, X, Y and Z of the node. this matrix should contain all
                nodes of the FemModel. The function will extract only the needed lines
                
        rho: float
                resistivity of the element

        isSparse: bool, optional
                computes Local matrix as sparse

        propertiesDict: dictionary, optional
                dictionary containing the properties of the simplex:
                'physical': physical entity group
                'region': region of the element
        """
        self.number = elemNbr
        self.dim = dimension
        self.type = 'simplex'.lower()
        self.nNodes = connectivity.shape[0]
        self.connectivity = connectivity.astype(int)
        self.coords = coords[connectivity, :]
        self.centroid = np.mean(self.coords, axis=0)
        self.propertiesDict = propertiesDict
        self.rho = rho
        self.Kgeom = np.array([])  # type: Array[float]
        self.isSparse = isSparse
        self.isRegion = False

    def saveKgeom(self, fileName,  # type: str
                  binary=False  # type: bool
                  ):
        """
        save geometric component of the matrix to a text file.

        Parameters
        ----------
        fileName: str
            file path
        binary: bool, optional
            save in binary format. Used only if matrix is not sparse. Default: False
        """
        if self.isSparse:
            scipySparse.save_npz(fileName, self.Kgeom, compressed=True)

        else:
            if binary:
                np.save(fileName, self.Kgeom)
            else:
                np.savetxt(fileName, self.Kgeom)

    def setRho(self, rho  # type: float
               ):
        """
        set the resistivity

        Parameters
        ----------
        rho: float
            resistivity value
        """
        self.rho = rho

    def calcSimplexVolume(self):
        """
        Compute the volume of the simplex. 1D: Length, 2D: area, 3D: volume
        """
        vol = -1

        if self.dim == 1:
            vol = scipyLinalg.norm(self.coords[1, :] - self.coords[0, :])
        if self.dim == 2:
            v1 = self.coords[1, :] - self.coords[0, :]
            v2 = self.coords[2, :] - self.coords[0, :]
            vol = 0.5 * scipyLinalg.norm(np.cross(v1, v2))
        if self.dim == 3:
            v1 = self.coords[1, :] - self.coords[0, :]
            v2 = self.coords[2, :] - self.coords[0, :]
            v3 = self.coords[3, :] - self.coords[0, :]

            vol = (1.0 / 6.0) * scipyLinalg.norm(np.dot(np.cross(v1, v2), v3))

            # V2 = np.hstack((self.coords,np.ones((4,1))))  # vol = (1.0 / 6.0) * abs(scipyLinalg.det(V2))

        if vol < 1e-12:
            print("Warning: element %d with small volume: %e (GmshElmNbr %d)" % (self.number,vol,self.propertiesDict['gmshElemNbr']))
            print("Centroid: x=%f  y=%f  z=%f" % (self.centroid[0],self.centroid[1],self.centroid[2]))

        if vol < 0:
            print("Warning: element %d with negative volume: %e  (GmshElmNbr %d)" % (self.number,vol,self.propertiesDict['gmshElemNbr']))
            print("Centroid: x=%f  y=%f  z=%f" % (self.centroid[0],self.centroid[1],self.centroid[2]))

        return vol

    def getBbox(self):
        """
        Retuns the boundingbox of the element

        Returns
        -------
        listLimits: list of np arrays
            list of limits in the form   [ [minX, minY, min Z] , [maxX, maxY, maxZ] ]

        """
        minimum = np.min(self.coords, axis=0)
        maximum = np.max(self.coords, axis=0)
        return [minimum, maximum]

    def getAspectRatio(self):
        """
        Retuns the aspect ratio of the simplex

        Returns
        -------
        aspect ratio: float
            value between 0.0 and 1.0
                0.0: zero-volume element
                1.0: equilateral simplex (equilateral triangle or regular tetrahedron)
        """
        if self.dim == 1:
            L = scipyLinalg.norm(self.coords[0, :] - self.coords[1, :])
            if L == 0:
                ratio = 0.0
            else:
                ratio = 1.0

        if self.dim == 2:
            a = scipyLinalg.norm(self.coords[0, :] - self.coords[1, :])
            b = scipyLinalg.norm(self.coords[0, :] - self.coords[2, :])
            c = scipyLinalg.norm(self.coords[1, :] - self.coords[2, :])
            area = areaTriangle(self.coords[0, :], self.coords[1, :], self.coords[2, :])
            semiPerimeter = (a + b + c)/2.0

            if area == 0:
                ratio = 0.0
            else:
                if area < 1e-12:
                    print("Warning: element %d with small area: %e (GmshElmNbr %d)" % (self.number,area,self.propertiesDict['gmshElemNbr']))
                    print("Centroid: x=%f  y=%f  z=%f" % (self.centroid[0],self.centroid[1],self.centroid[2]))

                # https://www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
                Circumradius = a * b * c / (4.0 * area)
                # https://www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-incircle
                Inradius = area / semiPerimeter

                ratio = 2.0 * Inradius / Circumradius

        if self.dim == 3:

            # Inradius:   https://en.wikipedia.org/wiki/Tetrahedron#Inradius
            # area of each face
            A1 = areaTriangle(self.coords[1, :], self.coords[2, :], self.coords[3, :])
            A2 = areaTriangle(self.coords[0, :], self.coords[2, :], self.coords[3, :])
            A3 = areaTriangle(self.coords[0, :], self.coords[1, :], self.coords[3, :])
            A4 = areaTriangle(self.coords[0, :], self.coords[1, :], self.coords[2, :])
            volume = self.calcSimplexVolume()

            if volume == 0:
                ratio = 0.0
            else:
                if volume < 1e-12:
                    print("Warning: element %d with small volume: %e (GmshElmNbr %d)" % (self.number,volume,self.propertiesDict['gmshElemNbr']))
                    print("Centroid: x=%f  y=%f  z=%f" % (self.centroid[0],self.centroid[1],self.centroid[2]))

                Inradius = 3.0 * volume / (A1 + A2 + A3 + A4)

                # Circumradius    https://en.wikipedia.org/wiki/Tetrahedron#Circumradius
                # Lenghts
                a = scipyLinalg.norm(self.coords[1, :] - self.coords[0, :])
                A = scipyLinalg.norm(self.coords[2, :] - self.coords[3, :])
                b = scipyLinalg.norm(self.coords[2, :] - self.coords[0, :])
                B = scipyLinalg.norm(self.coords[1, :] - self.coords[3, :])
                c = scipyLinalg.norm(self.coords[3, :] - self.coords[0, :])
                C = scipyLinalg.norm(self.coords[1, :] - self.coords[2, :])

                Circumradius = np.sqrt((A * a + B * b + C * c) * (-A * a + B * b + C * c) * (A * a - B * b + C * c) * (A * a + B * b - C * c)) / (
                      24 * volume)

                # http://support.moldex3d.com/r15/en/modelpreparation_reference-pre_meshqualitydefinition.html
                ratio = 3.0 * Inradius / Circumradius

        return ratio


class LinearTriangle(SimplexUniformRho):
    """
    3-node Linear triangle element, with uniform resistivity/conductivity
    """

    def __init__(self, elemNbr,  # type: int
                 connectivity,  # type: Array[int]
                 coords,  # type: Array[float]
                 rho,  # type float
                 height2D=1.0,  # type: float
                 propertiesDict=None  # type: dict
                 ):
        """
        3-node Linear triangle element, with uniform resistivity/conductivity
        
        Parameters
        ----------
        elemNbr: int
                number of the element
        
        connectivity: 1D numpy array
                nodes of the element, in global terms. the local order of the nodes will be the same of
                connectivity input
                
        coords: 2D numpy array
                each line is composed by 3 columns, X, Y and Z of the node. this matrix should contain all nodes
                of the FemModel. The function will extract only the needed lines
                
        rho: float
                resistivity of the element
                
        height2D: float
                associated height of the triangular element
        propertiesDict: dictionary, optional
                dictionary containing the properties of the simplex:
                'physical': physical entity group
                ''
        """
        dimension = 2
        super().__init__(elemNbr, dimension, connectivity, coords, rho, False, propertiesDict)
        self.type = '3-node triangle, linear'.lower()
        self.height2D = height2D
        self.area = self.calcSimplexVolume()
        self._calc_localKgeom()

    def _calc_localKgeom(self):
        """
        Compute the geometric component of the local stiffness matrix
        """
        # passing the vectors to a local system of coordinates such that
        # e_1 and e_2 are contained the plane of the triangle
        v2 = self.coords[1, :] - self.coords[0, :]
        v3 = self.coords[2, :] - self.coords[0, :]
        v1local = np.array([0, 0])
        [v2local, v3local] = localRefSystem2D(v2, v3)

        M = np.ones([3, 3])
        M[:, 1:] = np.vstack([v1local, v2local, v3local])

        F = scipyLinalg.inv(M)[1:, :]

        # does not need to divide by (2xArea)^2 bc I am inverting M directly
        self.Kgeom = self.height2D * self.area * np.dot(F.T, F)


class LinearTetrahedron(SimplexUniformRho):
    """
    4-node Linear tetrahedron element, with uniform resistivity/conductivity
    """

    def __init__(self, elemNbr,  # type: int
                 connectivity,  # type: Array[int]
                 coords,  # type: Array[float]
                 rho,  # type float
                 propertiesDict=None  # type: dict
                 ):
        """
        4-node Linear tetrahedron element, with uniform resistivity/conductivity
        
        Parameters
        ----------
        elemNbr: int
                number of the element
        
        connectivity: 1D numpy array
                nodes of the element, in global terms. the local order of the nodes will be
                the same of connectivity input
                
        coords: 2D numpy array
                each line is composed by 3 columns, X, Y and Z of the node. this matrix should contain all nodes of
                the FemModel. The function will extract only the needed lines
                
        rho: float
                resistivity of the element
                
        propertiesDict: dictionary
                dictionary containing the properties of the simplex:
                'physical': physical entity group
                ''
        """
        dimension = 3
        super().__init__(elemNbr, dimension, connectivity, coords, rho, False, propertiesDict)
        self.type = '4-node tetrahedron, linear'.lower()
        self.volume = self.calcSimplexVolume()
        self._calc_localKgeom()

    def _calc_localKgeom(self):
        """
        Compute the geometric component of the local stiffness matrix
        """
        M = np.hstack((np.ones([4, 1]), self.coords))
        F = scipyLinalg.inv(M)[1:, :]

        # does not need to divide by (6xVolume)^2 bc I am inverting M directly
        self.Kgeom = self.volume * np.dot(F.T, F)


class UniformRhoRegion():
    """
    Uniform rho region element.
    """

    def __init__(self, dimension,  # type: int
                 elemNbr,  # type: int
                 connectivities,  # type: Array[int]
                 coords,  # type: Array[float]
                 rho,  # type float
                 height2D,  # type: float
                 isSparse=False,  # type: bool
                 propertiesDict=None  # type: dict
                 ):
        """
        
        Parameters
        ----------
        
        dimension : int {2,3}
                dimension of the element
                
        elemNbr: int
                number of the element
        
        connectivities: 2D numpy array
                nodes of the elements, in global terms. the local order of the nodes will be
                the same of connectivity input
                lines: elements that compose the region
                cols: connectivity of each element
                
        coords: 2D numpy array
                each line is composed by 3 columns, X, Y and Z of the node. this matrix should contain all nodes of
                the FemModel. The function will extract only the needed lines
                
        rho: float
                resistivity of the element
                
        height2D: float
                associated height of the triangular element. Used only if dim=2

        isSparse: bool, optional
                computes Local matrix as sparse

        propertiesDict: dictionary, optional
                dictionary containing the properties of the simplex:
                'physical': physical entity group
        """
        self.number = elemNbr
        self.dim = dimension
        self.type = 'uniform region, linear'.lower()
        self.propertiesDict = propertiesDict
        self.rho = rho
        self.isSparse = isSparse
        self.isRegion = True

        if self.dim == 2:
            self.height2D = height2D
        else:
            self.height2D = None

        # build local connectivity
        self.connectivity, connectivityLocal = np.unique(connectivities, return_inverse=True)
        self.connectivity = self.connectivity.astype(int)
        self.connectivityElementsLocal = connectivityLocal.reshape(len(connectivityLocal) // connectivities.shape[1], connectivities.shape[1])

        # total number of elements and nodes of the region
        self.nNodes = self.connectivity.shape[0]
        self.nElements = self.connectivityElementsLocal.shape[0]

        self.coords = coords[self.connectivity, :]

        self.centroid = np.mean(self.coords, axis=0)

        self.elements = None
        self.appendElements()

        if self.isSparse:
            self._calc_localKgeom_Sparse()
        else:
            self._calc_localKgeom()

    def saveKgeom(self, fileName,  # type: str
                  binary=False  # type: bool
                  ):
        """
        save geometric component of the matrix to a text file.

        Parameters
        ----------
        fileName: str
            file path
        binary: bool, optional
            save in binary format. Used only if matrix is not sparse. Default: False
        """
        if self.isSparse:
            scipySparse.save_npz(fileName, self.Kgeom, compressed=True)

        else:
            if binary:
                np.save(fileName, self.Kgeom)
            else:
                np.savetxt(fileName, self.Kgeom)

    def setRho(self, rho  # type: float
               ):
        """
        set the resistivity of the region and all sub elements

        Parameters
        ----------
        rho: float
            resistivity value
        """
        self.rho = rho
        for e in self.elements:
            e.rho = rho

    def appendElements(self):
        """
        Create elements composing the region.
        """

        if self.dim == 2:
            args = [(i, c, self.coords, self.rho, self.height2D, self.propertiesDict.copy()) for i, c in enumerate(self.connectivityElementsLocal)]
            for c in args:
                c[5]['gmshElemNbr'] = c[5]['gmshElemNbr'][c[0]]
            with mp.Pool(processes=tools.nCores) as p:
                self.elements = p.starmap(LinearTriangle, args)

        if self.dim == 3:
            args = [(i, c, self.coords, self.rho, self.propertiesDict.copy()) for i, c in enumerate(self.connectivityElementsLocal)]
            for c in args:
                c[4]['gmshElemNbr'] = c[4]['gmshElemNbr'][c[0]]
            with mp.Pool(processes=tools.nCores) as p:
                self.elements = p.starmap(LinearTetrahedron, args)

    def _calc_localKgeom(self):
        """
        Compute the geometric component of the local stiffness matrix
        """
        self.Kgeom = np.zeros([self.nNodes, self.nNodes])
        for e in self.elements:
            self.Kgeom[np.ix_(e.connectivity, e.connectivity)] += e.Kgeom

    def _calc_localKgeom_Sparse(self):
        """
        Compute the geometric component of the local stiffness matrix in sparse form.
        """
        # find total number of elements
        count = 0
        for e in self.elements:
            temp = scipySparse.coo_matrix(e.Kgeom)
            count += temp.nnz

        data = np.zeros(count)
        rowIdx = np.zeros(count)
        colIdx = np.zeros(count)

        position = 0
        for e in self.elements:
            temp = scipySparse.coo_matrix(e.Kgeom)
            data[position:position + temp.nnz] = temp.data
            rowIdx[position:position + temp.nnz] = e.connectivity[temp.row]
            colIdx[position:position + temp.nnz] = e.connectivity[temp.col]
            position += temp.nnz

        self.KgeomSp = scipySparse.coo_matrix((data, (rowIdx, colIdx)), shape=(self.nNodes, self.nNodes))


class CoreElectrodeHua():
    """
    Core component of Hua's complete electrode model
      2D: rectangular region, compressed into 3 nodes.
      3D: hexahedron region, compressed into 4 nodes.
      Virtual node is the last one

    """

    def __init__(self, dimension,  # type: int
                 elemNbr,  # type: int
                 connectivity,  # type: Array[int]
                 coords,  # type: Array[float]
                 rhoT,  # type float
                 height2D=1.0,  # type: float
                 propertiesDict=None  # type: dict
                 ):
        """
        Core component of Hua's electrode model
        2D: rectangular region, compressed into 3 nodes.
        3D: hexahedron region, compressed into 4 nodes.
        Virtual node is the last one
        
        Parameters
        ----------
        
        dimension : int {2,3}
                dimension of the model
        elemNbr: int
                number of the element
        
        connectivity: 1D numpy array
                nodes of the element, in global terms. the local order of the nodes will be
                the same of connectivity input
                The last node is the number of the virtual node.
                
        coords: 2D numpy array
                each line is composed by 3 columns, X, Y and Z of the node. this matrix should contain
                all nodes of the FemModel, except the virtual node. The function will extract only the needed lines
                
        rho: float
                electrode parameter. This is in fact [rho*t] from hua's model
                
        height2D: float
                associated height of the triangular element. Used only if dim=2
                
        propertiesDict: dictionary
                dictionary containing the properties of the simplex:
                'physical': physical entity group
        """
        self.number = elemNbr
        self.dim = dimension
        self.type = 'completeElectrode core Hua'.lower()
        self.connectivity = connectivity.astype(int)
        self.coords = coords[connectivity[:-1], :]
        self.centroid = np.mean(self.coords, axis=0)
        self.isSparse = False
        self.isRegion = False

        if self.dim == 2:
            self.nNodes = 3
            self.height2D = height2D
        else:
            self.nNodes = 4
            self.height2D = None
        self.propertiesDict = propertiesDict
        self.rhoT = rhoT

        self.Kgeom = None
        self._calc_localKgeom()

    def _calc_localKgeom(self):
        """
        Compute the geometric component of the local stiffness matrix
        """
        if self.dim == 2:
            length = scipyLinalg.norm(self.coords[1, :] - self.coords[0, :])
            self.Kgeom = (length * self.height2D / 6.0) * np.array([[2.0, 1.0, -3.0], [1.0, 2.0, -3.0], [-3.0, -3.0, 6.0]])
        else:
            v1 = self.coords[1, :] - self.coords[0, :]
            v2 = self.coords[2, :] - self.coords[0, :]
            area = 0.5 * scipyLinalg.norm(np.cross(v1, v2))
            self.Kgeom = (area / 3.0) * np.array([[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, -1.0], [0.0, 0.0, 1.0, -1.0], [-1.0, -1.0, -1.0, 3.0]])


class CompleteElectrodeHua(UniformRhoRegion):
    """
    Hua's complete electrode model
    """

    def __init__(self, dimension,  # type: int
                 elemNbr,  # type: int
                 connectivities,  # type: Array[int]
                 coords,  # type: Array[float]
                 rhoT,  # type float
                 height2D,  # type: float
                 virtualNodeNbr,  # type: int
                 isSparse=False,  # type: bool
                 propertiesDict=None  # type: dict
                 ):
        """
        Hua electrode model

        Parameters
        ----------

        dimension : int {2,3}
                dimension of the electrode model

        elemNbr: int
                number of the element. this number also considers the elements of the domain

        connectivities: 2D numpy array
                nodes of the elements, in global terms. the local order of the nodes will be
                the same of connectivity input. This array must not contain the virtual node.
                    lines: elements that compose the electrode
                    cols: connectivity of each element

        coords: 2D numpy array
                each line is composed by 3 columns, X, Y and Z of the node. this matrix should contain
                all nodes of the FemModel. The function will extract only the needed lines

        rhoT: float
                electrode parameter. This is in fact [rho*t] from hua's model!

        height2D: float
                associated height of the triangular element. Used only if dim=2

        virtualNodeNbr: int
                Virtual node of the electrode, in global terms. this node will be the last

        isSparse: bool, optional
                computes Local matrix as sparse

        propertiesDict: dictionary
                dictionary containing the properties of the element
        """

        self.number = elemNbr
        self.dim = dimension
        self.type = 'completeElectrode Hua'.lower()
        self.propertiesDict = propertiesDict
        self.rhoT = rhoT
        self.isSparse = isSparse
        self.isRegion = True

        if self.dim == 2:
            self.height2D = height2D
        else:
            self.height2D = None

        # register the virtual node
        self.virtualNodeNbr = virtualNodeNbr
        connectivities = np.hstack((connectivities, self.virtualNodeNbr * np.ones([connectivities.shape[0], 1])))

        # build local connectivity
        self.connectivity, connectivityLocal = np.unique(connectivities, return_inverse=True)
        self.connectivity = self.connectivity.astype(int)
        self.connectivityElementsLocal = connectivityLocal.reshape(len(connectivityLocal) // connectivities.shape[1], connectivities.shape[1])

        # total number of elements and nodes of the region
        self.nNodes = self.connectivity.shape[0]
        self.nElements = self.connectivityElementsLocal.shape[0]

        self.coords = coords[self.connectivity[:-1], :]  # does not contain the coords of the virtual node!

        self.centroid = np.mean(self.coords, axis=0)

        self.elements = None
        self.appendElements()

        if self.isSparse:
            self._calc_localKgeom_Sparse()
        else:
            self._calc_localKgeom()

    def appendElements_parrallel_SLOW(self):
        """
        Create elements composing the region.

        This function uses multiprocessing.Pool method. It seems very slow for electrodes. I am using another version that is faster.
        """

        args = [(self.dim, i, c, self.coords, self.rhoT, self.height2D, self.propertiesDict.copy()) for i, c in
                enumerate(self.connectivityElementsLocal)]
        for c in args:
            c[6]['gmshElemNbr'] = c[6]['gmshElemNbr'][c[1]]
        with mp.Pool(processes=tools.nCores) as p:
            self.elements = p.starmap(CoreElectrodeHua, args)

        self.nElements = len(self.elements)


    def appendElements(self):
        """
        Create elements composing the region.

        This functions does not use multiprocessing, but is faster.
        """
        self.elements=[]

        for i, c in enumerate(self.connectivityElementsLocal):
            proPDict = self.propertiesDict.copy()
            proPDict['gmshElemNbr'] = proPDict['gmshElemNbr'][i]
            self.elements.append(CoreElectrodeHua(self.dim,i,c,self.coords, self.rhoT, self.height2D, proPDict))

        self.nElements = len(self.elements)
