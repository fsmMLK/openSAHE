# -*- coding: utf-8 -*-
import os

# relative to staticComponent folder
paths = {'atlas': 'atlas/', 'input': 'inputData/', 'output': 'outputData/' }

# relative to staticComponent folder
singularityImgPath='../../sigularityImages/nipype_ANTS_SPM12_SingularityImg.simg'

# set number of processors
nCores = max(1,os.cpu_count() - 1)

def splitPath(path):
    """
    splits a path between its components
    Ex:  /path/to/file.ext
            dirName: /path/to/   <-- includes last /!!
            filePrefixName: file
            fileExtension: .ext
    Parameters
    ----------
    path: string
        string representing a path
    Returns
    -------
        [dirName, filePrefixName, fileExtension]
    """
    dirName = os.path.dirname(path) + '/'
    filename_w_ext = os.path.basename(path)
    filePrefixName, fileExtension = os.path.splitext(filename_w_ext)
    return [dirName, filePrefixName, fileExtension]

class plane:
    """  plane equation
       ax+by+cz+d=0
       Normal=(a,b,c)
       d=- normal (dot) P1
    https://tutorial.math.lamar.edu/classes/calciii/eqnsofplanes.aspx
    """

    def __init__(self, P1, P2, P3):
        self.P1 = np.array(P1)
        self.P2 = np.array(P2)
        self.P3 = np.array(P3)

        s1 = self.P2 - self.P1
        s2 = self.P3 - self.P1
        self.normal = np.cross(s1, s2)
        self.normal /= np.linalg.norm(self.normal)
        self.a = self.normal[0]
        self.b = self.normal[1]
        self.c = self.normal[2]
        self.d = -np.dot(self.P1, self.normal)

    def evalPoint(self, Point):
        """  computes the plane equation ax+by+cz+d at a given point.
                - The point is in the plane if the result is zero
                - The point is above the plane if the result is positive
                - The point is below the plane if the result is negative
        """
        return self.a * Point[0] + self.b * Point[1] + self.c * Point[2] + self.d

    def dist(self, Point):
        # https://mathinsight.org/distance_point_plane#:~:text=The%20shortest%20distance%20from%20a,as%20a%20gray%20line%20segment.
        # in my case self.normal already has norm 1, so I don't have to divide by sqrt(a^2+b^2+c^2)
        return np.abs(self.signedDist(Point))

    def signedDist(self, Point):
        # https://mathinsight.org/distance_point_plane#:~:text=The%20shortest%20distance%20from%20a,as%20a%20gray%20line%20segment.
        # in my case self.normal already has norm 1, so I don't have to divide by sqrt(a^2+b^2+c^2)
        return self.evalPoint(Point)

    def evalCube(self, cube):
        X, Y, Z = np.meshgrid(np.arange(cube.shape[0]), np.arange(cube.shape[1]), np.arange(cube.shape[2]), indexing='ij')
        return self.a * X + self.b * Y + self.c * Z + self.d
