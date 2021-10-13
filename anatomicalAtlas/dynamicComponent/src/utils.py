# -*- coding: utf-8 -*-
import os

# relative to dynamicComponent folder
paths = {'atlas': 'atlas/', 'input': 'inputData/', 'output': 'outputData/' }

# relative to dynamicComponent folder
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

