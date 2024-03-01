# openSAHE - Open Source Statistical Anatomical Atlas of the Human head for Electrophysiology Applications

Volume conductor problems in cerebral electrophysiology and bioimpedance do not have analytical solutions for nontrivial geometries and require a 3D model of the head and its electrical properties for solving the associated PDEs numerically.
Ideally, the model should be made with patient-specific information. In clinical practice, this is not always the case and an average head model is often used. Also, the electrical properties of the tissues might not be completely known due to natural variability. Anatomical atlases are important tools for in silico studies on cerebral circulation and electrophysiology that require statistically consistent data, e.g., machine learning, sensitivity analyses, and as a benchmark to test inverse problem solvers.

The objective of this work is to develop a 4D (3D+T) statistical anatomical atlas of the electrical properties of the upper part of the human head for cerebral electrophysiology and bioimpedance applications. 

The atlas was constructed based on 3D magnetic resonance images (MRI) of 107 human individuals and comprises the electrical properties of the main internal structures and can be adjusted for specific electrical frequencies. T1w+T2w MRI images were used to segment the main structures of the head while angiography MRI was used to segment the main artery. The proposed atlas also comprises a time-varying model of arterial brain circulation, based on the solution of the Navier-Stokes equation in the main arteries and their vascular territories.


## Citing the atlas

Please cite this article in your work

Moura, Fernando S, Beraldo, Roberto G, Ferreira, Leonardo A, & Siltanen, Samuli. (2021). Anatomical atlas of the upper part of the human head for electroencephalography and bioimpedance applications. Physiological Measurement, 42(10). 105015. https://doi.org/10.1088/1361-6579/ac3218


```
@article{Moura_2021,
	author = {Fernando S Moura and Roberto G Beraldo and Leonardo A Ferreira and Samuli Siltanen},
	title = {Anatomical atlas of the upper part of the human head for electroencephalography and bioimpedance applications},
	journal = {Physiological Measurement},
	doi = {10.1088/1361-6579/ac3218},
	url = {https://doi.org/10.1088/1361-6579/ac3218},
	year = 2021,
	month = {oct},
	publisher = {{IOP} Publishing},
	volume = {42},
	number = {10},
	pages = {105015}
}
```


If you use the source code, please also cite

Moura, Fernando S, Beraldo, Roberto G, Ferreira, Leonardo A, & Siltanen, Samuli. (2021). openSAHE: Open Source Statistical Anatomical Atlas of the Human head for Electrophysiology Applications (source code) (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.5567086

```
@software{moura_fernando_s_2021_5567086,
  author       = {Moura, Fernando S and Beraldo, Roberto G and Ferreira, Leonardo A and Siltanen, Samuli},
  title        = {{openSAHE: Open Source Statistical Anatomical Atlas of the Human head for Electrophysiology Applications (source code)}},
  month        = oct,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.5567086},
  url          = {https://doi.org/10.5281/zenodo.5567086}
}
```

If you use the precomputed atlases, please also cite

Moura, Fernando S, Beraldo, Roberto G, Ferreira, Leonardo A, & Siltanen, Samuli. (2021). openSAHE: Open Source Statistical Anatomical Atlas of the Human head for Electrophysiology Applications (precomputed atlases) (1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5559624

```
@dataset{moura_fernando_s_2021_5559624,
  author       = {Moura, Fernando S and Beraldo, Roberto G and Ferreira, Leonardo A and Siltanen, Samuli},
  title        = {{openSAHE: Open Source Statistical Anatomical Atlas  of the Human head for Electrophysiology Applications (precomputed atlases)}},
  month        = oct,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.5559624},
  url          = {https://doi.org/10.5281/zenodo.5559624}
}
```


## Precomputed atlases used in the publication

Precomputed atlases used in the publication (see above) can be found at 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5559624.svg)](https://doi.org/10.5281/zenodo.5559624)

Source code version used for the publication (see above) can be found at

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5567086.svg)](https://doi.org/10.5281/zenodo.5567086)

## Installation

This toolbox was developed in a linux machine. No tests were performed in other OSs.

#### Requirements

The toolbox requires the following software

  - [Julia](https://julialang.org/downloads/)
  - [OpenBF](https://github.com/INSIGNEO/openBF)
  - [Intel Python 3.8+](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/distribution-for-python.html) via [conda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html)
  - [SingularityCE](https://github.com/sylabs/singularity)
  - neurodocker image created with singularityCE (see instructions below)

#### Intel python (current: kubuntu 20-04)

Instructions for miniconda. Anaconda users can follow the same instructions

1. Miniconda install. Download [install script](https://docs.conda.io/en/latest/miniconda.html) 

2. run the installer

~~~
sudo bash Miniconda3-latest-Linux-x86_64.sh
~~~

3. Initialize conda to add its initialization to your .bashrc

~~~
/opt/miniconda3/bin/conda init
~~~

4. Add Intel' python channel. (other option is to edit ```.condarc``` created in your home folder. see below)

~~~
conda config --add channels intel
~~~


5. (optional but recommended =) ) Avoid activating conda's (base) in all terminals. Open (or create) a 
   file ~/.condarc and paste the following:

~~~
channels:        <-- channels to be considered priority channels are on top.
 - intel
 - conda-forge
 - defaults
auto_activate_base: false   <-- disable (base) auto load
~~~

6. Create a new python environment for the atlas

~~~
conda create -n atlasIntelPython_3 pyqt pyyaml matplotlib nipype nibabel meshio lxml pycairo psutil nptyping tornado simpleitk scikit-image pandas jill
~~~


7. If matplotlib shows the following error:

~~~
Bad key "text.kerning_factor" on line 4 in /home/samyak/anaconda3/lib/python3.7/site-packages/matplotlib/mpl-data/stylelib/_classic_test_patch.mplstyle. You probably need to get an updated matplotlibrc file from https://github.com/matplotlib/matplotlib/blob/v3.1.3/matplotlibrc.template or from the matplotlib source distribution
 ~~~

  Do the following steps ( [source](https://stackoverflow.com/questions/61171307/jupyter-notebook-shows-error-message-for-matplotlib-bad-key-text-kerning-factor) ) :
   
     1- Go to ~/anaconda3/lib/python3.7/site-packages/matplotlib/mpl-data/stylelib/
     2- open _classic_test_patch.mplstyle and comment out the line of text.kerning_factor:6

#### Julia and openBF

(Tested in Kubuntu 20.04)

1. install ```Jill``` using pip3.

~~~
pip3 install jill -U
~~~

 This comand will add a few binaries inside ~/.
local/bin/


2. Install julia

~~~
~/.local/bin/jill install
~~~

3. Install openBF. Run julia and type the following

~~~
julia> ]
pkg> add https://github.com/INSIGNEO/openBF
~~~

4. (optional) if you want to update openBF:

~~~
julia> ]
pkg> update openBF
~~~

5. (optional) if you want to test it

~~~
julia> ]
pkg> test openBF
~~~

6. Open the console and find julia's executable using ```which``` command

~~~
$which julia
path/to/juliaExe/julia
~~~

7. Open forwardProblem/tools.py and add the path to the executable to the ```juliaExe``` variable

~~~
juliaExe = 'path/to/juliaExe/julia'
~~~

8. Look for the folder where the  package was installed. In linux machine it is located at 

~~~
~/.julia/packages/openBF/XXXXX/
~~~

9. Open forwardProblem/tools.py and add this path to the ```openBFDir``` variable

~~~
openBFDir = '~/.julia/packages/openBF/XXXXX/'
~~~

#### Neurodocker Singularity Image creation

As of today (Oct/2021) singularity does not have binaries for Kubuntu. The following instructions compiles 
singularity in your machine (kubuntu 20.04, Oct/2021). You can skip these steps if you can install singularity using 
other methods.
installed versions: singularity-ce-3.8.3, go v. 1.17

1. dependencies
   ~~~
   sudo apt-get update && sudo apt-get install -y build-essential uuid-dev libgpgme-dev squashfs-tools libseccomp-dev wget pkg-config git cryptsetup-bin
   ~~~

2. install go
~~~
export VERSION=1.17 OS=linux ARCH=amd64
wget https://dl.google.com/go/go$VERSION.$OS-$ARCH.tar.gz
sudo tar -C /usr/local -xzvf go$VERSION.$OS-$ARCH.tar.gz
rm go$VERSION.$OS-$ARCH.tar.gz
~~~

3. setup GO
~~~
echo 'export GOPATH=${HOME}/go' >> ~/.bashrc
echo 'export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin' >> ~/.bashrc
source ~/.bashrc
~~~

4. download singularity
~~~
export VERSION=3.8.3
wget https://github.com/sylabs/singularity/releases/download/v${VERSION}/singularity-ce-${VERSION}.tar.gz
tar -xzf singularity-ce-${VERSION}.tar.gz
cd singularity-ce-{$VERSION}
~~~

5. make and install
~~~
./mconfig --prefix=/opt/singularity
make -C ./builddir
~~~

6. create link
~~~
sudo ln -s /opt/singularity/bin/singularity /usr/local/bin/singularity
~~~

7. testing
~~~
singularity version
singularity exec library://alpine cat /etc/alpine-release
~~~

#### Singularity image with ANTS and SPM12

Use the config file in src/singularityImages/confSingularity_nipype_ANTS_SPM12_PYTHON.txt.

~~~
cd path/to/atlas/singularityImages
sudo singularity build nipype_ANTS_SPM12_SingularityImg_test.simg confSingularity_nipype_ANTS_SPM12_PYTHON.txt
~~~

This command will take time to prepare the image... have a cup of coffe =).

Contents of the image
 - Base: Ubuntu 20.04
 - MATLAB Runtime R2019b Update 3 glnxa64
 - SPM12 version r7771 (method binaries)
 - Python3.6
 - Nipype


## Usage

The project has 3 main python functions

```
./anatomicalAtlas/staticComponent/src/main_staticAtlas.py
./anatomicalAtlas/dynamicComponent/src/main_dynamicAtlas.py
./forwardProblem/EITmodel.py
```

The two first files creat the static and dynamic components of the atlas. The last is used to solve Electrical 
impedance tomography forward problem with the atlas.


#### Static Component of the atlas

This component is generated by calling

```
python3 main_staticAtlas.py
```

The user can edit 3 variables inside `main_staticAtlas.py` to specify the type of atlas

```
rFactList = [ 2 ]  # list with resampling factors. use integer values
propList = [ 'resistivity' ] # electrical property list. Valid values: 'resistivity', 'conductivity' , 'relPermittivity'
freqHzList = [ 1000 ]  # list of frequencies in Hz
```

 - rFact controls the resampling factor used for the atlas. USE INTEGER VALUES
      - rFact=1: no resampling (highest resolution 1x1x1 mm voxels)
      - rFact=2: 2x downlampling. resamples, reducing by a factor of 2
      - ...
      - rFact=n: nx downlampling. resamples, reducing by a factor of n
 - prop: electrical property to be considered. Valid values: 'resistivity', 'conductivity' , 'relPermittivity'
 - freqHz: frequency in hertz.

If the user set more than one element per list, one atlas will be generated for each combination. (Imagine there are 3 nested for loops)

Input T1 and T2 MRI images files must be placed inside  `./anatomicalAtlas/staticComponent/inputData` folder. 
Intermediate files will be created inside `./anatomicalAtlas/staticComponent/outputData` and the files of the 
atlas will be craeted inside `./anatomicalAtlas/staticComponent/atlas`. These folders can be configured in `.
/anatomicalAtlas/staticComponent/src/utils.py`

The output files inside the `./anatomicalAtlas/staticComponent/atlas` folder are:

**File name prefix**
The resulting files of the atlas have the prefix `Atlas_[propType]_freq_XXX_RFact_YYY`, where
 - `[propType]` is the property set on `propList` input list
 - `XXX` is the frequency set on `freqHzList` input list
 - `YYY` is the resampling factor set on `rFactList` input list
 - 
1. **HEAD GEOMETRY:** Files related to the geometry of the entire head:

   - `[PREFIX]_Mask.nii` Binary image in NIfTI format with the mask of the entire head, that is, if the voxel lies within the head volume, then the voxels 
     receives a `True` value. These voxels are refered as **valid pixels**
   - `[PREFIX]_Mask.npy` The same `[PREFIX]_Mask.nii` but in numpy format.
   - `[PREFIX]_Mask_indices.csv` CSV file with the indices of the valid voxels. Each line is in the form `i,j,k` where the voxel is located at `mask[i,j,k]`
   - `[PREFIX]_Mask_coords.csv` CSV file with the coords of the valid voxels. Each line is in the form `x,y,z` where the i-th line is associated 
     with the voxel with indices at the same i-th line in `[PREFIX]_Mask_indices.csv`
  - `[PREFIX]_validPixels.npy` vector with the indices of the valid pixels. The elements of this vectors are computed by collapsing the 3D mask image
    into one dimension using `numpy.ndarray.flatten('C')` (‘C’ stands for  flattening in row-major (C-style) order.), followed by searching the 
    elements with `True` value.
  - `[PREFIX]_vascularTerritories.nii` Image in NIfTI format with the segments that define the vascular territories.

     Territory | region value | Territory | region value
     --- | --- | --- | --- 
     Left Anterior Cerebral Artery |  1  | Right Anterior Cerebral Artery | 2
     Left Middle Cerebral Artery |  3  | Right Middle Cerebral Artery | 4
     Left Posterior Cerebral Artery |  5  | Right Posterior Cerebral Artery | 6
     Left Superior Cerebellar Artery |  7  | Right Superior Cerebellar Artery | 8
     Stem | 9 | |
     Left External Carotid Artery | 10  | Right External Carotid Artery | 11

  - `[PREFIX]_vascularTerritories.npy` Equal to `[PREFIX]_vascularTerritories.nii`, but in numpy array and only for valid pixels.
  
2. **BRAIN GEOMETRY:** Files related to the geometry of the entire brain volume, defined as Gray matter + White Matter + CSF:

   - `[PREFIX]_BRAIN_Mask.nii` Equal to `[PREFIX]_Mask.nii`, but for GM+WM+CSF volume only.
   - `[PREFIX]_BRAIN_Mask.npy` Equal to `[PREFIX]_Mask.npy`, but for GM+WM+CSF volume only.
   - `[PREFIX]_BRAIN_Mask_indices.csv` Equal to `[PREFIX]_Mask_indices.npy`, but for GM+WM+CSF volume only.
   - `[PREFIX]_BRAIN_Mask_coords.csv` Equal to `[PREFIX]_Mask_coords.npy`, but for GM+WM+CSF volume only.
   
3. **SCALP GEOMETRY:** Files related to the geometry of the scalp volume.

   - `[PREFIX]_BRAIN_Mask.nii` Equal to `[PREFIX]_Mask.nii`, but for scalp volume only.
   - `[PREFIX]_BRAIN_Mask.npy` Equal to `[PREFIX]_Mask.npy`, but for scalp volume only.
   - `[PREFIX]_BRAIN_Mask_indices.csv` Equal to `[PREFIX]_Mask_indices.npy`, but for scalp volume only.
   - `[PREFIX]_BRAIN_Mask_coords.csv` Equal to `[PREFIX]_Mask_coords.npy`, but for scalp volume only.

2. **ATLAS STATISTICS:** Files related to the statistics of the atlas
  
   - `[PREFIX]_Avg.nii` Average image of the atlas in NIfTI format. The values are in international System of Units (SI)
   - `[PREFIX]_Avg.npy` Numpy array with the average. The values are in international System of Units (SI). Only valid pixels are stored. See 
     valid pixels below.
   - `[PREFIX]_CovK.npy` Covariance matrix factor matrix `K` in numpy .npy array fromat. The values are in international System of Units 
     (SI). Only valid pixels are stored. See valid pixels below.

      <img src="https://render.githubusercontent.com/render/math?math=\Gamma = K^T K" width="100px">
 
   
#### Dynamic Component of the atlas

This component is generated by calling

```
python3 main_dynamicAtlas.py
```

The user can edit 1 variable inside this file to specify the type of atlas

```
rFactList = [ 4 ]  # list with resampling factors. use integer values
```

 - rFact controls the resampling factor used for the atlas. USE INTEGER VALUES
      - rFact=1: no resampling (highest resolution 1x1x1 mm voxels)
      - rFact=2: 2x downlampling. resamples, reducing by a factor of 2
      - ...
      - rFact=n: nx downlampling. resamples, reducing by a factor of n

   Obs: the user does not specify frequency and property type here because the dynamic atlas is computed in a 
   different way. The type and frequency is specified in another moment.

Input Angiographic MRI images files must be placed inside  `./anatomicalAtlas/dynamicComponent/inputData` folder. 
Intermediate files will be created inside `./anatomicalAtlas/dynamicComponent/outputData` and the files of the 
atlas will be craeted inside `./anatomicalAtlas/dynamicComponent/atlas`. These folders can be configured in `.
/anatomicalAtlas/dynamicComponent/src/utils.py`

The resulting files of the atlas have the prefix `Atlas_normalized_Rfactor_YYY`, where
 - `YYY` is the resampling factor set on `rFactList` input list

The output files inside the `./anatomicalAtlas/dynamicComponent/atlas` folder are:

1. **HEAD GEOMETRY:** Files related to the geometry of the entire head:

   - `[PREFIX]_Mask.nii` Same of the static component.
   - `[PREFIX]_Mask.npy` Same of the static component.
   - `[PREFIX]_Mask_indices.csv` Same of the static component.
   - `[PREFIX]_Mask_coords_aligned.csv` Same of the static component.
  - `[PREFIX]_validPixels.npy` Same of the static component.

2. **BLOOD GEOMETRY:** Files related to the geometry of the arterial tree:

   - `[PREFIX]_BLOOD_Mask.nii` Equal to `[PREFIX]_Mask.nii`, but for arterial tree volume only.
   - `[PREFIX]_BLOOD_Mask.npy` Equal to `[PREFIX]_Mask.npy`, but for arterial tree volume only.
   - `[PREFIX]_BLOOD_Mask_indices.csv` Equal to `[PREFIX]_Mask_indices.npy`, but for arterial tree volume only.
   - `[PREFIX]_BLOOD_Mask_coords.csv` Equal to `[PREFIX]_Mask_coords.npy`, but for arterial tree volume only.


#### Electrical impedance tomography example

One example of usage is provided in the `forwardProblem` folder. It can be called by:

```
python3 EITmodel.py -i path/to/configurationFile.conf
```

where some configuration files are provided in the `./forwardProblem/inputFiles` directory. In this same directory there is a zip file with the FEM meshes that must be uncompressed.

The configuration file specifies the solver. Description of the fields are presented in the file.

 - The segment of the configuration file associated with the atlas is `EITmodel>AnatomicalAtlas`
 - Frequency of the atlas is configure in the section `EITmodel>current>frequency_Hz`
 - Segment associated with blood flow simulation is `EITmodel>AnatomicalAtlas>openBF`
 - Segment associated with Visser model is `EITmodel>AnatomicalAtlas>visserModel`
 - Segment associated with the forward problem is `EITmodel>forwardProblem`
 - Segment associated with the FEM mesh is `EITmodel>FEMmodel`


