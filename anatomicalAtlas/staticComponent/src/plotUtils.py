# -*- coding: utf-8 -*-

import copy
import os

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import skimage.measure as skimageMeas

matplotlib.use('Qt5Agg')


# https://matplotlib.org/3.1.0/tutorials/colors/colormap-manipulation.html
def plot_linearmap(cmapColor):
    rgba = cmapColor(np.linspace(0, 1, 256))
    fig, ax = matplotlib.pyplot.subplots(figsize=(4, 3), constrained_layout=True)
    col = ['r', 'g', 'b']
    for xx in [0.25, 0.5, 0.75]:
        ax.axvline(xx, color='0.7', linestyle='--')
    for i in range(3):
        ax.plot(np.arange(256) / 256, rgba[:, i], color=col[i])
    ax.set_xlabel('index')
    ax.set_ylabel('RGB')
    matplotlib.pyplot.show()


def showMask3D(imgArray1, imgArray2=None, showDiff=False):
    verts1, faces1, normals, values = skimageMeas.marching_cubes(volume=imgArray1, level=None, spacing=(1.0, 1.0, 1.0), gradient_direction='descent',
                                                                 step_size=3, allow_degenerate=True, method='lewiner', mask=None)
    nplots = 1
    if imgArray2 is not None:
        nplots = 2
    if showDiff and imgArray2 is not None:
        nplots = 3

    fig = plt.figure()
    ax1 = fig.add_subplot(1, nplots, 1, projection='3d')
    ax1.plot_trisurf(verts1[:, 0], verts1[:, 1], faces1, verts1[:, 2], cmap='Spectral', lw=1)
    plt.title('array1')

    if imgArray2 is not None:
        verts2, faces2, _, _ = skimageMeas.marching_cubes(volume=imgArray2, level=None, spacing=(1.0, 1.0, 1.0), gradient_direction='descent',
                                                          step_size=3, allow_degenerate=True, method='lewiner', mask=None)
        ax2 = fig.add_subplot(1, nplots, 2, projection='3d')
        ax2.plot_trisurf(verts2[:, 0], verts2[:, 1], faces2, verts2[:, 2], cmap='Spectral', lw=1)
        plt.title('array2')

    if showDiff and imgArray2 is not None:
        verts3, faces3, _, _ = skimageMeas.marching_cubes(volume=np.logical_xor(imgArray1, imgArray2), level=None, spacing=(1.0, 1.0, 1.0),
                                                          gradient_direction='descent', step_size=3, allow_degenerate=True, method='lewiner',
                                                          mask=None)
        ax3 = fig.add_subplot(1, nplots, 3, projection='3d')
        ax3.plot_trisurf(verts3[:, 0], verts3[:, 1], faces3, verts3[:, 2], cmap='Spectral', lw=1)
        plt.title('difference')

    plt.show()


def comp_2Dplot(array1, array2):
    """
    plot two 3D arrays representing images with interactive controls

    Inputs:
    - array1, array2: arrays to be plotted
    """
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    fig.subplots_adjust(bottom=0.25)

    axcolor = 'lightgoldenrodyellow'

    map_color = plt.cm.Greys_r
    map_color.set_bad(color='red')

    ax.imshow(array1, cmap=map_color, aspect='auto')
    ax2.imshow(array2, cmap=map_color, aspect='auto')

    title_obj = plt.title('comparison')
    plt.setp(title_obj, position=(0.47, 37.0))
    plt.setp(title_obj, size=14)


def saveImgSlice(array, fileName, colormap=plt.cm.Greys_r, plane='coronal', slice=0, showColorBar=True, cbarFontSize=15, cbarLabel='label',
                 pixelAspectRatio=1.0, logScale=False):
    """ save one slice of a 3d array representing an image"""
    dpiImage = 300

    colormap = copy.copy(colormap)
    colormap.set_bad(color=(0.9, 0.9, 0.9))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if plane.lower() == 'coronal':
        im = ax.imshow(np.rot90(array[:, slice, :], k=1), cmap=colormap,
                       aspect=pixelAspectRatio)  # for logscale,  # add norm=matplotlib.colors.LogNorm()
    if plane.lower() == 'sagital':
        im = ax.imshow(np.rot90(array[slice, :, :], k=1), cmap=colormap,
                       aspect=pixelAspectRatio)  # for logscale,  # add norm=matplotlib.colors.LogNorm()
    if plane.lower() == 'transversal':
        im = ax.imshow(np.rot90(array[:, :, slice], k=0), cmap=colormap,
                       aspect=pixelAspectRatio)  # for logscale,  # add norm=matplotlib.colors.LogNorm()

    cmin = np.nanpercentile(array, 0)
    cmax = np.nanpercentile(array, 100)

    im.set_clim(cmin, cmax)

    # title_obj = plt.title('Plano Transversal')
    # plt.setp(title_obj, position=(0.47, 37.0))
    # plt.setp(title_obj, size=14)
    ax.axis('off')

    # plot colorbar in the image.
    if showColorBar and False:
        width = 0.1
        height = 0.9  # 0.0 to 1.0
        offsetY = (1.0 - height) / 2
        cax = ax.inset_axes([1.04, offsetY, width, height], transform=ax.transAxes)

        Segmented = False
        if Segmented:
            cbar = fig.colorbar(im, ticks=[0, 1, 2, 3, 4, 5])
            cbar.ax.set_yticklabels(['a', 'b', 'c', 'd', 'e', 'f'])  # vertically oriented colorbar
        else:
            cbar = fig.colorbar(im, ax=[ax], cax=cax)

        cbar.ax.tick_params(labelsize=cbarFontSize)
        cbar.set_label(cbarLabel, size=cbarFontSize)

    plt.savefig(fileName, dpi=dpiImage, facecolor='w', edgecolor='w', orientation='portrait', format=None, transparent=False, bbox_inches='tight',
                pad_inches=0.05, metadata=None)
    # plt.show()

    plt.close()

    if showColorBar:
        createColorbar(os.path.splitext(fileName)[0] + '_colorbar_hor.png', orientation='horizontal', dpiImage=dpiImage, colormap=colormap,
                       cbarFontSize=cbarFontSize, vmin=cmin, vmax=cmax, cbarLabel=cbarLabel, logScale=logScale)
        createColorbar(os.path.splitext(fileName)[0] + '_colorbar_ver.png', orientation='vertical', dpiImage=dpiImage, colormap=colormap,
                       cbarFontSize=cbarFontSize, vmin=cmin, vmax=cmax, cbarLabel=cbarLabel, logScale=logScale)


def createColorbar(fileName, orientation='horizontal', dpiImage=300, colormap=plt.cm.Greys_r, cbarFontSize=15, vmin=0, vmax=100, cbarLabel=None,
                   logScale=False):
    fig = plt.figure()

    thickness = 0.05  # 0.0 to 1.0
    length = 1.2  # 0.0 to 1.0

    if orientation.lower() == 'horizontal':
        ax = fig.add_axes([0.0, 0.0, length, thickness])
    else:
        ax = fig.add_axes([0.0, 0.0, thickness, length])

    if logScale:
        cb = matplotlib.colorbar.ColorbarBase(ax, orientation=orientation, cmap=colormap, norm=matplotlib.colors.LogNorm(vmin, vmax),  # vmax and vmin
                                              label='This is a label')
    else:
        cb = matplotlib.colorbar.ColorbarBase(ax, orientation=orientation, cmap=colormap, norm=matplotlib.colors.Normalize(vmin, vmax),
                                              # vmax and vmin
                                              label='This is a label')

    cb.ax.tick_params(labelsize=cbarFontSize)
    if cbarLabel is not None:
        cb.set_label(cbarLabel, size=cbarFontSize)

    plt.savefig(fileName, dpi=dpiImage, facecolor='w', edgecolor='w', format=None, transparent=False, bbox_inches='tight', pad_inches=0.05,
                metadata=None)

    plt.close()


def saveNIIsliceSet(inputFileNii, colormap=plt.cm.Greys_r, corSlice=None, traSlice=None, sagSlice=None, showColorBar=True, cbarFontSize=10,
                    cbarLabel='label', pixelAspectRatio=1.0):
    """ saves one slice of each plane from a .nii image. Each image is saved in a different file """
    img = nib.load(inputFileNii)
    array = img.get_fdata()
    saveArraysliceSet(array, inputFileNii.replace('.nii', ''), colormap, corSlice, traSlice, sagSlice, showColorBar, cbarFontSize, cbarLabel,
                      pixelAspectRatio)


def saveArraysliceSet(array, fileNamePrefix, colormap=plt.cm.Greys_r, corSlice=None, traSlice=None, sagSlice=None, showColorBar=True, cbarFontSize=10,
                      cbarLabel='label', pixelAspectRatio=1.0):
    """ saves one slice of each plane from a 3D array representing an image. Each image is saved in a different file """
    # single_3Dplot(array)
    if sagSlice is not None:
        saveImgSlice(array, fileNamePrefix + '_sagital_slice_%03d.png' % sagSlice, colormap, 'sagital', sagSlice, showColorBar, cbarFontSize,
                     cbarLabel, pixelAspectRatio)
    if traSlice is not None:
        saveImgSlice(array, fileNamePrefix + '_transversal_%03d.png' % traSlice, colormap, 'transversal', traSlice, showColorBar, cbarFontSize,
                     cbarLabel, pixelAspectRatio)
    if corSlice is not None:
        saveImgSlice(array, fileNamePrefix + '_coronal_%03d.png' % corSlice, colormap, 'coronal', corSlice, showColorBar, cbarFontSize, cbarLabel,
                     pixelAspectRatio)


def saveNIIsliceSetMontage(inputFileNii, colormap=plt.cm.Greys_r, corSlice=0, traSlice=0, sagSlice=0, showColorBar=True, cbarFontSize=10,
                           cbarLabel='label', valueLimits=None, pixelAspectRatio=[1.0, 1.0, 1.0], logScale=False):
    """ creates a montage figure with 3 images (each plane, specified by the number of the slide)"""

    # valueLimits: list [vmin, vmax] for color scale. Use None to compute limits from the image
    img = nib.load(inputFileNii)
    array = img.get_fdata()
    saveArraysliceSetMontage(array, inputFileNii.replace('.nii', ''), colormap, corSlice, traSlice, sagSlice, showColorBar, cbarFontSize, cbarLabel,
                             valueLimits, pixelAspectRatio, logScale)


def saveArraysliceSetMontage(array, fileNamePrefix, colormap=plt.cm.Greys_r, corSlice=0, traSlice=0, sagSlice=0, showColorBar=True, cbarFontSize=15,
                             cbarLabel='label', valueLimits=None, pixelAspectRatio=[1.0, 1.0, 1.0], logScale=False,orientation='horiz'):
    """ creates a montage figure with 3 images (each plane, specified by the number of the slide)"""
    # valueLimits: list [vmin, vmax] for color scale. Use None to compute limites from the image
    dpiImage = 300
    colormap = copy.copy(colormap)
    colormap.set_bad(color=(0.9, 0.9, 0.9))

    coronalImage = np.rot90(array[:, corSlice, :], k=1)
    sagitalImage = np.rot90(array[sagSlice, :, :], k=1)
    transveImage = np.rot90(array[:, :, traSlice], k=0)

    maxHor = max(sagitalImage.shape[1],transveImage.shape[1],coronalImage.shape[1])
    maxVer = max(sagitalImage.shape[0],transveImage.shape[0],coronalImage.shape[0])

    # create plots
    if orientation.lower() == 'horiz':
        fig, ax = plt.subplots(1, 3, dpi=dpiImage,
                               gridspec_kw={'width_ratios': [sagitalImage.shape[1], transveImage.shape[1], coronalImage.shape[1]]})
    else:
        fig, ax = plt.subplots(3, 1, dpi=dpiImage, gridspec_kw={'height_ratios': [sagitalImage.shape[0], transveImage.shape[0], coronalImage.shape[0]]})

    if orientation.lower() == 'horiz':
        # add black borders to the image to make them all the same sizes
        if maxVer > sagitalImage.shape[0]:
            outVal = sagitalImage[0,0]
            delta=maxVer-sagitalImage.shape[0]
            patch1=np.zeros((delta, sagitalImage.shape[1]), dtype=int)*outVal
            sagitalImage = np.concatenate((patch1, sagitalImage), axis=0)

        if maxVer > transveImage.shape[0]:
            outVal = transveImage[0, 0]
            delta = maxVer - transveImage.shape[0]
            patch1=np.zeros((int(delta/2), transveImage.shape[1]), dtype=int)*outVal
            patch2=np.zeros((delta-int(delta/2), transveImage.shape[1]), dtype=int)*outVal
            transveImage = np.concatenate((patch1, transveImage, patch2), axis=0)

        if maxVer > coronalImage.shape[0]:
            outVal = coronalImage[0,0]
            delta=maxVer-coronalImage.shape[0]
            patch1=np.zeros((delta, coronalImage.shape[1]), dtype=int)*outVal
            coronalImage = np.concatenate((patch1, coronalImage), axis=0)

    else:
        # add black borders to the image to make them all the same sizes
        if maxHor > sagitalImage.shape[1]:
            outVal = sagitalImage[0,0]
            delta=maxHor-sagitalImage.shape[1]
            patch1=np.zeros((sagitalImage.shape[0],int(delta/2) ), dtype=int)*outVal
            patch2=np.zeros((sagitalImage.shape[0],delta-int(delta/2)), dtype=int)*outVal
            sagitalImage = np.concatenate((patch1, sagitalImage,patch2), axis=1)

        if maxHor > transveImage.shape[1]:
            outVal = transveImage[0,0]
            delta=maxHor-transveImage.shape[1]
            patch1=np.zeros((transveImage.shape[0],int(delta/2) ), dtype=int)*outVal
            patch2=np.zeros((transveImage.shape[0],delta-int(delta/2)), dtype=int)*outVal
            transveImage = np.concatenate((patch1, transveImage,patch2), axis=1)

        if maxHor > coronalImage.shape[1]:
            outVal = coronalImage[0,0]
            delta=maxHor-coronalImage.shape[1]
            patch1=np.zeros((coronalImage.shape[0],int(delta/2) ), dtype=int)*outVal
            patch2=np.zeros((coronalImage.shape[0],delta-int(delta/2)), dtype=int)*outVal
            coronalImage = np.concatenate((patch1, coronalImage,patch2), axis=1)

    if logScale:
        im0 = ax[0].imshow(sagitalImage, cmap=colormap, interpolation='nearest', aspect=pixelAspectRatio[0], norm=matplotlib.colors.LogNorm())
        im1 = ax[1].imshow(transveImage, cmap=colormap, interpolation='nearest', aspect=pixelAspectRatio[1], norm=matplotlib.colors.LogNorm())
        im2 = ax[2].imshow(coronalImage, cmap=colormap, interpolation='nearest', aspect=pixelAspectRatio[2], norm=matplotlib.colors.LogNorm())
    else:
        im0 = ax[0].imshow(sagitalImage, cmap=colormap, interpolation='nearest', aspect=pixelAspectRatio[0])
        im1 = ax[1].imshow(transveImage, cmap=colormap, interpolation='nearest', aspect=pixelAspectRatio[1])
        im2 = ax[2].imshow(coronalImage, cmap=colormap, interpolation='nearest', aspect=pixelAspectRatio[2])

    if valueLimits is None:
        cmin = np.nanpercentile(array, 0)
        cmax = np.nanpercentile(array, 100)
        print('cmin, cmax: %f, %f' % (cmin, cmax))
    else:
        cmin = valueLimits[0]
        cmax = valueLimits[1]

    im0.set_clim(cmin, cmax)
    im1.set_clim(cmin, cmax)
    im2.set_clim(cmin, cmax)

    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')

    # plot colorbar in the image.
    if showColorBar and False:
        width = 0.1
        height = 0.9  # 0.0 to 1.0
        offsetY = (1.0 - height) / 2
        cax = ax[2].inset_axes([1.04, offsetY, width, height], transform=ax[2].transAxes)

        Segmented = False
        if Segmented:
            cbar = fig.colorbar(im2, ticks=[0, 1, 2, 3, 4, 5])
            cbar.ax.set_yticklabels(['a', 'b', 'c', 'd', 'e', 'f'])  # vertically oriented colorbar
        else:
            cbar = fig.colorbar(im2, ax=[ax[2]], cax=cax)

        cbar.ax.tick_params(labelsize=cbarFontSize)
        cbar.set_label(cbarLabel, size=cbarFontSize)

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    fileName = fileNamePrefix + '_montage_slices_c%03d_s%03d_t%03d.png' % (corSlice, traSlice, sagSlice)
    plt.savefig(fileName, dpi=dpiImage, facecolor='w', edgecolor='w', orientation='portrait', format=None, transparent=False, bbox_inches='tight',
                pad_inches=0.05, metadata=None)
    # plt.show()
    plt.close()

    if showColorBar:
        createColorbar(os.path.splitext(fileName)[0] + '_colorbar_hor.png', orientation='horizontal', dpiImage=dpiImage, colormap=colormap,
                       cbarFontSize=cbarFontSize, vmin=cmin, vmax=cmax, cbarLabel=cbarLabel, logScale=logScale)
        createColorbar(os.path.splitext(fileName)[0] + '_colorbar_ver.png', orientation='vertical', dpiImage=dpiImage, colormap=colormap,
                       cbarFontSize=cbarFontSize, vmin=cmin, vmax=cmax, cbarLabel=cbarLabel, logScale=logScale)


def single_3Dplot_NII(NIIfile):
    """plot Nii file with interactive controls"""
    array = nib.load(NIIfile).get_fdata()
    single_3Dplot(array, scale='linear')


def single_3Dplot(array, scale='linear', colormap='jet', colorLimits='slice'):
    """
    plot 3D array with interactive controls

    Inputs:
    scale: 'linear', 'log'
    colormap: 'jet', 'gray'
    colorLimits: 'slice', 'full'
    """
    from matplotlib.widgets import Slider, Button
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(bottom=0.25)

    global atual_ax, atual_val

    atual_ax = 0
    atual_val = 0

    axcolor = 'lightgoldenrodyellow'

    if colormap.lower() == 'jet':
        map_color = plt.cm.jet
    if colormap.lower() == 'gray':
        map_color = plt.cm.Greys_r

    map_color.set_bad(color=(1.0, 0.8, 0.8))

    axprox = plt.axes([0.53, 0.02, 0.05, 0.05])
    button_prox = Button(axprox, '>', color=axcolor, hovercolor='0.575')

    axanter = plt.axes([0.43, 0.02, 0.05, 0.05])
    button_anter = Button(axanter, '<', color=axcolor, hovercolor='0.575')

    axprox10 = plt.axes([0.6, 0.02, 0.05, 0.05])
    button_prox10 = Button(axprox10, '>>', color=axcolor, hovercolor='0.575')

    axanter10 = plt.axes([0.36, 0.02, 0.05, 0.05])
    button_anter10 = Button(axanter10, '<<', color=axcolor, hovercolor='0.575')

    ax_i = fig.add_axes([0.15, 0.1, 0.75, 0.02], facecolor=axcolor)
    ax_j = fig.add_axes([0.15, 0.14, 0.75, 0.02], facecolor=axcolor)
    ax_k = fig.add_axes([0.15, 0.18, 0.75, 0.02], facecolor=axcolor)

    s_i = Slider(ax_i, 'Sagital', 0, np.size(array, 0) - 1, valinit=0, valfmt="%i")
    s_j = Slider(ax_j, 'Coronal', 0, np.size(array, 1) - 1, valinit=0, valfmt="%i")
    s_k = Slider(ax_k, 'Transversal', 0, np.size(array, 2) - 1, valinit=0, valfmt="%i")
    sliderList = [s_i, s_j, s_k]

    if scale.lower() == 'linear':
        im = ax.imshow(np.rot90(array[:, :, 0], k=0), cmap=map_color)
    else:
        im = ax.imshow(np.rot90(array[:, :, 0], k=0), cmap=map_color, norm=matplotlib.colors.LogNorm())

    if colorLimits.lower() == 'full':
        # print('%f  %f' % (np.nanmin(array),np.nanmax(array)))
        im.set_clim(np.nanmin(array), np.nanmax(array))

    title_obj = plt.title('Plano Transversal')
    plt.setp(title_obj, position=(0.47, 37.0))
    plt.setp(title_obj, size=14)

    def updateImgs(planeIdx, axis):
        global atual_val, atual_ax

        atual_val = planeIdx
        atual_ax = axis

        for i, slider in enumerate(sliderList):
            if i != axis:
                slider.reset()

        if axis == 0:
            im = ax.imshow(np.rot90(array[int(planeIdx), :, :], k=0), cmap=map_color)
            plt.title('Plano Sagital')
        if axis == 1:
            im = ax.imshow(np.rot90(array[:, int(planeIdx), :], k=0), cmap=map_color)
            plt.title('Plano Coronal')
        if axis == 2:
            im = ax.imshow(np.rot90(array[:, :, int(planeIdx)], k=0), cmap=map_color)
            plt.title('Plano Transversal')

        if colorLimits.lower() == 'full':
            # print('%f  %f' % (np.nanmin(array),np.nanmax(array)))
            im.set_clim(np.nanmin(array), np.nanmax(array))

        plt.setp(title_obj, size=14)
        plt.show()

    s_i.on_changed(lambda x: updateImgs(x, axis=0))
    s_j.on_changed(lambda x: updateImgs(x, axis=1))
    s_k.on_changed(lambda x: updateImgs(x, axis=2))

    def changeImgStep(event, step):
        global atual_val
        atual_val = (atual_val + step) % np.size(array, atual_ax)
        sliderList[atual_ax].set_val(atual_val)

    button_prox.on_clicked(lambda x: changeImgStep(x, step=1))
    button_anter.on_clicked(lambda x: changeImgStep(x, step=-1))
    button_prox10.on_clicked(lambda x: changeImgStep(x, step=10))
    button_anter10.on_clicked(lambda x: changeImgStep(x, step=-10))

    plt.show()


def comp_3Dplot_NII(NIIfile_1, NIIfile_2):
    "plot 2 nii files for comparison with interactive controls"
    array1 = nib.load(NIIfile_1).get_fdata()
    array2 = nib.load(NIIfile_2).get_fdata()
    comp_3Dplot(array1, array2, scale='linear')


def comp_3Dplot(array1, array2, scale='linear', colormap='jet', colorLimits='full'):
    """
    plot 2 3d arrays  comparison with interactive controls

    Inputs:
    - array1, array2: arrays to be plotted
    scale: 'linear', 'log'
    colormap: 'jet', 'gray'
    colorLimits: 'slice', 'full'
    """
    from matplotlib.widgets import Slider, MultiCursor, Button
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    fig.subplots_adjust(bottom=0.25)

    global atual_ax, atual_val

    atual_ax = 0
    atual_val = 0

    axcolor = 'lightgoldenrodyellow'

    if colormap.lower() == 'jet':
        map_color = plt.cm.jet
    if colormap.lower() == 'gray':
        map_color = plt.cm.Greys_r

    map_color.set_bad(color=(1.0, 0.8, 0.8))

    axprox = plt.axes([0.53, 0.02, 0.05, 0.05])
    button_prox = Button(axprox, '>', color=axcolor, hovercolor='0.575')

    axanter = plt.axes([0.43, 0.02, 0.05, 0.05])
    button_anter = Button(axanter, '<', color=axcolor, hovercolor='0.575')

    axprox10 = plt.axes([0.6, 0.02, 0.05, 0.05])
    button_prox10 = Button(axprox10, '>>', color=axcolor, hovercolor='0.575')

    axanter10 = plt.axes([0.36, 0.02, 0.05, 0.05])
    button_anter10 = Button(axanter10, '<<', color=axcolor, hovercolor='0.575')

    ax_i = fig.add_axes([0.15, 0.1, 0.75, 0.02], facecolor=axcolor)
    ax_j = fig.add_axes([0.15, 0.14, 0.75, 0.02], facecolor=axcolor)
    ax_k = fig.add_axes([0.15, 0.18, 0.75, 0.02], facecolor=axcolor)

    s_i = Slider(ax_i, 'Sagital', 0, np.size(array1, 0) - 1, valinit=0, valfmt="%i")
    s_j = Slider(ax_j, 'Coronal', 0, np.size(array1, 1) - 1, valinit=0, valfmt="%i")
    s_k = Slider(ax_k, 'Transversal', 0, np.size(array1, 2) - 1, valinit=0, valfmt="%i")
    sliderList = [s_i, s_j, s_k]

    if scale == 'linear':
        p1 = ax1.imshow(np.rot90(array1[:, :, 0], k=0), cmap=map_color)
        p2 = ax2.imshow(np.rot90(array2[:, :, 0], k=0), cmap=map_color)
    else:
        p1 = ax1.imshow(np.rot90(array1[:, :, 0], k=0), cmap=map_color, norm=matplotlib.colors.LogNorm())
        p2 = ax2.imshow(np.rot90(array2[:, :, 0], k=0), cmap=map_color, norm=matplotlib.colors.LogNorm())
    MultiCursor(fig.canvas, (ax1, ax2), color='r', lw=1, horizOn=True, vertOn=True)

    if colorLimits.lower() == 'full':
        p1.set_clim(np.min(array1), np.max(array1))
        p2.set_clim(np.min(array2), np.max(array2))

    title_obj = plt.title('Plano Transversal')
    plt.setp(title_obj, position=(0.47, 37.0))
    plt.setp(title_obj, size=14)

    def updateImgs(planeIdx, axis):
        global atual_val, atual_ax

        atual_val = planeIdx
        atual_ax = axis

        for i, slider in enumerate(sliderList):
            if i != axis:
                slider.reset()

        if axis == 0:
            ax1.imshow(np.rot90(array1[int(planeIdx), :, :], k=0), cmap=map_color)
            ax2.imshow(np.rot90(array2[int(planeIdx), :, :], k=0), cmap=map_color)
            plt.title('Plano Sagital')
        if axis == 1:
            ax1.imshow(np.rot90(array1[:, int(planeIdx), :], k=0), cmap=map_color)
            ax2.imshow(np.rot90(array2[:, int(planeIdx), :], k=0), cmap=map_color)
            plt.title('Plano Coronal')
        if axis == 2:
            ax1.imshow(np.rot90(array1[:, :, int(planeIdx)], k=0), cmap=map_color)
            ax2.imshow(np.rot90(array2[:, :, int(planeIdx)], k=0), cmap=map_color)
            plt.title('Plano Transversal')

        MultiCursor(fig.canvas, (ax1, ax2), color='r', lw=1, horizOn=True, vertOn=True)
        plt.setp(title_obj, size=14)
        plt.show()

    s_i.on_changed(lambda x: updateImgs(x, axis=0))
    s_j.on_changed(lambda x: updateImgs(x, axis=1))
    s_k.on_changed(lambda x: updateImgs(x, axis=2))

    def changeImgStep(event, step):
        global atual_val
        atual_val = (atual_val + step) % np.size(array1, atual_ax)
        sliderList[atual_ax].set_val(atual_val)

    button_prox.on_clicked(lambda x: changeImgStep(x, step=1))
    button_anter.on_clicked(lambda x: changeImgStep(x, step=-1))
    button_prox10.on_clicked(lambda x: changeImgStep(x, step=10))
    button_anter10.on_clicked(lambda x: changeImgStep(x, step=-10))

    plt.show()
