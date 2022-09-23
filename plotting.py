from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

def plot_process(im_fft, im_fft_filtered, im_gray, im_filtered, im_wrapped, im_unwrapped, roi):
    fig, axs = plt.subplots(3, 2, sharex=True, sharey=True)

    axs[0, 0].imshow(np.abs(im_fft).astype(np.uint8))
    axs[0, 0].set_title('Image FFT')
    axs[0, 1].imshow(np.abs(im_fft_filtered).astype(np.uint8))
    axs[0, 1].set_title(f'Image FFT filtered (roi {roi})')

    axs[1, 0].imshow(im_gray, cmap='gray')
    axs[1, 0].set_title('Original image')
    axs[1, 1].imshow(np.abs(im_filtered).astype(np.uint8), cmap='gray')
    axs[1, 1].set_title('Filtered image')

    axs[2, 0].imshow(im_wrapped, cmap='gray')
    axs[2, 0].set_title('Wrapped image')
    axs[2, 1].imshow(im_unwrapped, cmap='viridis')
    axs[2, 1].set_title('Unwrapped phase')
    return fig

def rebin(ndarray, new_shape, operation='average'):
    '''
    Resizes a 2d array by averaging or repeating elements,
    new dimensions must be integral factors of original dimensions
    '''
    if not operation.lower() in ['sum', 'mean', 'average', 'avg']:
        raise ValueError("Operation {} not supported.".format(operation))
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c // d) for d, c in zip(new_shape,
                                                     ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        if operation.lower() == "sum":
            ndarray = ndarray.sum(-1 * (i + 1))
        elif operation.lower() in ["mean", "average", "avg"]:
            ndarray = ndarray.mean(-1 * (i + 1))
    return ndarray

def plot_surface(im_unwrapped, config, conversionFactorXY, unitXY, unitZ, overlay_image=np.empty(0)):
    X = np.arange(0, im_unwrapped.shape[1]) * conversionFactorXY
    Y = np.arange(0, im_unwrapped.shape[0]) * conversionFactorXY
    X, Y = np.meshgrid(X, Y)
    Z = np.flipud(im_unwrapped)

    simplify = None
    if simplify:
        new_shape = (X.shape[0] // simplify, X.shape[1] // simplify)
        X = rebin(X, new_shape)
        Y = rebin(Y, new_shape)
        Z = rebin(Z, new_shape)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')

    if overlay_image.any():
        cax = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False, rstride=10, cstride=10,
                              facecolors=overlay_image, rasterized=True)
    else:
        cax = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=False)

    if config.getboolean("PLOTTING", "PLOT_SURFACE_EXTRASMOOTH"):
        cax.set_rcount = 200
        cax.set_ccount = 200
    # ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(im_unwrapped) * config.getint("PLOTTING", "PLOT_SURFACE_SCALEZ")))
    fig.colorbar(cax, pad=0.1, label=f'[{unitZ}]', shrink=0.5)
    ax.set_xlabel(f'[{unitXY}]')
    ax.set_ylabel(f'[{unitXY}]')
    ax.set_ylabel(f'[{unitXY}]')
    fig.tight_layout()

    return fig

def plot_imunwrapped(im_unwrapped, config, conversionFactorXY, unitXY, unitZ):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    X = np.arange(0, im_unwrapped.shape[1]) * conversionFactorXY
    Y = np.arange(0, im_unwrapped.shape[0]) * conversionFactorXY
    X, Y = np.meshgrid(X, Y)
    Z = np.flipud(im_unwrapped)
    levels = np.linspace(Z.min(), Z.max(), config.getint("PLOTTING", "PLOT_SURFACE_FLAT_LEVELS"))
    cax = ax.contourf(X, Y, Z, levels=levels)

    # cax = ax.imshow(im_unwrapped, cmap=cm.viridis, extent=[0, im_unwrapped.shape[1], 0, im_unwrapped.shape[0]] * config.getint("GENERAL", "CONVERSION_FACTOR"))
    ax.set_xlabel(f'[{unitXY}]')
    ax.set_ylabel(f'[{unitXY}]')
    fig.colorbar(cax, pad=0.1, label=f'[{unitZ}]', shrink=0.5)
    fig.tight_layout()
    return fig

def plot_imwrapped(im_wrapped, config, conversionFactorXY, unitXY):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    cax = ax.imshow(im_wrapped, cmap='gray', extent=np.array([0, im_wrapped.shape[1], 0, im_wrapped.shape[0]]) * conversionFactorXY)
    ax.set_xlabel(f'[{unitXY}]')
    ax.set_ylabel(f'[{unitXY}]')
    title = f"{config['FOURIER_ADVANCED']['ROI_EDGE']=},{config['FOURIER_ADVANCED']['BLUR']=}"
    if config.getboolean("FOURIER_ADVANCED", "SECOND_FILTER"):
        title = title + (f"\n {config['FOURIER_ADVANCED']['ROI_EDGE_2']=},{config['FOURIER_ADVANCED']['BLUR_2']=}")
    ax.set_title(title)
    fig.colorbar(cax, pad=0.1, label='Units of pi', shrink=0.5)
    fig.tight_layout()
    return fig