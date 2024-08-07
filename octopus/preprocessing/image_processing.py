import numpy as np
from scipy.ndimage import minimum_filter, gaussian_filter, median_filter, convolve
from skimage import restoration
from skimage.measure import shannon_entropy
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse
from skimage.util import random_noise


def kernel_builder(size, b2d=False, normalize=False, vertical_edges=False, unit=False):
    '''
    This builds a kernel to detect edges edges in an image. This is effectively an extension to the Sobel operator.

    INPUTS:
    --------------
        size (tuple) : size of kernel

        b2d (bool, default False) : If flagged, kernel will output kernel suitable for detecting bright-to-dark intensity transitions.

        normalize (bool, default False) : If flagged, kernel will be normalised to (0,1).

        vertical_edges (bool, default False) : If flagged, kernel will detect vertical edges rather than horizontal ones by default.

        unit (bool, default False) : If flagged, non-zero values will only contain 1.
    '''
    # Set up kernel dimensions and kernel array
    kernel = np.zeros(size)
    N, M = size
    mid_r = N // 2
    mid_c = M // 2

    # Loop through half the rows to fill with incremental values. If unit=True, then kernel values are all 1 or -1
    if unit:
        for i in range(mid_r):
            kernel[i, :] = 1

    else:
        for i in range(mid_r):
            for j in range(M):
                # weight = 1/np.sqrt((i-mid_r)**2+(j-mid_c)**2)
                # weight = max(0, mid_r - np.sqrt((i-mid_r)**2+(j-mid_c)**2) + 1)
                weight = max(0, mid_r + 1 - abs(i - mid_r) - abs(j - mid_c))
                kernel[i, j] = 1 + weight

    # Array values are reflected vertically to fill remaining rows of kernel
    kernel[mid_r + 1:, :] = -np.flip(kernel[0:mid_r, :], axis=0)

    # If bright-to-dark kernel, flip kernel
    if b2d:
        kernel = np.flipud(kernel)

    # To detect vertical edges, take the transpose of kernel
    if vertical_edges:
        kernel = kernel.T

    # To normalize values
    if normalize:
        kernel = kernel / kernel.max()

    return kernel


def normalise(img, minmax_val=(0, 1), astyp=np.float32):
    """
    Normalise image between [0, 1]

    INPUTS:
    ----------------
        img (2darray) : Image

        minmax_val (2-tuple) : Tuple storing minimum and maximum values

        astyp (object) : What data type to store normalised aimage
    """
    # Extract minimum and maximum values
    min_val, max_val = minmax_val

    # Convert to float type
    img = img.astype(np.float32)

    # Normalise
    img -= img.min()
    img /= img.max()

    # Rescale to max_val
    img *= (max_val - min_val)
    img += min_val

    return img.astype(astyp)


def compute_grad_image(img, kernel, norm=True, astyp=np.float32):
    """
    Computes the gradient image using user-defined kernels to measure the horizontal and vertical gradients in both the
    bright-to-dark and dark-to-bright transitions.

    INPUTS:
    -----------------
        img (2darray) : Input image.

        kernel (2darray) : Discrete derivative filter to convolve image with.

        norm (bool, default True) : If flagged, image gradient is normalised to (0,1).

        astyp (data-type, default np.float32) : Which type to set image data as
    """

    # Compute gradient image
    grad_img = convolve(img, kernel, mode='nearest')
    grad_img[np.where(grad_img < 0)] = 0
    if normalise:
        output = normalise(grad_img, minmax_val=(0, 1), astyp=astyp)
    else:
        output = grad_img.astype(int)

    return output


def denoise(image, technique, kwargs, plot=False, verbose=False):
    """This function denoises an image using various algorithms specified by technique.

    INPUT:
    ---------
        image (2darray): Grayscale image to be denoised

        technique (str) : Type of denoising algorithm to fit.

        kwargs (dict): Dictionary of parameters for algorithm
    """
    if technique == 'nl':
        denoised_img = restoration.denoise_nl_means(image, **kwargs)
    elif technique == 'tvc':
        denoised_img = restoration.denoise_tv_chambolle(image, **kwargs)
    elif technique == 'wavelet':
        denoised_img = restoration.denoise_wavelet(image, **kwargs)
    elif technique == 'tvb':
        denoised_img = restoration.denoise_tv_bregman(image, **kwargs)
    elif technique == 'median':
        denoised_img = median_filter(image, **kwargs)
    elif technique == 'gaussian':
        denoised_img = gaussian_filter(image, **kwargs)
    elif technique == 'minimum':
        denoised_img = minimum_filter(image, **kwargs)
    else:
        print("Denoising technique not implemented.")
        denoised_img = None

    if verbose:
        psnr = round(peak_signal_noise_ratio(image, denoised_img), 2)
        ss = round(structural_similarity(image, denoised_img), 2)
        nmse = round(normalized_root_mse(image, denoised_img, normalization='min-max'), 5)
        entropy = round(shannon_entropy(denoised_img), 3)
        print(
            f'Peak-SNR: {psnr}.\nStructural Similarity: {ss}.\nMean Square Error: {nmse}.\nShannon Entropy: {entropy}.\n')

    return denoised_img


def construct_test_image(size, amplitude, curvature, noise_level, ltype, intensity, gaps=False):
    """
    This constructs a test image with a well-defined edge(s) with added noise on top.

    INPUTS:
    ----------
        size (tuple) : Size of test image.

        amplitude (float) : Amplitude of edge.

        curvature (float) : Frequency of edge.

        noise_level (float) : Noise variance to add on top.

        intensity (float) : Intensity of lighter region of image.

        ltype (str) : Type of edges. Can be straight or sinusoidal and can be basic or complex.

        gaps (bool, default False) : If flagged, add gaps to occlude parts of edge of interest.
    """
    # Extract test image size, define test_img and set up evenly spaced points to evaluate curve at
    global ywave1_idx
    M, N = size
    test_img = np.zeros((M, N))
    x = np.linspace(-np.pi, np.pi, N)
    if amplitude > M:
        A = M // 2
    else:
        A = amplitude // 2

    # Initialise indexes of edge
    xwave_idx = np.arange(0, N, 1)
    ywave_idx = []

    # For each column, evaluate curve, append row index with evaluation and set pixel intensity
    if ltype == 'sinusoidal':
        for j in range(N):
            wave = np.rint(A * np.sin(N * curvature * x[j])) + M // 2
            ywave_idx.append(wave.astype('int'))
            test_img[ywave_idx[j]:M, j] = intensity

    if ltype == 'multi-sinusoidal':
        ywave1_idx = []
        for j in range(N):
            wave = np.rint(A * np.sin(N * curvature * x[j])) + M // 2
            ywave_idx.append(wave.astype('int'))
            ywave1_idx.append(ywave_idx[j] + A // 2)
            test_img[ywave_idx[j]:M, j] = intensity
            test_img[ywave_idx[j] + A // 2:M, j] = 1 - intensity

    if ltype == 'close multi-sinusoidal':
        ywave1_idx = []
        for j in range(N):
            wave = np.rint(A * np.sin(N * curvature * x[j])) + M // 2
            ywave_idx.append(wave.astype('int'))
            ywave1_idx.append(ywave_idx[j] + A // 6)
            test_img[ywave_idx[j]:M, j] = intensity
            test_img[ywave_idx[j] + A // 6:M, j] = 1 - intensity

    elif ltype == 'co-sinusoidal':
        for j in range(N):
            wave = np.rint(A * np.cos(N * curvature * x[j])) + M // 2
            ywave_idx.append(wave.astype('int'))
            test_img[ywave_idx[j]:M, j] = intensity

    elif ltype == 'diag':
        for j in range(N):
            test_img[j:, j] = intensity
            ywave_idx.append(j)

    elif ltype == 'straight':
        test_img[int(M // 2):, :] = intensity
        for j in range(N):
            ywave_idx.append(M // 2)

    # Store edge indexes
    edge_idx = np.asarray(list(zip(ywave_idx, xwave_idx)))

    if ltype == 'multi-sinusoidal' or ltype == 'close multi-sinusoidal':
        edge_idx = np.concatenate((edge_idx, np.asarray(list(zip(ywave1_idx, xwave_idx)))), axis=0)

    # Gaps to simulate vessel shadows
    if gaps:
        test_img[:, 20:30] = 0
        test_img[:, N // 2:(N // 2 + 10)] = 0
        test_img[:, N - 100:N - 90] = 0
        test_img[:, N // 4:(N // 4 + 20)] = 0

    # Add random Gaussian noise
    test_img = random_noise(test_img, mode='gaussian', mean=0, var=noise_level, rng=1)

    return test_img, edge_idx
