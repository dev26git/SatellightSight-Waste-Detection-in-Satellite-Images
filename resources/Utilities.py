import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import normalize

# GLCM Utility function
properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']


def get_glcm_features(img, props, dists=[5], agls=[0], lvl=256, sym=True, norm=True):

    glcm = graycomatrix(img,
                        distances=dists,
                        angles=agls,
                        levels=lvl,
                        symmetric=sym,
                        normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in graycoprops(glcm, name)[0]]
    for item in glcm_props:
        feature.append(item)

    return feature


# LBP Utility function
def get_lbp_features(img):
    # Split the image into RGB channels
    r, g, b = cv2.split(img)

    # Calculate LBP image and histogram for each channel
    radius = 1
    n_points = 8
    lbp_r = local_binary_pattern(r, n_points, radius, 'uniform')
    lbp_g = local_binary_pattern(g, n_points, radius, 'uniform')
    lbp_b = local_binary_pattern(b, n_points, radius, 'uniform')
    hist_r, _ = np.histogram(lbp_r, bins=np.arange(0, n_points + 3), density=True)
    hist_g, _ = np.histogram(lbp_g, bins=np.arange(0, n_points + 3), density=True)
    hist_b, _ = np.histogram(lbp_b, bins=np.arange(0, n_points + 3), density=True)

    # # Normalize histograms
    hist_r = normalize(hist_r[:, np.newaxis], axis=0).ravel()
    hist_g = normalize(hist_g[:, np.newaxis], axis=0).ravel()
    hist_b = normalize(hist_b[:, np.newaxis], axis=0).ravel()

    # # Concatenate histograms
    hist = np.concatenate([hist_r, hist_g, hist_b])

    # # Extract features
    mean_r = np.mean(lbp_r)
    mean_g = np.mean(lbp_g)
    mean_b = np.mean(lbp_b)
    std_r = np.std(lbp_r)
    std_g = np.std(lbp_g)
    std_b = np.std(lbp_b)
    skewness_r = np.mean(((lbp_r - mean_r) / std_r) ** 3)
    skewness_g = np.mean(((lbp_g - mean_g) / std_g) ** 3)
    skewness_b = np.mean(((lbp_b - mean_b) / std_b) ** 3)
    kurtosis_r = np.mean(((lbp_r - mean_r) / std_r) ** 4) - 3
    kurtosis_g = np.mean(((lbp_g - mean_g) / std_g) ** 4) - 3
    kurtosis_b = np.mean(((lbp_b - mean_b) / std_b) ** 4) - 3

    arr = [mean_r, mean_g, mean_b, std_r, std_g, std_b, skewness_r, skewness_g, skewness_b, kurtosis_r, kurtosis_g, kurtosis_b]
    return np.concatenate((arr, hist))


def get_features(img_rgb):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # get glcm features
    glcm_features = get_glcm_features(img_gray, props=properties)
    # get lbp features
    lbp_features = get_lbp_features(img_rgb)

    return np.concatenate((glcm_features, lbp_features))


def get_windows(img, window_height: int, window_width: int):
    img_height, img_width, _ = img.shape

    top = 0
    bottom = window_height

    img_list = []
    while bottom <= img_height:
        left = 0
        right = window_width
        while right <= img_width:
            # cropped_img = img[top:bottom, left:right]
            img_list.append(img[top:bottom, left:right])

            left += round(window_width / 2)
            right += round(window_width / 2)

        top += round(window_height / 2)
        bottom += round(window_height / 2)

    return img_list


def get_window_coords(img, window_height, window_width):
    img_height, img_width, _ = img.shape

    top = 0
    bottom = window_height

    window_coord_list = []  # Will store coordinates of all windows as tuples
    while bottom <= img_height:
        left = 0
        right = window_width
        while right <= img_width:
            # cropped_img = img[top:bottom, left:right]
            window_coord_list.append((top, bottom, left, right))

            left += round(window_width / 2)
            right += round(window_width / 2)

        top += round(window_height / 2)
        bottom += round(window_height / 2)

    return window_coord_list

