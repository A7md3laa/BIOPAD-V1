import cv2
import numpy as np

def build_gabor_filters():
    """Constructs and caches the spatial Gabor filter bank."""
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 4):
        # Parameters fixed: sigma=4.0, lambda=10.0
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters

GABOR_FILTERS = build_gabor_filters()

def extract_features_from_array(img) -> np.ndarray:
    """Executes the Stage 1 spatial feature extraction pipeline from an image array."""
    # Step 1-3: Resize, Equalize, Blur
    img_resized = cv2.resize(img, (128, 128))
    equalized = cv2.equalizeHist(img_resized)
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
    
    # Step 4: Directional Gabor convolution + pooling
    features = []
    for kern in GABOR_FILTERS:
        f_img = cv2.filter2D(blurred, cv2.CV_8UC3, kern)
        resized = cv2.resize(f_img, (16, 16), interpolation=cv2.INTER_AREA)
        features.append(resized.flatten())
        
    # Step 5: Fusion and normalisation -> 1024-D vector
    combined = np.concatenate(features) / 255.0
    return combined.reshape(1, -1)
