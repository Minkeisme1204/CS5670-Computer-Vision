import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Tạo kernel Gaussian 2D"""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def correlation_2d(img: np.ndarray, filter: np.ndarray) -> np.ndarray:
    H, W, C = img.shape
    kh, kw = filter.shape

    pad_h = kh // 2
    pad_w = kh // 2 

    padded = np.pad(img, pad_width=((pad_h, pad_h),(pad_w, pad_w), (0, 0)), mode='reflect')

    output = np.zeros_like(img, dtype=np.float64)

    for c in range(C):
        for i in range(H): 
            for j in range(W):
                region = padded[i:i+kh, j:j+kw, c] 
                output[i, j, c] = np.sum(region * filter)
    
    return output

def hybrid_images(img1_name: str, img2_name: str):
    """Hybrid images is Combining a lowpassed image with another image with highpassed component"""
    img1 = cv2.imread(img1_name)
    img2 = cv2.imread(img2_name)

    filter = gaussian_kernel(21, 5)

    lowpass = correlation_2d(img1, filter)
    highpass = img2 - correlation_2d(img2, filter)

    hybrid = np.clip(lowpass + highpass, 0, 255).astype(np.int8)
    hybrid = hybrid.astype(np.uint8)
    return hybrid 

hybrid = hybrid_images("left.jpg", "right.jpg")

cv2.imshow("Hybrid Image", hybrid)
cv2.waitKey(0)
cv2.destroyAllWindows()







