import math 
import cv2
import numpy as np 
from filters import Sobel, Gaussian

class KeypointsDetector(object): 
    def detectKeypoints(self, image):
        raise NotImplementedError()
    
class Harris_KD(KeypointsDetector): 
    def __init__(self, kernel_size=21, sigma=5.,  k=0.05, threshold=1e6):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.k = k 
        self.threshold = threshold
        self.sobel = Sobel()
        self.gaussian = Gaussian(kernel_size=kernel_size, sigma=sigma)

    def compute(self, image):
        height, width, channel = image.shape

        output_img = np.zeros(shape=(height, width))
        orientation_img  = np.zeros(shape=(height, width))
        gray_img = 0.299*image[:, :, 2] + 0.587*image[: , :, 1] + 0.114*image[:, :, 0]
        gray_img = gray_img.astype(np.uint8)

        # Step 1: Compute Gradients
        Ix = self.sobel.correlate(gray_img, axis=0)
        Iy = self.sobel.correlate(gray_img, axis=1)

        # Step 2: Compute matrix M 
        Ixx = Ix**2
        Ixy = Ix*Iy
        Iyy = Iy**2

        # Step 3: Smooth the products with Gaussian filter
        w_Ixx = Gaussian.correlate(Ixx)
        w_Iyy = Gaussian.correlate(Iyy)
        w_Ixy = Gaussian.correlate(Ixy)

        # Step 4: Compute Harris response
        det_M = w_Ixx * w_Iyy - w_Ixy**2
        trace = w_Ixx + w_Iyy
        output_img = det_M - 0.06 * (trace**2)
        orientation_img = np.degrees(np.arctan2(Iy.flatten(), Ix.flatten()).reshape(orientation_img.shape))

        return output_img, orientation_img

    def non_max_suppression(self, harris_img, window_size=3, threshold=1e6):
        output_img = np.zeros_like(harris_img, np.bool)

        height, width = harris_img.shape
        offset = window_size // 2

        for i in range(1, height - 1):
            for j in range(1, width - 1): 
                if harris_img[i, j] > threshold: 
                    local_patch = harris_img[i - offset: i - offset + 1, j - offset: j - offset + 1 ] 
                    if harris_img[i, j] == np.max(local_patch): 
                        output_img[i, j] = 255
        
        return output_img
    
    def detectKeypoints(self, image):
        height, width, channel = image.shape
        features = []

        harris_img, orientation_img = self.compute(image)
        harris_nms_img = self.non_max_suppression(harris_img, window_size=3, threshold=self.threshold)

        for i in range(height): 
            for j in range(width): 
                if not harris_nms_img[i, j] == 255: 
                    continue
                
                f = cv2.KeyPoint()

                f.pt = (j, i)
                f.size = 10
                f.angle = orientation_img[i, j]
                f.response = harris_img[i, j]

                features.append(f)

        return features

class ShiTomasi_KD(KeypointsDetector): 
    def __init__(self, kernel_size=21, sigma=5.0, threshold=1e6):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.threshold = threshold
        self.sobel = Sobel()
        self.gaussian = Gaussian(kernel_size=kernel_size, sigma=sigma)

    def compute(self, image):
        height, width, channel = image.shape

        response_img = np.zeros(shape=(height, width))
        orientation_img = np.zeros(shape=(height, width))
        gray_img = 0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]
        gray_img = gray_img.astype(np.uint8)

        # Step 1: Compute Gradients
        Ix = self.sobel.correlate(gray_img, axis=0)
        Iy = self.sobel.correlate(gray_img, axis=1)

        # Step 2: Compute second moment matrix elements
        Ixx = Ix ** 2
        Ixy = Ix * Iy
        Iyy = Iy ** 2

        # Step 3: Smooth the components with Gaussian filter
        w_Ixx = self.gaussian.correlate(Ixx)
        w_Iyy = self.gaussian.correlate(Iyy)
        w_Ixy = self.gaussian.correlate(Ixy)

        # Step 4: Compute min eigenvalue (Shi-Tomasi response)
        for i in range(height):
            for j in range(width):
                M = np.array([[w_Ixx[i, j], w_Ixy[i, j]],
                              [w_Ixy[i, j], w_Iyy[i, j]]])
                eigvals = np.linalg.eigvalsh(M)
                response_img[i, j] = np.min(eigvals)

        orientation_img = np.degrees(np.arctan2(Iy.flatten(), Ix.flatten()).reshape(orientation_img.shape))

        return response_img, orientation_img

    def non_max_suppression(self, response_img, window_size=3, threshold=1e6):
        output_img = np.zeros_like(response_img, np.bool_)
        height, width = response_img.shape
        offset = window_size // 2

        for i in range(offset, height - offset):
            for j in range(offset, width - offset): 
                if response_img[i, j] > threshold: 
                    local_patch = response_img[i - offset:i + offset + 1, j - offset:j + offset + 1]
                    if response_img[i, j] == np.max(local_patch): 
                        output_img[i, j] = 255
        
        return output_img

    def detectKeypoints(self, image):
        height, width, channel = image.shape
        features = []

        response_img, orientation_img = self.compute(image)
        nms_img = self.non_max_suppression(response_img, window_size=3, threshold=self.threshold)

        for i in range(height): 
            for j in range(width): 
                if nms_img[i, j] != 255: 
                    continue
                
                f = cv2.KeyPoint()
                f.pt = (j, i)  # lưu ý: (x, y) = (col, row)
                f.size = 10
                f.angle = orientation_img[i, j]
                f.response = response_img[i, j]

                features.append(f)

        return features
    
class SIFT_KD(KeypointsDetector):
    def __init__(self, num_octaves=4, num_scales=5, sigma=1.6, threshold=5):
        self.num_octaves = num_octaves
        self.num_scales = num_scales
        self.sigma = sigma
        self.threshold = threshold

    def compute(self, image):
        height, width, channel = image.shape
        gray_img = 0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]
        gray_img = gray_img.astype(np.uint8)

        # Step 1: Build Gaussian pyramid
        gaussian_pyramid = self.build_gaussian_pyramid(gray_img)

        # Step 2: Build Difference of Gaussian (DoG) pyramid
        dog_pyramid = self.build_dog_pyramid(gaussian_pyramid)

        # Step 3: Detect extrema in DoG pyramid (scale-space keypoints)
        response_img = np.zeros_like(gray_img, dtype=np.float32)
        orientation_img = np.zeros_like(gray_img, dtype=np.float32)

        for o, dog_octave in enumerate(dog_pyramid):
            for s in range(1, len(dog_octave)-1):
                prev, curr, next = dog_octave[s-1], dog_octave[s], dog_octave[s+1]
                for i in range(1, curr.shape[0]-1):
                    for j in range(1, curr.shape[1]-1):
                        patch = np.stack([
                            prev[i-1:i+2, j-1:j+2],
                            curr[i-1:i+2, j-1:j+2],
                            next[i-1:i+2, j-1:j+2]
                        ])
                        val = curr[i, j]
                        if abs(val) > self.threshold and (val == patch.max() or val == patch.min()):
                            y, x = i * (2 ** o), j * (2 ** o)
                            if y < height and x < width:
                                response_img[y, x] = abs(val)
                                orientation_img[y, x] = 0  # Orientation assignment not yet implemented

        return response_img, orientation_img

    def build_gaussian_pyramid(self, image):
        k = 2**(1 / (self.num_scales - 3))
        pyramid = []
        for o in range(self.num_octaves):
            octave = []
            base = cv2.resize(image, (image.shape[1] >> o, image.shape[0] >> o), interpolation=cv2.INTER_NEAREST)
            for s in range(self.num_scales):
                sigma = self.sigma * (k ** s)
                blurred = cv2.GaussianBlur(base, (0, 0), sigmaX=sigma, sigmaY=sigma)
                octave.append(blurred)
            pyramid.append(octave)
        return pyramid

    def build_dog_pyramid(self, gaussian_pyramid):
        dog_pyramid = []
        for octave in gaussian_pyramid:
            dog_octave = []
            for i in range(1, len(octave)):
                dog = cv2.subtract(octave[i], octave[i - 1])
                dog_octave.append(dog)
            dog_pyramid.append(dog_octave)
        return dog_pyramid

    def non_max_suppression(self, response_img, window_size=3, threshold=5):
        output_img = np.zeros_like(response_img, np.bool_)
        height, width = response_img.shape
        offset = window_size // 2

        for i in range(offset, height - offset):
            for j in range(offset, width - offset):
                if response_img[i, j] > threshold:
                    local_patch = response_img[i - offset:i + offset + 1, j - offset:j + offset + 1]
                    if response_img[i, j] == np.max(local_patch):
                        output_img[i, j] = 255

        return output_img

    def detectKeypoints(self, image):
        height, width, channel = image.shape
        features = []

        response_img, orientation_img = self.compute(image)
        nms_img = self.non_max_suppression(response_img, window_size=3, threshold=self.threshold)

        for i in range(height):
            for j in range(width):
                if nms_img[i, j] != 255:
                    continue

                f = cv2.KeyPoint()
                f.pt = (j, i)  # Note: (x, y) = (col, row)
                f.size = 10
                f.angle = orientation_img[i, j]
                f.response = response_img[i, j]

                features.append(f)

        return features
                                        