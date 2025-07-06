import numpy as np

class Filter(object): 
    def __init__(self):
        self.values = None

    def correlate(self, img): 
        raise NotImplementedError("Subclasses should implement this!")

    def _convolve(self, img, kernel):
        """
        Apply a 2D convolution to each channel of the image using the given kernel.
        This implementation handles multi-channel images (e.g., RGB).
        """
        H, W, C = img.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        # Pad the image to handle borders
        padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='reflect')

        # Output image initialized
        result = np.zeros((H, W, C), dtype=np.float32)

        # Perform convolution channel by channel
        for c in range(C):
            for i in range(H):
                for j in range(W):
                    region = padded[i:i+kh, j:j+kw, c]
                    result[i, j, c] = np.sum(region * kernel)

        return result


class Sobel(Filter): 
    def __init__ (self):
        super().__init__()
        # Define 2 Sobel kernels: horizontal (x) and vertical (y)
        self.values = np.array([
            [  # Sobel x (horizontal)
                [-1, 0, 1], 
                [-2, 0, 2], 
                [-1, 0, 1]
            ], 
            [  # Sobel y (vertical)
                [-1, -2, -1], 
                [0, 0, 0], 
                [1, 2, 1]
            ]
        ], dtype=np.float32)

    def correlate(self, img, axis=0):
        """
        Apply the selected Sobel filter (horizontal or vertical) to the input image.
        axis=0 → Sobel X, axis=1 → Sobel Y
        """
        kernel = self.values[axis]
        return self._convolve(img, kernel)
    
class Gaussian(Filter): 
    def __init__(self, kernel_size=5, sigma=1.): 
        """
        Initialize the Gaussian filter with a kernel of given size and sigma.
        """
        super().__init__()
        self.values = self.generate_kernel(kernel_size, sigma)

    def generate_kernel(self, size, sigma):
        """
        Initialize the Gaussian filter with a kernel of given size and sigma.
        """ 
        ax = np.linspace(-(size // 2), size // 2, size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel / np.sum(kernel)
    

    def correlate(self, img):
        """
        Apply the Gaussian filter to the input image.
        """
        return self._convolve(img, self.values)
    
