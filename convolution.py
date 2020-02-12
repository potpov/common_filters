from PIL import Image
import numpy as np


class Convolution:

    def __init__(self, filename):
        self.set_image(filename)

    def set_image(self, filename):
        img = Image.open(filename)
        self.image = np.array(img.convert("L"))
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]
        self.result = np.ndarray((self.height, self.width))

    def create_padding(self, kernel_size):
        """
            add padding to the original image stored in the object
            according to the current kernel size, return the new padded image
            Params:
                - kernel_size: kernel dimension
            Return:
                - padded image
        """
        tmp = self.image
        kernel_offset = int((kernel_size-1)/2)
        padded = np.full((tmp.shape[0]+(kernel_offset*2), tmp.shape[1]+(kernel_offset*2)), 0)
        padded[kernel_offset:-kernel_offset, kernel_offset:-kernel_offset] = tmp
        return padded

    def apply_filter(self, kernel):
        """
            calculate the new image applying a custom filter
            Params:
                - kernel: might be gaussian or mean filter
            Return:
                - new image stored in the object
        """
        padded = self.create_padding(kernel.shape[0])
        offset = int((kernel.shape[0] - 1) / 2)
        for i in range(offset, self.height):
            for j in range(offset, self.width):
                self.result[i - offset, j - offset] = np.sum(
                    padded[i - offset:i + offset + 1, j - offset: j + offset + 1] * kernel
                )

    def apply_median(self, kernel_size=3):
        """
            calculate the median convolution directly
            Params:
                - kernel_size: kernel dimension
            Return:
                - new image stored in the object
        """
        padded = self.create_padding(kernel_size)
        offset = int((kernel_size - 1) / 2)
        for i in range(offset, self.height):
            for j in range(offset, self.width):
                self.result[
                    i - offset, j - offset
                ] = np.sort(padded[
                    i - offset:i + offset + 1, j - offset: j + offset + 1
                    ].flatten())[int(((kernel_size*kernel_size)-1)/2)]

    def apply_bilateral(self, kernel_size=3, sigr=1, sigd=1):
        """
            calculate the bilateral filter and convolution
            Params:
                - kernel_size: kernel dimension
                - sigr: sigma value for range domain
                - sigd: sigma value for spatial domain
            Return:
                - new filtered image stored in the object
        """
        kernel = np.ndarray((kernel_size, kernel_size))
        padded = self.create_padding(kernel_size)
        offset = int((kernel_size - 1) / 2)
        for l in range(offset, self.height + offset):
            for m in range(offset, self.width + offset):
                norm = 0
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        d = ((i - l) ** 2 + (j - m) ** 2) / sigd ** 2
                        r = ((padded[l-(i - offset), m-(j - offset)] - padded[l, m]) / sigr) ** 2
                        kernel[i, j] = np.exp(-0.5 * (d + r))
                        norm = norm + kernel[i, j]

                self.result[l - offset, m - offset] = np.sum(
                    padded[l - offset:l + offset + 1, m - offset: m + offset + 1] * kernel
                ) / norm

    def save(self, destination_name):
        im = Image.fromarray(self.result).convert('RGB')
        im.save(destination_name)

    @staticmethod
    def get_mean_filter(size):
        """
            create the mean filter
            Params:
                - size: kernel dimension
            Return:
                - mean kernel
        """
        return np.ones((size, size)) / (size * size)

    @staticmethod
    def get_gaussian_filter(size, sig=1):
        """
            create the gaussian kernel
            Params:
                - size: kernel dimension
                - sigma: sigma value, default=1
            Return:
                - gaussian kernel
        """
        kernel = np.ndarray((size, size))
        center = int((size - 1) / 2)

        coeff = 1 / ((sig ** 2) * 2 * np.pi)
        exp_denum = 2 * (sig ** 2)
        for i in range(kernel.shape[0]):
            exp_x = (i - center) ** 2
            for j in range(kernel.shape[1]):
                exp_y = (j - center) ** 2
                kernel[i, j] = coeff * np.exp(-1 * (exp_x + exp_y) / exp_denum)
        return kernel


def main():
    cv = Convolution("monna.png")

    mean_kernel = Convolution.get_mean_filter(3)
    cv.apply_filter(mean_kernel)
    cv.save("mean_result.png")

    cv.apply_median(3)
    cv.save("median_result.png")

    gaussian_kernel = Convolution.get_gaussian_filter(5)
    cv.apply_filter(gaussian_kernel)
    cv.save("gaussian_result.png")

    cv.apply_bilateral(21, sigd=100, sigr=100)
    cv.save("bilateral_result.png")

    cv.set_image("cat.png")
    cv.apply_bilateral(11, sigd=100, sigr=100)
    cv.save("bilateral_cat_result.png")


if __name__ == "__main__":
    main()
