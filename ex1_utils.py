"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import math
from typing import List
import cv2
import numpy as np
import matplotlib.pyplot as plt

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 208583476


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    # Gray scale image
    if representation == 1:
        # reading the picture with cv2 GRAYSCALE
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # RGB image
    else:
        # reading the picture with cv2 coloring image and convert it to RGB
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #  normalized to the range [0, 1]
    img_rgb = cv2.normalize(img, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img_rgb


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    # Gray scale image
    if representation == 1:
        img = imReadAndConvert(filename, representation)
        plt.imshow(img, cmap='gray')
        plt.show()

    # RGB image
    else:
        img = imReadAndConvert(filename, representation)
        print("color;", img.shape)
        plt.imshow(img)
        plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    yiqFromRgb = np.array([[0.299, 0.587, 0.114],
                           [0.59590059, -0.27455667, -0.32134392],
                           [0.21153661, -0.52273617, 0.31119955]])
    yiq = np.dot(imgRGB.reshape(-1, 3), yiqFromRgb).reshape(imgRGB.shape)

    return yiq


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    yiqFromRgb = np.array([[0.299, 0.587, 0.114],
                           [0.59590059, -0.27455667, -0.32134392],
                           [0.21153661, -0.52273617, 0.31119955]])
    rgbFromYiq = np.linalg.inv(yiqFromRgb)
    rgb = np.dot(imgYIQ.reshape(-1, 3), rgbFromYiq).reshape(imgYIQ.shape)

    return rgb


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    # check if the picture is RGB or GRAY SCALE
    mode = isGrey(imgOrig)

    # Gray image
    if mode:
        old_img = cv2.normalize(imgOrig, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # RGB image
    else:
        # Transform the picture to YIQ color space
        imgYIQ = transformRGB2YIQ(imgOrig)
        # In YIQ image the histogram Equalize procedure should only operate on the Y channel
        old_img = cv2.normalize(imgYIQ[:, :, 0], 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Compute the image histogram
    histOrg, bins = np.histogram(old_img.flatten(), 256, [0, 256])

    # Compute the cumulative histogram
    cdf = np.cumsum(histOrg)

    # Normalize the cumulative histogram
    cdf_norm = cdf / cdf.max()

    # Creating matrix for the new image
    imEq = np.zeros_like(old_img)

    # Compute the new intensities by
    # normalized histogram by the maximal intensity value,
    #  and round the values to get integers
    # Map the new intensities to the new picture
    for i in range(len(cdf_norm)):
        imEq[old_img == i] = np.around(cdf_norm[i] * 255).astype(int)

    # Compute the new image histogram
    histEQ, bins = np.histogram(imEq.ravel(), 256, [0, 256])

    # Normalized the new image
    imEq = cv2.normalize(imEq, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    if mode:
        return imEq, histEQ, histOrg
    else:
        imgYIQ = transformRGB2YIQ(imgOrig)
        imgYIQ[:, :, 0] = imEq
        imEq = transformYIQ2RGB(imgYIQ)
        return imEq, histEQ, histOrg


def isGrey(img: np.ndarray):
    """
    Function that check if image is GRAY_SCALE image or RGB picture
    by checking the shape of the image
    :param img
    :ret
"""
    if len(img.shape) == 2:
        return True
    else:
        return False


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
    Quantized an image in to **nQuant** colors
    :param imOrig: The original image (RGB or Gray scale)
    :param nQuant: Number of colors to quantize the image to
    :param nIter: Number of optimization loops
    :return: (List[qImage_i],List[error_i])
"""
    images = []
    errors = []
    gray = isGrey(imOrig)
    if gray:
        img = cv2.normalize(imOrig, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        # Transform the picture to YIQ color space
        # In YIQ image the quantization procedure should only operate on the Y channel
        img = cv2.normalize(transformRGB2YIQ(imOrig)[:, :, 0], 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Compute the image histogram
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    numPixels = img.size

    # Compute the initial borders
    borders = initialBorders(img, nQuant)

    # Compute qi values
    qi = updateQi(borders, hist)
    images.append(updateImage(img, borders, qi))
    errors.append(mse(img, images[0]))

    for i in range(nIter - 1):
        # Compute the image histogram
        hist, bins = np.histogram(images[-1].flatten(), 256, [0, 256])

        # Update the borders according to qi values
        borders = updateBorders(qi)

        # Update the qi values by the new borders
        qi = updateQi(borders, hist)

        # Creating the new image by the borders and qi values
        newImg = updateImage(images[i], borders, qi)

        # Compute the mse error
        error = mse(images[i], newImg)

        # If we have not been able to improve the image we will stop the process
        if errors[i] == error:
            break
        errors.append(error)
        images.append(newImg)

        # If the error is 0 we will stop in the process
        if error == 0:
            break

    if gray:
        for i in range(len(images)):
            images[i] = cv2.normalize(images[i], 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    else:
        for i in range(len(images)):
            images[i] = updateYiqImg(imOrig, images[i])

    return images, errors


def initialBorders(img, nQuant: int) -> np.ndarray:
    """
    A Function that compute the initial borders of an image in the
    quantization progress, this function set the initial division
    such that each segment will contain approximately
    the same number of pixels
    :param img
    :param nQuant: number of intensities
    :return borders
    """
    arr = img.flatten()
    arr.sort()
    q = []
    from collections import Counter
    print(Counter(arr).most_common())
    colors = sorted(set(img.flatten()))
    for i in range(nQuant):
        qi = colors[int(len(colors) * i / nQuant)]
        if qi in q:
            qi += 1
        if qi >= 256:
            break
        q.append(qi)
    q.append(256)
    return np.asarray(q)


def updateBorders(qi: List[int]) -> np.ndarray:
    """
    A Function that update the borders of the image on the
    quantization progress, this function set the borders
    by computing the average between each two qi values.
    :param qi
    :return borders
    """
    result = [0]
    for i in range(len(qi) - 1):
        result.append(int(round((qi[i] + qi[i + 1]) / 2)))
    result.append(256)
    return np.asarray(result)


def updateQi(borders: np.ndarray, hist: np.ndarray) -> List[int]:
    """
    A Function that update the qi values of the image on the
    quantization progress, this function set the qi values
    by computing the weighted average  between each two borders.
    :param borders:
    :param hist: histogram of the original image
    :return List[int] qi values
    """
    qi = []
    for i in range(len(borders) - 1):
        low = borders[i]
        high = borders[i + 1]
        sumHist = hist[low: high]
        if sumHist.sum() == 0:
            qi.append(low)
        else:
            qi.append(((sumHist * (np.arange(low, high))).sum() / hist[low:high].sum()))
    return qi


def updateImage(im: np.ndarray, borders: np.ndarray, qi: List[int]) -> np.ndarray:
    """
    A Function that update the new image by the borders and qi values in the
    quantization progress
    :param im: the last picture we updated
    :param borders
    :param qi values
    :return np.array of the new image
    """
    arr = np.zeros(256, dtype=np.uint8)
    for color, low, high in zip(qi, borders, borders[1:]):
        arr[low:high] = int(round(color))
    return arr[im].astype(im.dtype)


def updateYiqImg(img_: np.ndarray, newYscale: np.ndarray):
    """
    A Function that update the Y channel of YIQ image and convert the picture to RBG picture.
    :param img_: YIQ image
    :param newYscale: new Y channel
    :return np.array update RGB image
    """
    yioImg = transformRGB2YIQ(img_)
    normalY = cv2.normalize(newYscale, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    yioImg[:, :, 0] = normalY
    return transformYIQ2RGB(yioImg)


def mse(prev_img: np.ndarray, new_img: np.ndarray) -> float:
    """
    Given two images this function calculates the mse error
    :param new_img:
    :param prev_img:
    :return mse error
    """
    return np.sqrt(((prev_img - new_img) ** 2).sum()) / prev_img.size
