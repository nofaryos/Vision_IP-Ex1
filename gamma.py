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
import cv2 as cv

from ex1_utils import LOAD_GRAY_SCALE, imReadAndConvert


def gammaImage(image, gamma: float):
    newImg = pow(image, gamma)
    newImg /= 255
    return newImg


def on_trackbar(beta):
    pass


def gammaDisplay(img_path: str, rep: int):
    # Create the window trackbar
    cv.namedWindow('Gamma Correction')
    # Create the trackbar
    cv.createTrackbar("0 x 2", "Gamma Correction", 0, 200, on_trackbar)

    while cv.getWindowProperty('Gamma Correction', 0) >= 0:
        image = cv.imread(img_path)
        slide = cv.getTrackbarPos("0 x 2", "Gamma Correction")
        slide = float(slide) / 100.0
        image = gammaImage(image, slide)
        # RGB picture
        if rep == 2:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            image = cv.imread(image, cv.IMREAD_GRAYSCALE)
        # Gray scale picture
        cv.imshow("Gamma Correction", image)

        if cv.waitKey(1) == 27:
            break
    cv.destroyAllWindows()


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
