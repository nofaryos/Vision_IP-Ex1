# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from ex1_utils import imDisplay, LOAD_GRAY_SCALE, imReadAndConvert, transformRGB2YIQ, LOAD_RGB, hsitogramEqualize, \
    initialBorders, updateBorders

import numpy as np
from matplotlib import pyplot as plt
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #filename = 'bac_con.png'
    filename = 'beach.jpg'
    #filename = 'water_bear.png'
    # 1:
    #img = imReadAndConvert(filename, 1)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # print(img.dtype)
    # print(np.amax(img))
    # print(np.amin(img))

    # 2:
    #imDisplay(filename, 1)

    # 3:
    # imYIQ = transformRGB2YIQ(img)
    # cv2.imshow('image', imYIQ)
    # cv2.waitKey(0)

    # 4
    # print(img.shape)
    #hsitogramEqualize(img)

    # 5
    # quantizeImage(img, 90, 2)
    #print(is_grey_scale(img))
    import cv2
    image = cv2.imread(filename, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)
    plt.imshow(image)
    plt.show()
    #cv2.imshow('a',image)
    #cv2.waitKey(0)
    print(updateBorders([0,5,7,9,10,45,255]))




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
