import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, show
import math
from numpy import ndarray
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors


def main():
    script_dir = os.path.dirname(__file__)
    rel_path = "images/vransko310x310.png"
    abs_file_path = os.path.join(script_dir, rel_path)
    image = imread(abs_file_path)
    shadowRemoval(image)
    # chromaticityGraph(budapest)

def shadowRemoval(image):
    caluclate2DChromaticitiesFromImage(image)
    pass


def getInvariantImage(image):
    pass

def caluclate2DChromaticitiesFromImage(image):
    log_chromaticies_3D = calculate3DChromaticitiesFromImage(image)
    U = [[1 / math.sqrt(2), -1 / math.sqrt(2), 0], [1 / math.sqrt(6), 1 / math.sqrt(6), -2 / math.sqrt(6)]]
    U = np.array(U)
    X = np.dot(log_chromaticies_3D, U.T)

def chromaticityGraph(image):
    rowsNum, columnNum, colorsNum = np.shape(image)
    #print(np.shape(image))
    img_b_g = np.empty(rowsNum * columnNum)
    img_r_g = np.empty(rowsNum * columnNum)
    pixels = image.flatten().reshape(rowsNum*columnNum, 3)

    for x in range(np.shape(pixels)[0]):
        #sum = pixels[x, 0] + pixels[x, 1] + pixels[x, 2]
        r = pixels[x, 0]
        g = pixels[x, 1]
        b = pixels[x, 2]

        b_g = b/g
        r_g = r/g
        if b_g == 0 or r_g == 0:
            print('ZERO')
            continue
        img_b_g[x] = math.log(b_g)
        img_r_g[x] = math.log(r_g)

    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.plot(img_r_g, img_b_g, 'o', markersize=0.5)
    #plt.xlim([-0.4, 1.4])
    #plt.ylim([-1.6, 0])
    plt.ylabel('Log(b/g)')
    plt.xlabel('Log(r/g)')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.show()

    ## Make log chroma image
    pixels_r_g = img_r_g.reshape((rowsNum, columnNum))
    pixels_b_g = img_b_g.reshape((rowsNum, columnNum))
    fig, ax = plt.subplots(1, 2, figsize=(17, 7), sharey=False)
    ax[0].imshow(pixels_r_g)
    ax[1].imshow(pixels_b_g)
    ax[0].set_title('pixels_r_g', fontsize=22)
    ax[1].set_title('pixels_b_g', fontsize=22)
    show()

def calculate3DChromaticitiesFromImage(image):
    r = image[:,:,0]
    g = image[:,:,1]
    b = image[:,:,2]
    imshow(g, cmap ='Reds')
    show()

    # 3D chromaticity with geometric mean
    geometric_mean = np.multiply(np.multiply(r,g),b) ** (1.0/3)
    geometric_mean[geometric_mean == 0] = 1
    chromaticity_r = np.log(np.divide(r, geometric_mean))
    chromaticity_g = np.log(np.divide(g, geometric_mean))
    chromaticity_b = np.log(np.divide(b, geometric_mean))
    return np.stack((chromaticity_r, chromaticity_g, chromaticity_b), axis=2)  # log chromaticity


def rgb_splitter(image):
    rgb_list = ['Reds','Greens','Blues']
    fig, ax = plt.subplots(1, 3, figsize=(17,7), sharey = False)
    for i in range(3):
        ax[i].imshow(image[:,:,i], cmap = rgb_list[i])
        ax[i].set_title(rgb_list[i], fontsize = 22)
        ax[i].axis('off')
    fig.tight_layout()
    show()


if __name__ == "__main__":
    main()
