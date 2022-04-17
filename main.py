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
    rel_path = "images/3.png"
    abs_file_path = os.path.join(script_dir, rel_path)
    image = imread(abs_file_path)
    #chromaticityGraph(image)
    shadowRemoval(image)
    #chromaticityGraph(image)

def shadowRemoval(image):
    angle = getBestAngle(image)
    # invariantImage = getInvariantImage(image, 100)
    # imshow(invariantImage)
    # show()
    pass

def getBestAngle(image):
    maxI = 0.95  #maximal pixel intensity
    minI = 0.05  #maximal pixel intensity
    entropy = []
    numOfPx = np.shape(image)[0] * np.shape(image)[1]
    numOfAngles = 181

    # invariantImage1 = getInvariantImage(image, 0)
    # plotHistogramFromInvrariantImage(invariantImage1)
    # plotInvariantImage(invariantImage1)
    # invariantImage2 = getInvariantImage(image, 100)
    # plotHistogramFromInvrariantImage(invariantImage2)
    # plotInvariantImage(invariantImage2)

    invariantImage1 = getInvariantImage(image, 126, plot=True)
    plotHistogramFromInvrariantImage(invariantImage1)
    plotInvariantImage(invariantImage1)

    for angle in range(0, 181):
        print("ANGLE > ", str(angle))
        invariantImage = getInvariantImage(image, angle)
        # plotInvariantImage(invariantImage)
        flattened = invariantImage.flatten()
        flattened = trimmed_percentiles(flattened, 40)
        #print("len:" + str(np.shape(flattened)) + " MIN: " + str(np.amin(flattened)) + "MAX: " + str(np.amax(flattened)))
        #print(flattened)

        #plotVarianceInvariantImage(invariantImage)

        shared_bins = np.histogram_bin_edges(flattened, bins='scott')
        hist, binedges = np.histogram(flattened, bins=shared_bins, density=True)
        #numOfPx2 = np.shape(invariantImage)[0] * np.shape(invariantImage)[1]

        #outliers = [x for x in invariantImageFlat if x>=maxI or x<=minI]
        #inliers = [x for x in invariantImageFlat if x<maxI and x>minI]

        # plt.figure()
        # plt.title("Grayscale Histogram")
        # plt.xlabel("grayscale value")
        # plt.ylabel("pixel count")
        # plt.plot(binedges[0:-1], hist)
        # plt.show()

        # meanValue = np.mean(invariantImageFlat)
        # std = np.std(invariantImageFlat)
        # print("meanValue ",  meanValue)
        # print("stdDev ", std)
        # for i in hist[0]:
        #     ent -= i * math.log(abs(i))
        data = hist
        data[data == 0] = 0.000001
        ent = -(data * np.log(np.abs(data))).sum()
        entropy.append(ent)
        #print(ent)
        print("angle: ", angle, "  e: ", ent)
        #plotVarianceInvariantImage(invariantImage)
        #plotHistogramFromInvrariantImage(invariantImage)

    bestAngle = entropy.index(min(entropy))
    print("BEST angle: ", bestAngle, "  e: ", min(entropy))
    plt.figure()
    plt.title("Entropies")
    plt.plot(entropy)
    plt.show()

    pass

def trimmed_percentiles(data, percent):
    data = np.sort(data)
    trim = int(percent*np.shape(data)[0]/100.0)
    return data[trim:-trim]

def plotHistogramFromInvrariantImage(invariantImage):
    shared_bins = np.histogram_bin_edges(invariantImage.flatten(), bins='scott')
    hist, binedges = np.histogram(invariantImage.flatten(), bins=shared_bins)

    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    plt.plot(binedges[0:-1], hist)
    plt.show()

def plotInvariantImage(invariantImage):
    # plt.figure()
    # plt.imshow(invariantImage, cmap=plt.get_cmap('gray'),
    #            vmin=0, vmax=1)
    # plt.show()
    imshow(invariantImage)
    show()

def plotVarianceInvariantImage(invariantImage):
    invariantImage = invariantImage.flatten()
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.plot(invariantImage, len(invariantImage) * [1], "x")
    plt.show()

def getInvariantImage(image, rho, plot= False):
    chromaticities2D = caluclate2DChromaticitiesFromImage(image)
    cosine = np.cos(np.radians(rho))
    sine = np.sin(np.radians(rho))

    if plot:
        plot2DChromaticity(chromaticities2D[:, :, 0], chromaticities2D[:, :, 1], rho)

    firstCos = np.multiply(chromaticities2D[:, :, 0], cosine)
    secondSin = np.multiply(chromaticities2D[:, :, 1], sine)
    intrinsicImage = np.add(firstCos, secondSin)
    return intrinsicImage

def caluclate2DChromaticitiesFromImage(image):
    log_chromaticies_3D = calculate3DChromaticitiesFromImage(image)
    U = [[1 / math.sqrt(2), -1 / math.sqrt(2), 0], [1 / math.sqrt(6), 1 / math.sqrt(6), -2 / math.sqrt(6)]]
    U = np.array(U)
    X = np.dot(log_chromaticies_3D, U.T)
    #plot2DChromaticity(X[:,:,0], X[:,:,1])
    return X

def plot2DChromaticity(img_r_g, img_b_g, rho):
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.plot(img_r_g, img_b_g, 'o', markersize=0.4)
    plt.ylabel('Log(b/g)')
    plt.xlabel('Log(r/g)')

    # PLOT LINE WITH RHO
    # x,y = (np.amin(img_r_g), np.amin(img_b_g))
    # length = 1
    # # find the end point
    # endy = y + length * math.sin(math.radians(rho))
    # endx = length * math.cos(math.radians(rho))
    # plt.plot([x, endx], [y, endy])
    if rho == 90:
        plt.vlines(0, ymin=-1, ymax=1)
    else :
        m1, b1 = math.tan(math.radians(rho)), 0.0  # slope & intercept (line 1)
        print("m1: ", m1)
        x = np.linspace(-1, 1, 500)
        plt.plot(x, x * m1 + b1)


    plt.show()
    pass

def chromaticityGraph(image):
    rowsNum, columnNum, colorsNum = np.shape(image)
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
    # pixels_r_g = img_r_g.reshape((rowsNum, columnNum))
    # pixels_b_g = img_b_g.reshape((rowsNum, columnNum))
    # fig, ax = plt.subplots(1, 2, figsize=(17, 7), sharey=False)
    # ax[0].imshow(pixels_r_g)
    # ax[1].imshow(pixels_b_g)
    # ax[0].set_title('pixels_r_g', fontsize=22)
    # ax[1].set_title('pixels_b_g', fontsize=22)
    # show()

def calculate3DChromaticitiesFromImage(image):
    image = np.float64(image)
    r = image[:,:,0]
    g = image[:,:,1]
    b = image[:,:,2]
    r[r == 0] = 1
    g[g == 0] = 1
    b[b == 0] = 1
    # r = np.interp(r, [0, 255], [0, 1])
    # g = np.interp(g, [0, 255], [0, 1])
    # b = np.interp(b, [0, 255], [0, 1])

    # 3D chromaticity with geometric mean
    geometric_mean = np.multiply(np.multiply(r,g),b) ** (1.0/3)
    geometric_mean[geometric_mean == 0] = 1
    chromaticity_r = np.log(np.divide(r, geometric_mean))
    chromaticity_g = np.log(np.divide(g, geometric_mean))
    chromaticity_b = np.log(np.divide(b, geometric_mean))
    # print("chromaticity_r")
    # print(chromaticity_r)
    # print("chromaticity_g")
    # print(chromaticity_g)
    # print("chromaticity_b")
    # print(chromaticity_b)

    #c chormaticity to r chromaticity
    # chromaticity_sum = np.add(np.add(chromaticity_r, chromaticity_g), chromaticity_b)
    # chromaticity_sum[chromaticity_sum == 0] = 10
    # chromaticity_r = np.divide(chromaticity_r, chromaticity_sum)
    # chromaticity_g = np.divide(chromaticity_g, chromaticity_sum)
    # chromaticity_b = np.divide(chromaticity_b, chromaticity_sum)
    # # print("chromaticity_sum")
    # # print(chromaticity_sum)
    # # print("chromaticity_r")
    # # print(chromaticity_r)
    # chromaticity_r[chromaticity_r == 0] = 0.0001
    # chromaticity_g[chromaticity_g == 0] = 0.0001
    # chromaticity_b[chromaticity_b == 0] = 0.0001

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
