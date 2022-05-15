import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import cv2 as cv
import math
from scipy.sparse import diags
from scipy.sparse import linalg

TRIMMED_PERCENTILE = 1
THRESHOLD_GRADM_INVARIANT = 0.08
THRESHOLD_GRADIENT = 0
TOP_COLOR_PERCENTILE = 96

ddepth = cv.CV_64F
x_der = np.array([[0, 0, 0],
                  [0, -1, 1],
                  [0, 0, 0]])
y_der = np.array([[0, 0, 0],
                  [0, -1, 0],
                  [0, 1, 0]])
x_sobel = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
y_sobel = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])
gx_to_lapl = np.array([[0, 0, 0],
                       [-1, 1, 0],
                       [0, 0, 0]])
gy_to_lapl = np.array([[0, -1, 0],
                       [0, 1, 0],
                       [0, 0, 0]])

def main():
    script_dir = os.path.dirname(__file__)
    rel_path = "images/ball1.png"
    abs_file_path = os.path.join(script_dir, rel_path)
    imageIn = imread(abs_file_path)
    imageOut = shadowRemoval(imageIn)
    cv.imwrite("output.png", imageOut)

def shadowRemoval(image):
    angle, entropies = getBestAngle(image)

    invariantImage, _ = compute_invariant_image(image, angle)  # Invariant image
    [gX, gY] = filter2D_replicate(invariantImage, x_sobel, y_sobel)         # Invariant gradient
    gM_invariant = gradient_magnitude(gX, gY)

    laplacR, laplacG, laplacB = getLaplacianS(image, gM_invariant)          # S
    [nRows, nCol] = np.shape(laplacR)

    r = poison_equation(laplacR, nRows, nCol)
    g = poison_equation(laplacG, nRows, nCol)
    b = poison_equation(laplacB, nRows, nCol)

    r = (toGrayscale(r)).astype('uint8')
    g = (toGrayscale(g)).astype('uint8')
    b = (toGrayscale(b)).astype('uint8')

    [rMean, gMean, bMean] = computeTopMean(r, g, b)
    r2 = np.interp(r, [0, rMean], [0, 255]).astype('uint8')
    g2 = np.interp(g, [0, gMean], [0, 255]).astype('uint8')
    b2 = np.interp(b, [0, bMean], [0, 255]).astype('uint8')

    rgbImage = np.stack((r2, g2, b2), axis=2)   # Full color image

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, gridspec_kw={'width_ratios': [1, 1, 1, 1] , 'height_ratios': [1]}, sharey=False)
    ax1.imshow(image, cmap='gray')
    ax1.set_title("Original")
    ax1.axis('off')
    ax2.plot(entropies)
    ax2.set_title("Entropies")
    flattened = trimmed_percentiles(invariantImage.flatten() , TRIMMED_PERCENTILE) # For visualization of invariant image
    ax3.imshow(invariantImage, vmin=flattened[0], vmax=flattened[-1], cmap='gray')
    ax3.set_title("Invariant image")
    ax3.axis('off')
    ax4.imshow(rgbImage, cmap='gray')
    ax4.set_title("Shadows removed")
    ax4.axis('off')
    plt.show()

    # shadow_edge_map = shadowEdgeDetection(image, invariantImage)
    return rgbImage

def getBestAngle(image):
    angles = []
    entropy = []
    numOfAngles = 181
    step = 1

    for angleDeg in range(0, numOfAngles, step):
        angle = np.radians(angleDeg)
        angles.append(angle)
        invariantImage, chromaticities = compute_invariant_image(image, angle)
        flattened = trimmed_percentiles(invariantImage.flatten(), TRIMMED_PERCENTILE)

        hist, binedges = np.histogram(flattened, bins='scott', density=True)  # Calculate histogram
        hist = np.divide(hist, sum(hist), where=sum(hist) != 0)
        hist[hist == 0] = 0.00000000000001

        ent = -1 * np.sum(np.multiply(hist, np.log2(hist)))  # Entropy
        entropy.append(ent)

        print(np.degrees(angle).astype('uint8'), " entropy: ", ent)
        #plotHistogramAndVarianceFromInvrariantImage(invariantImage, chromaticities, angle)

    bestAngle = angles[entropy.index(min(entropy))]
    print("BEST angle: ", np.degrees(bestAngle).astype('uint8'), "  entropy: ", min(entropy))

    return bestAngle, entropy

def compute_invariant_image(image, angle):
    imgR = image[:, :, 0] / 255
    imgG = image[:, :, 1] / 255
    imgB = image[:, :, 2] / 255
    imgR[imgR == 0] = 1
    imgG[imgG == 0] = 1
    imgB[imgB == 0] = 1

    geomM = (np.multiply(np.multiply(imgR, imgG), imgB)) ** (1 / 3)
    chromaticity_r = np.log(np.divide(imgR, geomM))
    chromaticity_b = np.log(np.divide(imgB, geomM))
    return chromaticity_r * np.cos(angle) + chromaticity_b * np.sin(angle), [chromaticity_r, chromaticity_b]

def filter2D_replicate(img, xKernel, yKernel):
    border_type = cv.BORDER_REPLICATE
    ddepth = cv.CV_64F
    temp = cv.copyMakeBorder(img, 1, 1, 1, 1, border_type, None)
    fx = cv.filter2D(src=temp, ddepth=ddepth, kernel=xKernel)
    fy = cv.filter2D(src=temp, ddepth=ddepth, kernel=yKernel)
    fx = fx[1:-1, 1:-1]
    fy = fy[1:-1, 1:-1]
    return [fx, fy]

def getLaplacianS(image, invariantGradientMagnitude):
    # image = cv.pyrMeanShiftFiltering(src=image, sp=9, sr=50)
    imgR = image[:, :, 0]
    imgG = image[:, :, 1]
    imgB = image[:, :, 2]

    gX_red = cv.filter2D(src=imgR, ddepth=ddepth, kernel=x_der)  # Red gradient
    gY_red = cv.filter2D(src=imgR, ddepth=ddepth, kernel=y_der)
    gM_red = gradient_magnitude(gX_red, gY_red)

    gX_green = cv.filter2D(src=imgG, ddepth=ddepth, kernel=x_der)  # Green gradient
    gY_green = cv.filter2D(src=imgG, ddepth=ddepth, kernel=y_der)
    gM_green = gradient_magnitude(gX_green, gY_green)

    # Blue
    gX_blue = cv.filter2D(src=imgB, ddepth=ddepth, kernel=x_der)  # Blue gradient
    gY_blue = cv.filter2D(src=imgB, ddepth=ddepth, kernel=y_der)
    gM_blue = gradient_magnitude(gX_blue, gY_blue)


    redX2 = np.copy(gX_red)  # Threshold gradients
    redY2 = np.copy(gY_red)
    redX2[(gM_red > THRESHOLD_GRADIENT) & (invariantGradientMagnitude < THRESHOLD_GRADM_INVARIANT)] = 0
    redY2[(gM_red > THRESHOLD_GRADIENT) & (invariantGradientMagnitude < THRESHOLD_GRADM_INVARIANT)] = 0

    greenX2 = np.copy(gX_green)
    greenY2 = np.copy(gY_green)
    greenX2[(gM_green > THRESHOLD_GRADIENT) & (invariantGradientMagnitude < THRESHOLD_GRADM_INVARIANT)] = 0
    greenY2[(gM_green > THRESHOLD_GRADIENT) & (invariantGradientMagnitude < THRESHOLD_GRADM_INVARIANT)] = 0

    blueX2 = np.copy(gX_blue)
    blueY2 = np.copy(gY_blue)
    blueX2[(gM_blue > THRESHOLD_GRADIENT) & (invariantGradientMagnitude < THRESHOLD_GRADM_INVARIANT)] = 0
    blueY2[(gM_blue > THRESHOLD_GRADIENT) & (invariantGradientMagnitude < THRESHOLD_GRADM_INVARIANT)] = 0

    [redXX, _] = filter2D_replicate(redX2, gx_to_lapl, gy_to_lapl)  # Gradients to laplacian
    [_, redYY] = filter2D_replicate(redY2, gx_to_lapl, gy_to_lapl)

    [greenXX, _] = filter2D_replicate(greenX2, gx_to_lapl, gy_to_lapl)
    [_, greenYY] = filter2D_replicate(greenY2, gx_to_lapl, gy_to_lapl)

    [blueXX, _] = filter2D_replicate(blueX2, gx_to_lapl, gy_to_lapl)
    [_, blueYY] = filter2D_replicate(blueY2, gx_to_lapl, gy_to_lapl)

    laplacR = redXX + redYY
    laplacG = greenXX + greenYY
    laplacB = blueXX + blueYY

    return laplacR, laplacG, laplacB

def computeTopMean(red, green, blue):
    redTop = np.percentile(red, TOP_COLOR_PERCENTILE)
    greenTop = np.percentile(green, TOP_COLOR_PERCENTILE)
    blueTop = np.percentile(blue, TOP_COLOR_PERCENTILE)
    rMean = np.mean(red[red > redTop])
    gMean = np.mean(green[green > greenTop])
    bMean = np.mean(blue[blue > blueTop])

    return [rMean, gMean, bMean]

def toGrayscale(A):
    return cv.normalize(A, None, 0, 255, norm_type=cv.NORM_MINMAX)

def poison_equation(fun, width, height):
    # Grid parameters.
    nx = width
    ny = height

    b = np.copy(fun)
    bflat = b.flatten()

    A = mat_A(ny, nx)

    solution = linalg.spsolve(A, bflat)  # a_inverse.dot(bflat)
    pvec = np.reshape(solution, (nx, ny))

    return pvec

def mat_A(nx, ny):
    a = 1
    g = 1
    c = -2*a - 2*g

    diagonals = [g, a, c, a, g]  # Construct a sequence of main diagonal elements,
    offsets = [-nx, -1, 0, 1, nx]

    d2mat = diags(diagonals, offsets, shape=(nx * ny, nx * ny), format='csc')   # Call to the diags routine
    return d2mat

def trimmed_percentiles(data, percent):
    data = np.sort(data)
    if percent == 0:
        return data
    else:
        trim = int(percent * np.shape(data)[0] / 100.0)
    return data[trim:-trim]


def plotHistogramAndVarianceFromInvrariantImage(invariantImage, chromaticities, rho):
    flattened = trimmed_percentiles(invariantImage.flatten(), TRIMMED_PERCENTILE)

    figure, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, sharey=False)
    figure.suptitle("Angle: " + str(np.degrees(rho).astype('uint8')))

    # 1. Chromaticities plot
    ax1.plot(chromaticities[0], chromaticities[1], 'o', markersize=0.4)
    if rho == 90:
        ax1.vlines(0, ymin=-1, ymax=1)
    else:
        l = 5 # length
        m1, b1 = math.tan(rho), 0.0  # slope & intercept
        xl = np.linspace(-l, l, 100)
        ax1.set_ylim([-l, l])
        ax1.set_xlim([-l, l])
        ax1.plot(xl, xl * m1 + b1)

    # 2. Histogram
    hist, binedges = np.histogram(flattened, bins='scott')
    ax2.bar(binedges[:-1], hist, width=binedges[1] - binedges[0])

    # 3. Variance
    ax3.plot(flattened, len(flattened) * [1], "x", markersize="1")

    # 4. Image
    ax4.imshow(invariantImage, vmin=flattened[0], vmax=flattened[-1])

    plt.show()

def gradient_magnitude(gX, gY):
    return np.sqrt(gX ** 2.0 + gY ** 2.0)

if __name__ == "__main__":
    main()









# def jacobi(xk, b):
#     nx = np.shape(xk)[0] - 1
#     ny = np.shape(xk)[1] - 1
#     dx = 1 / nx
#     dy = 1 / ny
#
#     xkp1 = np.copy(xk)
#     for i in range(1, nx):
#         for j in range(1, ny):
#             xkp1[i, j] = (b[i, j] - ((xk[i + 1, j] + xk[i - 1, j]) / dx ** 2) - (
#                         (xk[i, j + 1] + xk[i, j - 1]) / dy ** 2)) / (-2 / dx ** 2 - 2 / dy ** 2)
#
#     return xkp1


# def shadowEdgeDetection(image, invariantImage):
#     img = cv.pyrMeanShiftFiltering(src=image, sp=9,
#                                    sr=50)  # plotInvariantImage(img) # sp – The spatial window radius., sr – The color window radius.
#     sigma = 0.33
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     # gray = cv.GaussianBlur(gray, (3, 3), 0.2)
#     v = np.median(gray)
#     # ---- apply automatic Canny edge detection using the computed median----
#     lower = int(max(0, (1.0 - sigma) * v))  # 255/3
#     upper = int(min(255, (1.0 + sigma) * v))  # 255
#     edges = cv.Canny(gray, threshold1=lower, threshold2=upper)
#
#     flattened = trimmed_percentiles(invariantImage.flatten(), TRIMMED_PERCENTILE)
#     invariantImage = np.interp(invariantImage, [flattened[0], flattened[-1]], [0, 255])
#     invariantImage = np.uint8(invariantImage)
#     # invariantImage = cv.GaussianBlur(invariantImage, (5, 5), 1.4)
#     # invariantImage = cv.pyrMeanShiftFiltering(src=np.stack((invariantImage, invariantImage, invariantImage), axis=2), sp=5, sr=30)
#     v = np.median(invariantImage)
#     # ---- apply automatic Canny edge detection using the computed median----
#     lower = 255 / 3  # int(max(0, (1.0 - sigma) * v)) #255/3
#     upper = int(min(255, (1.0 + sigma) * v))  # 255
#     print(lower, upper)
#     edges_invariant = cv.Canny(invariantImage, threshold1=lower, threshold2=upper, apertureSize=3)
#
#     kernel = np.ones((3, 3), np.uint8)
#     edges_invariant_dil = cv.dilate(edges_invariant, kernel, iterations=1)
#     edges_dil = cv.dilate(edges, kernel, iterations=1)
#
#     # f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, sharey=False)
#     # ax1.imshow(gray, cmap="gray")
#     # ax2.imshow(edges)
#     # ax3.imshow(invariantImage, cmap="gray")
#     # ax4.imshow(edges_invariant)
#     # plt.show()
#
#     shadow_edge_map = np.subtract(edges_dil, edges_invariant_dil)
#
#     f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, sharey=False)
#     ax1.imshow(edges_dil)
#     ax1.set_title("edges_dil")
#     ax2.imshow(edges_invariant_dil)
#     ax2.set_title("edges_invariant_dil")
#     ax3.imshow(np.bitwise_and(edges_dil, edges_invariant_dil))
#     ax3.set_title("mask")
#     ax4.imshow(shadow_edge_map)
#     ax4.set_title("diff")
#     plt.show()
#
#     # shadow_edge_map = (shadow_edge_map > 128) * 255
#     kernel = np.ones((3, 3), np.uint8)
#     return cv.dilate(shadow_edge_map, kernel, iterations=1)

#def calculate3DChromaticitiesFromImage(image):
#     image = image + 1
#     r = image[:, :, 0] / 255
#     g = image[:, :, 1] / 255
#     b = image[:, :, 2] / 255
#
#     r[r == 0] = 1
#     g[g == 0] = 1
#     b[b == 0] = 1
#
#     # DIVIDE BY GEOMETRIC MEAN - 3D chromaticity
#     geometric_mean = np.multiply(np.multiply(r, g), b) ** (1 / 3)
#     chromaticity_r = np.log(np.divide(r, geometric_mean))
#     chromaticity_g = np.log(np.divide(g, geometric_mean))
#     chromaticity_b = np.log(np.divide(b, geometric_mean))
#
#     # C chormaticity to R chromaticity
#     # chromaticity_sum = np.add(np.add(chromaticity_r, chromaticity_g), chromaticity_b)
#     # chromaticity_sum[chromaticity_sum == 0] = 0.00001
#     # chromaticity_r = np.divide(chromaticity_r, chromaticity_sum)
#     # chromaticity_g = np.divide(chromaticity_g, chromaticity_sum)
#     # chromaticity_b = np.divide(chromaticity_b, chromaticity_sum)
#
#     # AVOID ZEROES (temporary)
#     chromaticity_r[chromaticity_r == 0] = 0.00001
#     chromaticity_g[chromaticity_g == 0] = 0.00001
#     chromaticity_b[chromaticity_b == 0] = 0.00001
#
#     return np.stack((chromaticity_r, chromaticity_g, chromaticity_b), axis=2)  # log chromaticity