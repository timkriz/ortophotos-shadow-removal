import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, show
import cv2 as cv
from skimage import filters
import math
from numpy import ndarray
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import scipy
import scipy.fftpack
from scipy.sparse import diags

TRIMMED_PERCENT = 1


def main():
    script_dir = os.path.dirname(__file__)
    rel_path = "images/copy3.png"
    abs_file_path = os.path.join(script_dir, rel_path)
    image = imread(abs_file_path)
    # chromaticityGraph(image)
    shadowRemoval(image)


def shadowRemoval(image):
    # image = filters.gaussian(image, sigma=1, multichannel=True, preserve_range=True)  # Gauss filter
    angle = getBestAngle(image)
    invariantImage, chromaticities = getInvariantImage(image, angle)

    flattened = trimmed_percentiles(invariantImage.flatten(), TRIMMED_PERCENT)
    #invariantImage = np.interp(invariantImage, [flattened[0], flattened[-1]], [0, 1])
    #invariantImage = np.uint8(invariantImage)
    print("invariantImage")
    print(invariantImage)




    ddepth = cv.CV_64F
    x_der = np.array([[0, 0, 0],
                        [0, -1 , 1],
                        [0, 0, 0]])
    y_der = np.array([[0, 0, 0],
                      [0, -1, 0],
                      [0, 1, 0]])

    # INVARIANT
    sobelx = cv.filter2D(src=invariantImage, ddepth=ddepth, kernel=x_der)
    sobely = cv.filter2D(src=invariantImage, ddepth=ddepth, kernel=y_der)
    [magnitude, angle] = imgradient(sobelx, sobely)


    # Red
    imgR = image[:, :, 0]
    rgX = cv.filter2D(src=imgR, ddepth= ddepth, kernel=x_der)
    rgY = cv.filter2D(src=imgR, ddepth= ddepth, kernel=y_der)
    [Rmagnitude, Rangle] = imgradient(rgX, rgY)

    #Green
    imgG = image[:, : , 1]
    ggX = cv.filter2D(src=imgG, ddepth= ddepth, kernel=x_der)
    ggY = cv.filter2D(src=imgG, ddepth= ddepth, kernel=y_der)
    [Gmagnitude, Gangle] = imgradient(ggX, ggY)

    # Blue
    imgB = image[:, :, 2]
    bgX = cv.filter2D(src=imgB, ddepth= ddepth, kernel=x_der)
    bgY = cv.filter2D(src=imgB, ddepth= ddepth, kernel=y_der)
    [Bmagnitude, Bangle] = imgradient(bgX, bgY)


    # Threshold
    redX2 = np.copy(rgX)
    redY2 = np.copy(rgY)
    redX2[(Rmagnitude > 0) & (magnitude < 0.005)] = 0
    redY2[(Rmagnitude > 0) & (magnitude < 0.005)] = 0

    greenX2 = np.copy(ggX)
    greenY2 = np.copy(ggY)
    greenX2[(Gmagnitude > 0) & (magnitude < 0.005)] = 0
    greenY2[(Gmagnitude > 0) & (magnitude < 0.005)] = 0

    blueX2 = np.copy(bgX)
    blueY2 = np.copy(bgY)
    blueX2[(Bmagnitude > 0) & (magnitude < 0.005)] = 0
    blueY2[(Bmagnitude > 0) & (magnitude < 0.005)] = 0


    kxx = np.array([[0, 0, 0],
                      [-1, 1, 0],
                      [0, 0, 0]])
    kyy = np.array([[0, -1, 0],
                    [0, 1, 0],
                    [0, 0, 0]])

    redXX = cv.filter2D(src=redX2, ddepth=-1, kernel=kxx)
    redYY = cv.filter2D(src=redY2, ddepth=-1, kernel=kyy)

    print("invariant magnitude")
    print(magnitude)
    print("redXX")
    print(redXX)

    greenXX = cv.filter2D(src=greenX2, ddepth=-1, kernel=kxx)
    greenYY = cv.filter2D(src=greenY2, ddepth=-1, kernel=kyy)

    blueXX = cv.filter2D(src=blueX2, ddepth=-1, kernel=kxx)
    blueYY = cv.filter2D(src=blueY2, ddepth=-1, kernel=kyy)

    laplacR = redXX + redYY;
    laplacG = greenXX + greenYY;
    laplacB = blueXX + blueYY;
    [width, height] = np.shape(laplacR)

    print("np.shape(laplacR)")
    print(np.shape(laplacR))



    r2 = matrixSolving(laplacR, width, height)
    r = r2#matlab_mat2grey(r2)
    g = 0#matlab_mat2grey(matrixSolving(laplacG, width, height))
    b = 0#matlab_mat2grey(matrixSolving(laplacB, width, height))

    print("r2")
    print(r2)
    f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, sharey=False)
    ax1.imshow(invariantImage, cmap='gray')
    ax1.set_title("invariantImage")
    ax2.imshow(magnitude, cmap='gray')
    ax2.set_title("magnitude")
    ax3.imshow(laplacR, cmap='gray')
    ax3.set_title("laplacR")
    ax4.imshow(r2, cmap='gray')
    ax4.set_title("r2")
    plt.show();
    r = r.astype('uint8')
    g = g.astype('uint8')
    b = b.astype('uint8')

    f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, sharey=False)
    ax1.imshow(r, cmap='gray')
    ax1.set_title("r")
    ax2.imshow(g, cmap='gray')
    ax2.set_title("g")
    ax3.imshow(b, cmap='gray')
    ax3.set_title("b")
    ax4.imshow(r2, cmap='gray')
    ax4.set_title("r2")
    plt.show()


    [redMean, greenMean, blueMean] = findMeanOfMax(r, g, b);
    [redMeanMIN, greenMeanMIN, blueMeanMIN] = findMeanOfMin(r, g, b);
    # r[r > redMean] = 255
    # g[g > greenMean] = 255
    # b[b > blueMean] = 255
    print("redMean", "greenMean", "blueMean")
    print(redMean, greenMean, blueMean)
    print("np.amin(r)", "redMean")
    print(np.amin(r), redMean)

    r = np.interp(r, [redMeanMIN, redMean], [0, 255]).astype('uint8')
    g = np.interp(g, [greenMeanMIN, greenMean], [0, 255]).astype('uint8')
    b = np.interp(b, [blueMeanMIN, blueMean], [0, 255]).astype('uint8')


    rgbImage = np.stack((r, g, b), axis=2)
    imshow(rgbImage)
    show()

    f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, sharey=False)
    ax1.imshow(r, cmap='gray')
    ax1.set_title("r")
    ax2.imshow(g, cmap='gray')
    ax2.set_title("g")
    ax3.imshow(b, cmap='gray')
    ax3.set_title("b")
    ax4.imshow(rgbImage, cmap='gray')
    ax4.set_title("rgbImage")
    plt.show()


    #plotHistogramAndVarianceFromInvrariantImage(invariantImage, chromaticities, angle)
    # plotInvariantImage(invariantImage)

    #shadow_edge_map = shadowEdgeDetection(image, invariantImage)
    #image_recovery(image, invariantImage)

    pass

def findMeanOfMax(red, green, blue):
    redTop = np.percentile(red, 80)
    greenTop = np.percentile(green, 80)
    blueTop = np.percentile(blue, 80)

    print("redTop","greenTop", "blueTop")
    print(redTop,greenTop, blueTop)
    print(red[red > redTop])

    redMean = np.mean(red[red > redTop])
    greenMean = np.mean(green[green > greenTop])
    blueMean = np.mean(blue[blue > blueTop])

    return [redMean, greenMean, blueMean]

def findMeanOfMin(red, green, blue):
    redTop = np.percentile(red, 10)
    greenTop = np.percentile(green, 10)
    blueTop = np.percentile(blue, 10)

    print("redbottom","gb", "bb")
    print(redTop,greenTop, blueTop)
    print(red[red < redTop])

    redMean = np.mean(red[red < redTop])
    greenMean = np.mean(green[green < greenTop])
    blueMean = np.mean(blue[blue < blueTop])

    return [redMean, greenMean, blueMean]

# python function replica of matlab's mat2gray
def matlab_mat2grey(A):
    normalized = cv.normalize(A, None, 0, 255, norm_type=cv.NORM_MINMAX)
    return normalized.astype(np.uint8)

def matrixSolving (fun, width, height):
    # Grid parameters.
    nx = width
    ny = height
    print("nx: ", nx, "ny: ", ny)
    xmin, xmax = 0.0, 1.0  # limits in the x direction
    ymin, ymax = 0.0, 1.0  # limits in the y direction
    lx = xmax - xmin  # domain length in the x direction
    ly = ymax - ymin  # domain length in the y direction
    dx = lx / (nx - 1)  # grid spacing in the x direction
    dy = ly / (ny - 1)  # grid spacing in the y direction

    b = np.copy(fun)
    bflat = b[1:-1, 1:-1].flatten('F')
    p = np.empty((nx, ny))

    A = d2_mat_dirichlet_2d(nx, ny, dx, dy)
    print(A)
    Ainv = np.linalg.inv(A)

    pvec = np.reshape(np.dot(Ainv, bflat), (nx - 2, ny - 2), order='F')
    p[1:-1, 1:-1] = pvec













    # p0 = np.zeros((nx, ny))
    # pnew = p0.copy()
    # p = np.empty((nx, ny))
    # b = fun
    # for k in range(1, 10000):
    #     np.copyto(p, pnew)
    #     pnew[1:-1, 1:-1] = (0.25 * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2]
    #                             + p[1:-1, 2:] - b[1:-1, 1:-1] * dx ** 2))
    # u = np.copy(jacobi(u, b))


    # identity = np.identity(ny) * -4
    # diagonal1 = np.diag(np.ones((ny - 1)), 1)
    # diagonal2 = np.diag(np.ones((ny - 1)), -1)
    # a_off = np.identity(ny)
    # diag_mat = identity + diagonal1 + diagonal2
    #
    # A = csr_matrix((ny*nx,ny*nx), dtype = np.double)
    # ny=ny+1
    # for i in range(0, nx):
    #     A[i*ny+1 : i*ny+ny , i*ny+1:i*ny+ny] = diag_mat
    # # u = np.copy(jacobi(u, b))
    # #A((i - 1) * (ny) + 1: (i - 1) * (ny) + (ny), (i - 1) * (ny) + 1: (i - 1) * (ny) + (ny)) = diag_mat;
    #
    # for i in range(1, nx-1):
    #     A[(i-1)*ny+1 : (i-1)*ny+ny , i*ny+1:i*ny+ny] = a_off
    #     A[i*ny+1 : i*ny+ny , (i-1)*ny+1:(i-1)*ny+ny] = a_off
    #
    # #A((i-2)*(ny)+1:(i-2)*(ny) + (ny),(i-1)*(ny)+1:(i-1)*(ny) + (ny)) = a_off;
    # #A((i-1)*(ny)+1:(i-1)*(ny) + (ny),(i-2)*(ny)+1:(i-2)*(ny) + (ny)) = a_off;
    #
    # m = np.copy(fun)
    # m.T.ravel()
    # print(m)
    # print(m)
    #
    # u = np.zeros((nx+2, ny+2))
    # b = np.zeros((nx+2, ny+2))
    # b[1:-1, 1:-1] = fun
    # for k in range(1, 100):
    #     u = np.copy(jacobi(u, b))

    # u = np.zeros((nx + 2, ny + 2))
    # or k in range(1,5):f
    #     u_k_p_1 = -(dx**2) / 4 * b
    #
    #     for i in range(1, nx):
    #         for j in range(1, ny):
    #             u_k_p_1[i,j] =  u_k_p_1[i,j] + (u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1] - 4 * u[i, j]) / 4
    #             #b[i, j] = (u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1] - 4 * u[i, j]) / dx ** 2
    #     u = np.copy(u_k_p_1)

    # p0 = np.zeros((nx, ny))
    # pnew = p0.copy()
    # fun = fun.astype('float32')
    # #print(fun.astype('float32'))
    # currentLoopNum = 0
    # p = np.empty((nx, ny))
    # while (currentLoopNum < 4):
    #     np.copyto(p, pnew)
    #     pnew[1:-1, 1:-1] = (0.25 * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - fun[1:-1, 1:-1] * dx ** 2))
    #     currentLoopNum+=1
    #
    # f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, sharey=False)
    # ax1.imshow(u, cmap='gray')
    # ax1.set_title("pnew")
    # plt.show()
    #
    #
    #     print("iteration" + str(currentLoopNum))

    # identity = np.identity(ny) * -4
    # diagonal1 = np.diag(np.ones((ny - 1)), 1)
    # diagonal2 = np.diag(np.ones((ny - 1)), -1)
    # identity = identity + diagonal1 + diagonal2
    #print(identity)

    a_off = np.identity(ny)

    return p

def d2_mat_dirichlet_2d(nx, ny, dx, dy):
    """
    Constructs the matrix for the centered second-order accurate
    second-order derivative for Dirichlet boundary conditions in 2D

    Parameters
    ----------
    nx : integer
        number of grid points in the x direction
    ny : integer
        number of grid points in the y direction
    dx : float
        grid spacing in the x direction
    dy : float
        grid spacing in the y direction

    Returns
    -------
    d2mat : numpy.ndarray
        matrix to compute the centered second-order accurate first-order deri-
        vative with Dirichlet boundary conditions
    """
    a = 1.0 / dx**2
    g = 1.0 / dy**2
    c = -2.0*a - 2.0*g

    diag_a = a * np.ones((nx-2)*(ny-2)-1)
    diag_a[nx-3::nx-2] = 0.0
    diag_g = g * np.ones((nx-2)*(ny-3))
    diag_c = c * np.ones((nx-2)*(ny-2))

    # We construct a sequence of main diagonal elements,
    diagonals = [diag_g, diag_a, diag_c, diag_a, diag_g]
    # and a sequence of positions of the diagonal entries relative to the main
    # diagonal.
    offsets = [-(nx-2), -1, 0, 1, nx-2]

    # Call to the diags routine; note that diags return a representation of the
    # array; to explicitly obtain its ndarray realisation, the call to .toarray()
    # is needed. Note how the matrix has dimensions (nx-2)*(nx-2).
    d2mat = diags(diagonals, offsets).toarray()

    # Return the final array
    return d2mat

def jacobi(xk, b):
    nx = np.shape(xk)[0]-1
    ny = np.shape(xk)[1]-1
    dx = 1 / nx
    dy = 1 / ny

    xkp1 = np.copy(xk)
    for i in range(1, nx):
        for j in range(1, ny):
            xkp1[i,j] = (b[i,j] - ((xk[i+1,j] + xk[i-1, j]) / dx**2) - ((xk[i, j+1] + xk[i, j-1]) / dy**2)) / (-2 / dx**2 - 2/dy**2)

    return xkp1


def getBestAngle(image):
    entropy = []
    iPotential = []
    numOfAngles = 181
    angles = []
    step = 1

    # BEST: rtv.png : approx. 150, grass2.png: 100, ball1: 155
    for angle in range(155, 156, step):
        angles.append(angle)
        invariantImage, chromaticities = getInvariantImage(image, angle)
        flattened = invariantImage.flatten()
        flattened = trimmed_percentiles(flattened, TRIMMED_PERCENT)
        # print("len:" + str(np.shape(flattened)) + " MIN: " + str(np.amin(flattened)) + "MAX: " + str(np.amax(flattened)))
        # print("shared_bins: ", shared_bins)

        hist, binedges = np.histogram(flattened, bins='scott', density=True)  # Calculate histogram

        # meanValue = np.mean(invariantImageFlat)
        # std = np.std(invariantImageFlat)
        # for i in hist[0]:
        #     ent -= i * math.log(abs(i))
        hist[hist == 0] = 0.0000000001  # Temporary

        hist = hist / sum(hist)
        ent = -1 * np.sum(np.multiply(hist, np.log2(hist)))  # Entropy
        entropy.append(ent)

        # Information potential
        iPotential.append(getIPotential(chromaticities[0], chromaticities[1]))

        # print(angle, " entropy: ", ent)
        # plotHistogramAndVarianceFromInvrariantImage(invariantImage, chromaticities, angle)

    bestAngle = angles[entropy.index(min(entropy))]
    # bestAngle = angles[iPotential.index(max(iPotential))]
    print("BEST angle: ", bestAngle, "  entropy: ", min(entropy))
    # plt.figure()
    # plt.title("Entropies")
    # plt.plot(entropy)
    # plt.show()
    return bestAngle


def getIPotential(X, Y):
    N = np.shape(X)[0] * np.shape(X)[1]
    s = 1.
    print(np.shape(Y))
    return 0


def shadowEdgeDetection(image, invariantImage):
    img = cv.pyrMeanShiftFiltering(src=image, sp=9,
                                   sr=50)  # plotInvariantImage(img) # sp – The spatial window radius., sr – The color window radius.
    sigma = 0.33
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # gray = cv.GaussianBlur(gray, (3, 3), 0.2)
    v = np.median(gray)
    # ---- apply automatic Canny edge detection using the computed median----
    lower = int(max(0, (1.0 - sigma) * v))  # 255/3
    upper = int(min(255, (1.0 + sigma) * v))  # 255
    print(lower, upper)
    print(gray)
    edges = cv.Canny(gray, threshold1=lower, threshold2=upper)

    flattened = trimmed_percentiles(invariantImage.flatten(), TRIMMED_PERCENT)
    invariantImage = np.interp(invariantImage, [flattened[0], flattened[-1]], [0, 255])
    invariantImage = np.uint8(invariantImage)
    # invariantImage = cv.GaussianBlur(invariantImage, (5, 5), 1.4)
    # invariantImage = cv.pyrMeanShiftFiltering(src=np.stack((invariantImage, invariantImage, invariantImage), axis=2), sp=5, sr=30)
    v = np.median(invariantImage)
    # ---- apply automatic Canny edge detection using the computed median----
    lower = 255 / 3  # int(max(0, (1.0 - sigma) * v)) #255/3
    upper = int(min(255, (1.0 + sigma) * v))  # 255
    print(lower, upper)
    edges_invariant = cv.Canny(invariantImage, threshold1=lower, threshold2=upper, apertureSize=3)

    kernel = np.ones((3, 3), np.uint8)
    edges_invariant_dil = cv.dilate(edges_invariant, kernel, iterations=1)
    edges_dil = cv.dilate(edges, kernel, iterations=1)

    # f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, sharey=False)
    # ax1.imshow(gray, cmap="gray")
    # ax2.imshow(edges)
    # ax3.imshow(invariantImage, cmap="gray")
    # ax4.imshow(edges_invariant)
    # plt.show()

    shadow_edge_map = np.subtract(edges_dil, edges_invariant_dil)

    f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, sharey=False)
    ax1.imshow(edges_dil)
    ax1.set_title("edges_dil")
    ax2.imshow(edges_invariant_dil)
    ax2.set_title("edges_invariant_dil")
    ax3.imshow(np.bitwise_and(edges_dil, edges_invariant_dil))
    ax3.set_title("mask")
    ax4.imshow(shadow_edge_map)
    ax4.set_title("diff")
    plt.show()

    # shadow_edge_map = (shadow_edge_map > 128) * 255
    kernel = np.ones((3, 3), np.uint8)
    return cv.dilate(shadow_edge_map, kernel, iterations=1)


def image_recovery(image, shadow_edge_map):
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = image[:, :, 0]  # RED
    #laplacian = cv.Laplacian(gray, cv.CV_64F)

    # ANOTHER GRADIENT
    [grad_x, grad_y] = get_grads(gray)
    grad_x[(shadow_edge_map == 255)] = 0
    grad_y[(shadow_edge_map == 255)] = 0

    #print(laplacian)
    #S = laplacian
    #S[(shadow_edge_map == 255)] = 0
    # S = ((shadow_edge_map == 255) & (laplacian > 5)) * laplacian
    print("grad_x")
    print(grad_x)
    f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, sharey=False)
    ax1.imshow(gray)
    ax1.set_title("orig")
    ax2.imshow(grad_x, cmap='gray')
    ax2.set_title("laplacian")
    ax3.imshow(shadow_edge_map)
    ax3.set_title("shadow_edge_map")
    ax4.imshow(grad_y, cmap='gray')
    ax4.set_title("S")
    plt.show()

    # laplacian2 = cv.Laplacian(S, cv.CV_64F)

    integrate(gray, [grad_x, grad_y])

    return


def integrate(img, gradient):
    #[H, W] = np.shape(gradient)
    #gx = np.zeros((H, W))
    #gy = np.zeros((H, W))
    #gx[1:(H-1), 1:(W-1)] = gradient[1:(H-1), 2:W] - gradient[1:(H-1), 1:(W-1)]
    #gy[1:(H-1), 1:(W-1)] = gradient[2:H, 1:(W-1)] - gradient[1:(H-1), 1:(W-1)]
    #Find gradients
    # gx = np.zeros(H, W)
    # gy = np.zeros(H, W)
    # j = 1:(H - 1)
    # k = 1:W - 1


    ddepth = cv.CV_16S
    #grad_x = get_grads()#cv.Sobel(img, ddepth, dx=1, dy=0, ksize=3)
    #grad_y = get_grads#cv.Sobel(img, ddepth, dx=0, dy=1, ksize=3)

    [grad_x, grad_y] = gradient# get_grads(img)
    #grad_grad_x = cv.Sobel(gradient, ddepth, 1, 0, ksize=3, borderType=cv.BORDER_DEFAULT)
    #grad_grad_y = cv.Sobel(gradient, ddepth, 0, 1, ksize=3, borderType=cv.BORDER_DEFAULT)

    # f_x = np.fft.fft2(grad_x)
    # fs_x = np.fft.fftshift(f_x)
    # magnitude_spectrum_f_x = 20 * np.log(np.abs(fs_x))
    # f_y = np.fft.fft2(grad_y)
    # fs_y = np.fft.fftshift(f_y)
    # magnitude_spectrum_f_y = 20 * np.log(np.abs(fs_y))

    f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, sharey=False)
    ax1.imshow(grad_x, cmap='gray')
    ax1.set_title("orig grad_x")
    ax2.imshow(grad_y, cmap='gray')
    ax2.set_title("orig grad_y")
    ax3.imshow(grad_x, cmap='gray')
    ax3.set_title("gradient grad_x")
    ax4.imshow(grad_y, cmap='gray')
    ax4.set_title("gradient grad_y")
    plt.show()

    img_rec = poisson_solve(grad_x, grad_y, img);
    #img_rec_interp = np.clip(img_rec, 0, 255)
    #img_rec_interp = np.interp(img_rec, [np.amin(img_rec), np.amax(img_rec)], [0, 255])
    #img_rec_interp = img_rec_interp.astype('uint8')

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    ax1.imshow(img, cmap='gray')
    ax1.set_title("img grad_x")
    ax2.imshow(img_rec, cmap='gray')
    ax2.set_title("img_rec grad_x")
    plt.show()
    return


def trimmed_percentiles(data, percent):
    data = np.sort(data)
    if percent == 0:
        return data
    else:
        trim = int(percent * np.shape(data)[0] / 100.0)
    return data[trim:-trim]

def clip_image(img):
    flattened = img.flatten()
    flattened = trimmed_percentiles(flattened, TRIMMED_PERCENT)
    return np.clip(img, flattened[0], flattened[-1])

def plotHistogramAndVarianceFromInvrariantImage(invariantImage, chromaticities, rho):
    flattened = invariantImage.flatten()
    flattened = trimmed_percentiles(flattened, TRIMMED_PERCENT)

    figure, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, sharey=False)
    figure.suptitle("Angle: " + str(rho))

    # Chromaticities
    ax1.plot(chromaticities[0], chromaticities[1], 'o', markersize=0.4)
    if rho == 90:
        ax1.vlines(0, ymin=-1, ymax=1)
    else:
        m1, b1 = math.tan(math.radians(rho)), 0.0  # slope & intercept
        x = np.linspace(-1e16, 1e16, 500)
        ax1.plot(x, x * m1 + b1)

    # Histogram
    hist, binedges = np.histogram(flattened, bins='scott')
    ax2.bar(binedges[:-1], hist, width=binedges[1] - binedges[0])

    # Variance
    ax3.plot(flattened, len(flattened) * [1], "x", markersize="1")

    # Image
    ax4.imshow(invariantImage, vmin=flattened[0], vmax=flattened[-1])

    plt.show()


def plotInvariantImage(invariantImage):
    # plt.figure()
    # plt.imshow(invariantImage, cmap=plt.get_cmap('gray'),
    #            vmin=0, vmax=1)
    # plt.show()
    imshow(invariantImage)
    show()


def getInvariantImage(image, rho):
    #image = cv.pyrMeanShiftFiltering(src=image, sp=9, sr=50)
    chromaticities2D = caluclate2DChromaticitiesFromImage(image)
    cosine = np.cos(0.7)#np.radians(rho))
    sine = np.sin(0.7)#np.radians(rho))

    # (ψ1 cos θ + ψ2 sin θ)
    firstCos = np.multiply(chromaticities2D[:, :, 0], cosine)
    secondSin = np.multiply(chromaticities2D[:, :, 1], sine)
    intrinsicImage = np.add(firstCos, secondSin)

    return intrinsicImage, (chromaticities2D[:, :, 0], chromaticities2D[:, :, 1])


def caluclate2DChromaticitiesFromImage(image):
    log_chromaticies_3D = calculate3DChromaticitiesFromImage(image)
    U = [[1 / math.sqrt(2), -1 / math.sqrt(2), 0], [1 / math.sqrt(6), 1 / math.sqrt(6), -2 / math.sqrt(6)]]
    U = np.array(U)
    X = np.dot(log_chromaticies_3D, U.T)
    # plot2DChromaticity(X[:,:,0], X[:,:,1])
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
    else:
        m1, b1 = math.tan(math.radians(rho)), 0.0  # slope & intercept (line 1)
        # print("m1: ", m1)
        x = np.linspace(-1, 1, 500)
        plt.plot(x, x * m1 + b1)

    plt.show()
    pass


def chromaticityGraph(image):
    rowsNum, columnNum, colorsNum = np.shape(image)
    img_b_g = np.empty(rowsNum * columnNum)
    img_r_g = np.empty(rowsNum * columnNum)
    pixels = image.flatten().reshape(rowsNum * columnNum, 3)

    for x in range(np.shape(pixels)[0]):
        # sum = pixels[x, 0] + pixels[x, 1] + pixels[x, 2]
        r = pixels[x, 0]
        g = pixels[x, 1]
        b = pixels[x, 2]

        b_g = b / g
        r_g = r / g
        if b_g == 0 or r_g == 0:
            print('ZERO')
            continue
        img_b_g[x] = math.log(b_g)
        img_r_g[x] = math.log(r_g)

    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.plot(img_r_g, img_b_g, 'o', markersize=0.5)
    # plt.xlim([-0.4, 1.4])
    # plt.ylim([-1.6, 0])
    plt.ylabel('Log(b/g)')
    plt.xlabel('Log(r/g)')
    # plt.xscale('log')
    # plt.yscale('log')
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
    r = image[:, :, 0].astype('double') /255
    g = image[:, :, 1].astype('double') /255
    b = image[:, :, 2].astype('double') /255
    r[r == 0] = 1
    g[g == 0] = 1
    b[b == 0] = 1
    # DIVIDE BY GEOMETRIC MEAN - 3D chromaticity
    geometric_mean = np.multiply(np.multiply(r, g), b) ** (1 / 3)
    #geometric_mean[geometric_mean == 0] = 0
    chromaticity_r = np.log(np.divide(r, geometric_mean))
    chromaticity_g = np.log(np.divide(g, geometric_mean))
    chromaticity_b = np.log(np.divide(b, geometric_mean))
    # print("chromaticity_r")
    # print(chromaticity_r)
    # print("chromaticity_g")
    # print(chromaticity_g)
    # print("chromaticity_b")
    # print(chromaticity_b)

    # C chormaticity to R chromaticity
    # chromaticity_sum = np.add(np.add(chromaticity_r, chromaticity_g), chromaticity_b)
    # chromaticity_sum[chromaticity_sum == 0] = 0.00001
    # chromaticity_r = np.divide(chromaticity_r, chromaticity_sum)
    # chromaticity_g = np.divide(chromaticity_g, chromaticity_sum)
    # chromaticity_b = np.divide(chromaticity_b, chromaticity_sum)

    # AVOID ZEROES (temporary)
    chromaticity_r[chromaticity_r == 0] = 0.00001
    chromaticity_g[chromaticity_g == 0] = 0.00001
    chromaticity_b[chromaticity_b == 0] = 0.00001

    return np.stack((chromaticity_r, chromaticity_g, chromaticity_b), axis=2)  # log chromaticity


def rgb_splitter(image):
    rgb_list = ['Reds', 'Greens', 'Blues']
    fig, ax = plt.subplots(1, 3, figsize=(17, 7), sharey=False)
    for i in range(3):
        ax[i].imshow(image[:, :, i], cmap=rgb_list[i])
        ax[i].set_title(rgb_list[i], fontsize=22)
        ax[i].axis('off')
    fig.tight_layout()
    show()


def poisson_reconstruct(grady, gradx, boundarysrc):
    # Thanks to Dr. Ramesh Raskar for providing the original matlab code from which this is derived
    # Dr. Raskar's version is available here: http://web.media.mit.edu/~raskar/photo/code.pdf

    # Laplacian
    gyy = grady[1:, :-1] - grady[:-1, :-1]
    gxx = gradx[:-1, 1:] - gradx[:-1, :-1]
    f = np.zeros(boundarysrc.shape)
    f[:-1, 1:] += gxx
    f[1:, :-1] += gyy

    # Boundary image
    boundary = boundarysrc.copy()
    boundary[1:-1, 1:-1] = 0

    # Subtract boundary contribution
    f_bp = -4 * boundary[1:-1, 1:-1] + boundary[1:-1, 2:] + boundary[1:-1, 0:-2] + boundary[2:, 1:-1] + boundary[0:-2,
                                                                                                        1:-1]
    f = f[1:-1, 1:-1] - f_bp

    # Discrete Sine Transform
    tt = scipy.fftpack.dst(f, norm='ortho')
    fsin = scipy.fftpack.dst(tt.T, norm='ortho').T

    # Eigenvalues
    (x, y) = np.meshgrid(range(1, f.shape[1] + 1), range(1, f.shape[0] + 1), copy=True)
    denom = (2 * np.cos(math.pi * x / (f.shape[1] + 2)) - 2) + (2 * np.cos(math.pi * y / (f.shape[0] + 2)) - 2)

    f = fsin / denom

    # Inverse Discrete Sine Transform
    tt = scipy.fftpack.idst(f, norm='ortho')
    img_tt = scipy.fftpack.idst(tt.T, norm='ortho').T

    # New center + old boundary
    result = boundary
    result[1:-1, 1:-1] = img_tt

    return result

def get_grads(im):
    """
    return the x and y gradients.
    """
    [H,W] = im.shape
    Dx,Dy = np.zeros((H,W),'float32'), np.zeros((H,W),'float32')
    j,k = np.atleast_2d(np.arange(0,H-1)).T, np.arange(0,W-1)
    Dx[j,k] = im[j,k+1] - im[j,k]
    Dy[j,k] = im[j+1,k] - im[j,k]
    return Dx,Dy

def get_laplacian(Dx,Dy):
    """
    return the laplacian
    """
    [H,W] = Dx.shape
    Dxx, Dyy = np.zeros((H,W)), np.zeros((H,W))
    j,k = np.atleast_2d(np.arange(0,H-1)).T, np.arange(0,W-1)
    Dxx[j,k+1] = Dx[j,k+1] - Dx[j,k]
    Dyy[j+1,k] = Dy[j+1,k] - Dy[j,k]
    return Dxx+Dyy


def poisson_solve(gx, gy, bnd):
    # convert to double:
    gx = gx.astype('float32')
    gy = gy.astype('float32')
    bnd = bnd.astype('float32')

    H, W = bnd.shape
    L = get_laplacian(gx, gy)
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)

    ax1.imshow(L, cmap='gray')
    ax1.set_title("laplacian 1")


    # set the interior of the boundary-image to 0:
    bnd[1:-1, 1:-1] = 0
    # get the boundary laplacian:
    L_bp = np.zeros_like(L)
    L_bp[1:-1, 1:-1] = -4 * bnd[1:-1, 1:-1] \
                       + bnd[1:-1, 2:] + bnd[1:-1, 0:-2] \
                       + bnd[2:, 1:-1] + bnd[0:-2, 1:-1]  # delta-x
    L = L - L_bp
    L = L[1:-1, 1:-1]
    ax2.imshow(L, cmap='gray')
    ax2.set_title("laplacian 2")
    plt.show()

    # compute the 2D DST:
    L_dst = DST(DST(L).T).T  # first along columns, then along rows

    # normalize:
    [xx, yy] = np.meshgrid(np.arange(1, W - 1), np.arange(1, H - 1))
    D = (2 * np.cos(np.pi * xx / (W - 1)) - 2) + (2 * np.cos(np.pi * yy / (H - 1)) - 2)
    L_dst = L_dst / D

    img_interior = IDST(IDST(L_dst).T).T  # inverse DST for rows and columns

    img = bnd.copy()

    img[1:-1, 1:-1] = img_interior

    return img

def DST(x):
    """
    Converts Scipy's DST output to Matlab's DST (scaling).
    """
    X = scipy.fftpack.dst(x,type=1,axis=0)
    return X/2.0

def IDST(X):
    """
    Inverse DST. Python -> Matlab
    """
    n = X.shape[0]
    x = np.real(scipy.fftpack.idst(X,type=1,axis=0))
    return x/(n+1.0)

def imgradient(gX, gY) :
    magnitude = np.sqrt(gX ** 2.0 + gY ** 2.0)
    angle = np.arctan2(gY, gX) * (180 / np.pi)
    return [magnitude, angle]
if __name__ == "__main__":
    main()


