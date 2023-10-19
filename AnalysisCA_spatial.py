import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shapely
import math
from datetime import datetime
import warnings
import logging
import json

#I'm pretty sure this does not work as it's completely supposed to:
#The limits (right now) should be from [0 - some value], and does not work for e.g. [200 - 500]
def intersection_imageedge(a, b, limits):
    '''
    For a given image dimension (limits) and linear line (y=ax+b) it determines which borders of the image the line
    intersects and the length (in pixels) of the slice.
    :param a: slope of the linear line (y=ax+b)
    :param b: intersect with the y-axis of the linear line (y=ax+b)
    :param limits: image dimensions (x0, width, y0, height)
    :return: absolute length of slice (l) and border intersects booleans (top, bot, left, right).

    Note: line MUST intersects the image edges, otherwise it throws an error.
    '''
    # define booleans for the 4 image limit intersections.
    top, bot, left, right = False, False, False, False
    w, h = limits[1], limits[3]   # w = width, h = height

    '''
    Determining the intersections (assuming x,y=0,0 is bottom right).
    Bottom: for y=0: 0<=x<=w --> y=0, 0<=(y-b)/a<=w --> 0<=-b/a<=w
    Top: for y=h: 0<=x<=w --> y=h, 0<=(y-b)/a<=w --> 0<=(y-h)/a<=w
    Left: for x=0: 0<=y<=h --> x=0, 0<=ax+b<=h --> 0<=b<=h
    Right: for x=w: 0<=y<=h --> x=0, 0<=ax+b<=h --> 0<=a*w+b<=h
    '''
    if -b / a >= 0 and -b / a <= w:
        bot = True
    if (h - b) / a >= 0 and (h - b) / a <= w:
        top = True
    if b >= 0 and b <= h:
        left = True
    if (a * w) + b >= 0 and (a * w) + b <= h:
        right = True

    if top + bot + left + right != 2:  # we must always have 2 intersects for a linear line
        logging.error("The profile should always intersect 2 image edges.")
        exit()

    '''
    Determining the slice length.
    There are 6 possible intersections of the linear line with a rectangle.
    '''
    if top & bot:  # case 1
        # l = imageheight / sin(theta), where theta = atan(a)
        # == > l = (sqrt(a ^ 2 + 1) * imageheight) / a
        l = np.round((np.sqrt(a ** 2 + 1) * h) / a)  # l = profile length (density=1px)
    if left & right:  # case 2
        # l = imagewidth / cos(theta), where theta=atan(a)
        # ==> l = sqrt(a^2+1)*imagewidth
        l = np.round(np.sqrt(a ** 2 + 1) * w)  # l = profile length (density=1px)
    if left & top:  # case 3
        # dx = x(y=h), dy = h-y(x=0) --> l = sqrt(dx^2+dy^2)
        dx = (h - b) / a
        dy = h - b
        l = np.sqrt(dx**2 + dy**2)
    if top & right:  # case 4
        # dx = w-x(y=h), dy = h-y(x=0) --> l = sqrt(dx^2+dy^2)
        dx = w - ((h - b) / a)
        dy = h - b
        l = np.sqrt(dx ** 2 + dy ** 2)
    if bot & right:  # case 5
        # dx = w-x(y=h), dy = y(x=0) --> l = sqrt(dx^2+dy^2)
        dx = w - ((h - b) / a)
        dy = b
        l = np.sqrt(dx ** 2 + dy ** 2)
    if bot & left:  # case 6
        # dx = x(y=h), dy = y(x=0) --> l = sqrt(dx^2+dy^2)
        dx = (h - b) / a
        dy = b
        l = np.sqrt(dx ** 2 + dy ** 2)

    return np.abs(l), (top, bot, left, right)

def coordinates_on_line(a, b, limits):
    '''
    For a given image, a slope 'a' and offset 'b', this function gives the coordinates of all points on this
    linear line, bound by limits.
    y = a * x + b

    :param a: slope of line y=ax+b
    :param b: intersection of line with y-axis y=ax+b
    :param limits: (min x, max x, min y, max y)
    :return: zipped list of (x, y) coordinates on this line
    '''

    # Determine the slice length of the linear line with the image edges. This is needed because we need to know how
    # many x coordinates we need to generate
    #l, _ = intersection_imageedge(a, b, limits)
    l = 1
    # generate x coordinates, keep as floats (we might have floats x-coordinates), we later look for the closest x value
    # in the image. For now we keep them as floats to determine the exact y-value.
    # x should have length l. x_start and x_end are determined by where the line intersects the image border. There are
    # 4 options for the xbounds: x=limits[0], x=limits[1], x=x(y=limits[2]) --> (x=limits[2]-b)/a, x=x(y=limits[3]) -->
    # (x=limits[3]-b)/a.
    # we calculate all 4 possible bounds, then check whether they are valid (within image limits).
    # xbounds = np.array([limits[0], limits[1], (limits[2]-b)/a, (limits[3]-b)/a])
    # # filter so that 0<=x<=w and 0<=y(x)<=h. we round to prevent strange errors (27-10-22)
    # xbounds = np.delete(xbounds, (np.where(
    #     (xbounds < limits[0]) |
    #     (xbounds > limits[1]) |
    #     (np.round(a * xbounds + b) < limits[2]) |
    #     (np.round(a * xbounds + b) > limits[3])
    # )))
    # if len(xbounds) != 2:
    #     logging.error(f"Something went wrong in determining the x-coordinates for the image slice. {xbounds=}")
    #     exit()
    #x = np.linspace(np.min(xbounds)+1, np.max(xbounds)-1, int(l))
    if limits[0]<limits[1]:
        x = np.array(range(limits[0], limits[1], 1))
    else:
        x = np.array(range(limits[0], limits[1], -1))
    # calculate corresponding y coordinates based on y=ax+b, keep as floats
    y = (a * x + b)

    # return a zipped list of coordinates, thus integers
    return list(zip(x.astype(int), y.astype(int))), l

def get_normals(x, y, length=30):
    #from https://stackoverflow.com/questions/65310948/how-to-plot-normal-vectors-in-each-point-of-the-curve-with-a-given-length
    x0arr = []; dyarr = []; y0arr = []; dxarr = []
    for idx in range(len(x)-1):
        x0, y0, xa, ya = x[idx], y[idx], x[idx+1], y[idx+1]
        x0arr.append(x0); y0arr.append(y0)
        dx, dy = xa-x0, ya-y0
        norm = math.hypot(dx, dy) * 1/length
        dx /= norm
        dy /= norm
        dxarr.append(round(x0-dy))
        dyarr.append(round(y0+dx))
    return x0arr, dxarr, y0arr, dyarr    # return the normals

def get_normalsV2(x, y, length=30):
    #For each coordinate, fit with nearby points to a polynomial to better estimate the dx dy -> tangent
    x0arr = []; dyarr = []; y0arr = []; dxarr = []
    for idx in range(5, len(x)-5):
        #xarr = [x[idx-2], x[idx-1], x[idx], x[idx+1], x[idx+2]]             #define x'es to use for polynomial fitting
        #yarr = [y[idx - 2], y[idx - 1], y[idx], y[idx + 1], y[idx + 2]]     #define y's ...
        xarr = [x[idx - 4], x[idx - 3], x[idx - 2], x[idx - 1], x[idx], x[idx + 1], x[idx + 2], x[idx +3], x[idx + 4]]  # define x'es to use for polynomial fitting
        yarr = [y[idx - 4], y[idx - 3], y[idx - 2], y[idx - 1], y[idx], y[idx + 1], y[idx + 2], y[idx + 3], y[idx + 4]]  # define y's ...
        fit = np.poly1d(np.polyfit(xarr, yarr, 2))                                     #fit with second order polynomial
        x0 = x[idx]; x1 = x[idx]+1      #NOTE HERE:mannualy set to a +1 value, because if x[idx+1] is used, sometimes x1=x2 -> a NaN later on

        y0 = fit(x0); y1 = fit(x1)
        dx, dy = x1 - x0, y1 - y0
        norm = math.hypot(dx, dy) * 1/length
        dx /= norm
        dy /= norm
        x0arr.append(x0); y0arr.append(round(y0))
        dxarr.append(round(x0 - dy))
        dyarr.append(round(y0 + dx))
    return x0arr, dxarr, y0arr, dyarr  # return the normals


def get_normalsV3(x, y, L):
    # For each coordinate, fit with nearby points to a polynomial to better estimate the dx dy -> tangent
    # Take derivative of the polynomial to obtain tangent and use that one.
    x0arr = []; dyarr = []; y0arr = []; dxarr = []
    for idx in range(5, len(x) - 5):
        xarr = [x[idx - 5], x[idx - 4], x[idx - 3], x[idx - 2], x[idx - 1], x[idx], x[idx + 1], x[idx + 2], x[idx +3], x[idx + 4], x[idx + 5]]  # define x'es to use for polynomial fitting
        yarr = [y[idx - 5], y[idx - 4], y[idx - 3], y[idx - 2], y[idx - 1], y[idx], y[idx + 1], y[idx + 2], y[idx + 3], y[idx + 4], y[idx + 5]]  # define y's ...
        #xarr = [x[idx - 2], x[idx - 1], x[idx], x[idx + 1], x[idx + 2]]  # define x'es to use for polynomial fitting
        #yarr = [y[idx - 2], y[idx - 1], y[idx], y[idx + 1], y[idx + 2]]  # define y's ...

        x0 = x[idx]
        if idx == 738 or idx == 691 or idx == 524:
            print("hey")
        ft = np.polyfit(xarr, yarr, 2) # fit with second order polynomial
        fit = np.poly1d(ft)  # fit with second order polynomial
        y0 = fit(x0)
        ffit = lambda xcoord: 2 * fit[2] * xcoord + fit[1]  # derivative of a second order polynomial
        # ft = np.polyfit(xarr, yarr, 5)  # fit with fifth order polynomial
        # fit = np.poly1d(ft)  # fit with fifth order polynomial
        # y0 = fit(x0)
        # ffit = lambda xcoord: (5 * fit[5] * xcoord**4) + (4 * fit[4] * xcoord**3) + (3 * fit[3] * xcoord**2) + (2 * fit[2] * xcoord) + fit[1]  # derivative of a second order polynomial

        #if xarr[0] == x0 and xarr[1] == x0 and xarr[3] == x0 and xarr[4] == x0: #if all the x'es are the same for variable y: with a line fit x = a
        if np.sum(np.abs(np.array(xarr) - x0)) == 0:
            #TODO make in such a way that the LEFT side of the droplet will also yield normal vectors pointing INTO the dorplet
            nx = - L
            ny = 0
            xrange = np.ones(100) * x0
            yrange = np.linspace(yarr[0], yarr[-1], 100)
        #elif yarr[0] == yarr[2] and yarr[1] == yarr[2] and yarr[3] == yarr[2] and yarr[4] == yarr[2]:   #all y's are the same for variable x: fit y = a
        elif np.sum(np.abs(np.array(yarr) - yarr[0])) == 0:
            # TODO make in such a way that the TOP side of the droplet will also yield normal vectors pointing INTO the dorplet
            nx = 0
            ny = L
            xrange = np.linspace(xarr[0], xarr[-1], 100)
            yrange = np.ones(100) * y0
            # fit = np.poly1d(np.polyfit(xarr, yarr, 1))  # fit with first order polynomial
            # y0 = fit(x0)
            # ffit = lambda xcoord: fit[1] * xcoord  # derivative of a first order polynomial
        else:   #continue as normal
            dx = 1
            dy = ffit(x0)  #ffit(x0)

            normalisation = L / np.sqrt(1+dy**2)        #normalise by desired length vs length of vector

            # TODO make in such a way that the LEFT side of the droplet will also yield normal vectors pointing INTO the dorplet
            nx = -dy*normalisation
            ny = dx*normalisation
            if nx > 0:
                nx = +dy * normalisation
                ny = -dx * normalisation

            xrange = np.linspace(xarr[0], xarr[-1], 100)
            yrange = fit(xrange)
        x0arr.append(x0)
        y0arr.append(round(y0))
        dxarr.append(round(x0+nx))
        dyarr.append(round(y0+ny))
        #Below: for plotting & showing of a few polynomial fits - to investigate how good the fit is
        # if idx == 738 or idx == 691 or idx == 524:
        #     plt.plot(xarr, np.array(yarr), '.', label="coordinates")
        #     plt.plot(xrange, np.array(yrange), label="fit")
        #     plt.plot([x0,x0+nx], np.array([y0, y0+ny]), '-', label="normal to contour")
        #     plt.title(f"Zoomed-in coordinates of contour. idx = {idx}")
        #     plt.xlabel("x - coords")
        #     plt.ylabel("y - coords")
        #     plt.legend()
        #     plt.show()
        #     plt.close()
        #     print(f"idx: {idx} - {fit}")
    return x0arr, dxarr, y0arr, dyarr  # return the normals

def main():
    imgPath = "D:\\2023_08_07_PLMA_Basler5x_dodecane_1_28_S5_WEDGE_1coverslip spacer_COVERED_SIDE\\Basler_a2A5328-15ucBAS__40087133__20230807_165508421_0132.tiff"
    procStatsJsonPath = os.path.join("D:\\2023_08_07_PLMA_Basler5x_dodecane_1_28_S5_WEDGE_1coverslip spacer_COVERED_SIDE\Analysis_1\PROC_20230809115938\PROC_20230809115938_statistics.json")

    basePath = os.path.dirname(imgPath)
    analysisFolder = os.path.join(basePath,"Analysis CA Spatial")
    resizeFactor = 5            #=5 makes the image fit in your screen, but also has less data points when analysing
    FLIPDATA = True
    if not os.path.exists(analysisFolder):
        os.mkdir(analysisFolder)
        print('created path: ', analysisFolder)

    with open(procStatsJsonPath, 'r') as f:
        procStats = json.load(f)
    conversionZ = procStats["conversionFactorZ"]
    conversionXY = procStats["conversionFactorXY"]
    unitZ = procStats["unitZ"]
    unitXY = procStats["unitXY"]
    if unitZ != "nm" or unitXY != "mm":
        raise Exception("One of either units is not correct for good conversion. Fix manually or implement in code")

    # # Create a window to display the input video
    # cv2.namedWindow('Input', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Input', 500, 500)

    img = cv2.imread(imgPath)
    #convert to greyscale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    greyresizedimg = cv2.resize(grayImg, [round(5328/resizeFactor), round(4608/resizeFactor)], interpolation=cv2.INTER_AREA)
    # Apply adaptive thresholding to the blurred frame
    thresh = cv2.adaptiveThreshold(greyresizedimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    resizedimg = cv2.cvtColor(greyresizedimg, cv2.COLOR_GRAY2RGB)
    #TODO temporary, to check if coordinates can be converted properly
    greyresizedimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resizedimg = img
    # Find contours in the thresholded frame
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    list_of_pts = []
    contourList = []
    i = 0
    startEnd = [0]
    # Iterate over the contours
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        # Set a minimum threshold for contour area
        if area > 2000:
            # Check if the contour has at least 5 points
            if len(contour) >= 5:
                list_of_pts += [pt[0] for pt in contour]
                print(f"length of contours={len(contour)}")

                startEnd.append([])

                #contour = np.array(list_of_pts).reshape((-1, 1, 2)).astype(np.int32)
                contourList.append(contour)
                #contour = cv2.convexHull(contour)
                i =+ 1

    for i, contour in enumerate(contourList):
        if contour is not None and i == 2:
            #contour = [elem for elem in contour if not bool((np.isin(elem, contourList[i-1])).all())]
            #check per y value first, then compare x's to find largest x
            usableContour = []
            # ylist = np.array([elem[0][1] for elem in contour])
            # xlist = [elem[0][0] for elem in contour]
            #TODO temp to check coordinate conversion
            ylist = np.array([elem[0][1]*resizeFactor for elem in contour])
            xlist = [elem[0][0]*resizeFactor for elem in contour]
            for j in range(min(ylist), max(ylist)):
                indexesJ = np.where(ylist == j)[0]
                if len(indexesJ)>0:
                    xListpery = [xlist[x] for x in indexesJ]
                    usableContour.append([max(xListpery), j])
            useableylist = np.array([elem[1] for elem in usableContour])
            useablexlist = [elem[0] for elem in usableContour]
            #resizedimg = cv2.drawContours(resizedimg, np.array([usableContour]), -1, (0, 0, 255), 2)     #draws 1 good contour around the outer halo fringe - connects outer ends
            resizedimg = cv2.polylines(resizedimg, np.array([usableContour]), False, (0, 0, 255), 2)  # draws 1 good contour around the outer halo fringe

            #Should yield the normal for every point: output is original x&y, and corresponding normal x,y (defined as dx and dy) 30 points inwards
            x0arr, dxarr, y0arr, dyarr = get_normalsV3(useablexlist, useableylist, 225) #/resizeFactor
            angleDegArr = []
            for k in range(0, len(x0arr)):
                #TODO trying to get this to work: plotting normals obtained with above function get_normals
                #resizedimg = cv2.polylines(resizedimg, np.array([[x0arr[k], y0arr[k]], [dxarr[k], dyarr[k]]]), False, (0, 255, 0), 2)  # draws 1 good contour around the outer halo fringe#
                if k % 5 == 0:
                    resizedimg = cv2.line(resizedimg, ([x0arr[k], y0arr[k]]), ([dxarr[k], dyarr[k]]), (0, 255, 0), 2)  # draws 1 good contour around the outer halo fringe

                a = (dyarr[k] - y0arr[k])/(dxarr[k] - x0arr[k])
                b = y0arr[k] - a*x0arr[k]
                coords, l = coordinates_on_line(a, b, [x0arr[k], dxarr[k], y0arr[k], dyarr[k]])
                profile = [np.transpose(greyresizedimg)[pnt] for pnt in coords]
                # plt.plot(profile)
                # print(f"Coords are: \n {coords}\nprofile values are: \n{profile}")
                # plt.title(f"Profile with {len(profile)} datapoints")
                # plt.show()
                profile_fft = np.fft.fft(profile)  # transform to fourier space
                mask = np.ones_like(profile).astype(float)
                # NOTE: lowpass seems most important for a good sawtooth profile. Filtering half of the data off seems fine
                lowpass = round(len(profile) / 2); highpass = 2;
                mask[0:lowpass] = 0; mask[-highpass:] = 0
                profile_fft = profile_fft * mask
                profile_filtered = np.fft.ifft(profile_fft)
                # plt.plot(profile_fft)
                # plt.title("Fourier plot")
                # plt.show()

                # calculate the wrapped space
                wrapped = np.arctan2(profile_filtered.imag, profile_filtered.real)
                # plt.plot(wrapped)
                # plt.title(f"Wrapped profile")
                # plt.show()
                unwrapped = np.unwrap(wrapped)
                if FLIPDATA:
                    unwrapped = -unwrapped + max(unwrapped)
                unwrapped *= conversionZ / 1000         #if unwapped is in um: TODO fix so this can be used for different kinds of Z-unit
                x = np.arange(0, len(unwrapped)) * conversionXY * 1000 #TODO same ^
                coef1 = np.polyfit(x, unwrapped, 1)

                # plt.plot(x, unwrapped)
                # plt.title("Drop height vs distance (unwrapped profile)")
                # plt.show()
                a_horizontal = 0
                angleRad = math.atan((coef1[0] - a_horizontal) / (1 + coef1[0] * a_horizontal))
                angleDeg = math.degrees(angleRad)
                # Flip measured CA degree if higher than 45.
                if angleDeg > 45:
                    angleDeg = 90 - angleDeg
                print(f"Length of studied array is= {len(unwrapped)}")
                print(f"Calculated angle: {angleDeg} deg")

                angleDegArr.append(angleDeg)

            # Fit an ellipse around the contour
            #ellipse = cv2.fitEllipse(contour)
            # Draw the ellipse on the original frame
            #resizedimg = cv2.ellipse(resizedimg, ellipse, (0, 255, 0), 2)                #draws an ellips, which fits poorly
            # get the middlepoint of the contour and draw a circle in it
            #M = cv2.moments(contour)
            #cx = int(M["m10"] / M["m00"])
            #cy = int(M["m01"] / M["m00"])
            #resizedimg = cv2.circle(resizedimg, (cx, cy), 13, (255, 0, 0), -1)           #draws a white circle
            #cx_save.append(cx)
            #cy_save.append(cy)

            # Display the input and output frames
            #cv2.imshow('Input', resizedimg)
            #cv2.imshow('Output', thresh)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            plt.plot(y0arr, angleDegArr)
            plt.xlabel("Y-coord"); plt.ylabel("Calculated Contact Angle (deg)"); plt.title("Calculated Contact angles"); plt.show()
            tstring = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
            cv2.imwrite(os.path.join(analysisFolder, f"rawImage_contourLine{tstring}.png") , resizedimg)
            cv2.imwrite(os.path.join(analysisFolder, f"threshImage_contourLine{tstring}.png") , thresh)
if __name__ == "__main__":
    main()
    exit()