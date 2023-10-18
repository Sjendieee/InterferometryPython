import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shapely
import math
import time
import warnings

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


def get_normalsV3(x, y, L=30):
    # For each coordinate, fit with nearby points to a polynomial to better estimate the dx dy -> tangent
    # Take derivative of the polynomial to obtain tangent and use that one.
    x0arr = []; dyarr = []; y0arr = []; dxarr = []
    for idx in range(5, len(x) - 5):
        #xarr = [x[idx - 4], x[idx - 3], x[idx - 2], x[idx - 1], x[idx], x[idx + 1], x[idx + 2], x[idx +3], x[idx + 4]]  # define x'es to use for polynomial fitting
        #yarr = [y[idx - 4], y[idx - 3], y[idx - 2], y[idx - 1], y[idx], y[idx + 1], y[idx + 2], y[idx + 3], y[idx + 4]]  # define y's ...
        xarr = [x[idx - 2], x[idx - 1], x[idx], x[idx + 1], x[idx + 2]]  # define x'es to use for polynomial fitting
        yarr = [y[idx - 2], y[idx - 1], y[idx], y[idx + 1], y[idx + 2]]  # define y's ...

        x0 = x[idx]
        if idx == 738 or idx == 691 or idx == 524:
            print("hey")
        ft = np.polyfit(xarr, yarr, 2)
        fit = np.poly1d(ft)  # fit with second order polynomial
        y0 = fit(x0)
        ffit = lambda xcoord: 2 * fit[2] * xcoord + fit[1]  # derivative of a second order polynomial
        if xarr[0] == x0 and xarr[1] == x0 and xarr[3] == x0 and xarr[4] == x0: #if all the x'es are the same for variable y: with a line fit x = a
            nx = - L
            ny = 0
            xrange = np.ones(100) * x0
            yrange = np.linspace(yarr[0], yarr[-1], 100)
        elif yarr[0] == yarr[2] and yarr[1] == yarr[2] and yarr[3] == yarr[2] and yarr[4] == yarr[2]:   #all y's are the same for variable x: fit y = a
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
        if idx == 738 or idx == 691 or idx == 524:
            plt.plot(xarr, yarr, '.')
            plt.plot(xrange, yrange)
            plt.plot([x0,x0+nx], [y0, y0+ny], '-')
            plt.title(f"idx = {idx}")
            plt.show()
            plt.close()
            print(f"idx: {idx} - {fit}")
    return x0arr, dxarr, y0arr, dyarr  # return the normals

def main():
    imgPath = "D:\\2023_08_07_PLMA_Basler5x_dodecane_1_28_S5_WEDGE_1coverslip spacer_COVERED_SIDE\\Basler_a2A5328-15ucBAS__40087133__20230807_165508421_0132.tiff"
    basePath = os.path.dirname(imgPath)
    analysisFolder = os.path.join(basePath,"Analysis CA Spatial")

    if not os.path.exists(analysisFolder):
        os.mkdir(analysisFolder)
        print('created path: ', analysisFolder)

    # # Create a window to display the input video
    # cv2.namedWindow('Input', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Input', 500, 500)

    img = cv2.imread(imgPath)
    #convert to greyscale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resizedimg = cv2.resize(grayImg, [round(5328/5), round(4608/5)], interpolation=cv2.INTER_AREA)
    # Apply adaptive thresholding to the blurred frame
    thresh = cv2.adaptiveThreshold(resizedimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    resizedimg = cv2.cvtColor(resizedimg, cv2.COLOR_GRAY2RGB)
    # cv2.imshow("Greyscaled image", thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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
            ylist = np.array([elem[0][1] for elem in contour])
            xlist = [elem[0][0] for elem in contour]
            for j in range(min(ylist), max(ylist)):
                indexesJ = np.where(ylist == j)[0]
                if len(indexesJ)>1:
                    xListpery = [xlist[x] for x in indexesJ]
                else:
                    xListpery = xlist[indexesJ]
                    #[elem[0][0] for elem in contour if elem[0][1]==j]
                usableContour.append([max(xListpery), j])
            useableylist = np.array([elem[1] for elem in usableContour])
            useablexlist = [elem[0] for elem in usableContour]
            #resizedimg = cv2.drawContours(resizedimg, np.array([usableContour]), -1, (0, 0, 255), 2)     #draws 1 good contour around the outer halo fringe - connects outer ends
            resizedimg = cv2.polylines(resizedimg, np.array([usableContour]), False, (0, 0, 255), 2)  # draws 1 good contour around the outer halo fringe

            #Should yield the normal for every point: output is original x&y, and corresponding normal x,y (defined as dx and dy) 30 points inwards
            x0arr, dxarr, y0arr, dyarr = get_normalsV3(useablexlist, useableylist)
            for k in range(0, len(x0arr)):
                #TODO trying to get this to work: plotting normals obtained with above function get_normals
                #resizedimg = cv2.polylines(resizedimg, np.array([[x0arr[k], y0arr[k]], [dxarr[k], dyarr[k]]]), False, (0, 255, 0), 2)  # draws 1 good contour around the outer halo fringe#
                if k % 5 == 0:
                    resizedimg = cv2.line(resizedimg, ([x0arr[k], y0arr[k]]), ([dxarr[k], dyarr[k]]), (0, 255, 0), 2)  # draws 1 good contour around the outer halo fringe



            #TODO attempt at lines for determining a normal
            # line = shapely.linestrings([usableContour])
            # right = line.parallel_offset(20, 'right')
            # parralelPoint = right.boundary[1]


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
            cv2.imshow('Input', resizedimg)
            tstring = str(time.strftime("%H_%M_%S", time.localtime()))
            cv2.imwrite(os.path.join(analysisFolder, f"rawImage_contourLine{1}{tstring}.png") , resizedimg)
            #cv2.imshow('Output', thresh)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    exit()