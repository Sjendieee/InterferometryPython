import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shapely
import math

def get_normals(x,y, length=30):
    #from https://stackoverflow.com/questions/65310948/how-to-plot-normal-vectors-in-each-point-of-the-curve-with-a-given-length
    x0arr = []
    dyarr = []
    y0arr = []
    dxarr = []
    for idx in range(len(x)-1):
        x0, y0, xa, ya = x[idx], y[idx], x[idx+1], y[idx+1]
        x0arr.append(x0)
        y0arr.append(y0)
        dx, dy = xa-x0, ya-y0
        norm = math.hypot(dx, dy) * 1/length
        dx /= norm
        dy /= norm
        dxarr.append(round(x0-dy))
        dyarr.append(round(y0+dx))
    return x0arr, dxarr, y0arr, dyarr    # return the normals

def main():
    imgPath = "G:\\2023_08_07_PLMA_Basler5x_dodecane_1_28_S5_WEDGE_1coverslip spacer_COVERED_SIDE\\Basler_a2A5328-15ucBAS__40087133__20230807_165508421_0132.tiff"
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
            x0arr, dxarr, y0arr, dyarr = get_normals(useablexlist, useableylist)
            for k in range(0, len(x0arr)):
                #TODO trying to get this to work: plotting normals obtained with above function get_normals
                #resizedimg = cv2.polylines(resizedimg, np.array([[x0arr[k], y0arr[k]], [dxarr[k], dyarr[k]]]), False, (0, 255, 0), 2)  # draws 1 good contour around the outer halo fringe#
                resizedimg = cv2.line(resizedimg, ([x0arr[k], y0arr[k]]), ([dxarr[k], dyarr[k]]), False, (0, 255, 0), 2)  # draws 1 good contour around the outer halo fringe

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
            cv2.imwrite(os.path.join(analysisFolder, f"rawImage_contourLine{1}.png") , resizedimg)
            #cv2.imshow('Output', thresh)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    exit()