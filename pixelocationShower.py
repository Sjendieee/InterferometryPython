####Working for hexadecane      (pixelLocation working properly). Change P1 & P2 for other rawsliceimages
import glob
import os.path
import cv2
import numpy as np
import csv
from line_method import click_event, coordinates_on_line
""""
This part is to show dots of pixellocations for all swellingImages, without clickingevents (below; to obtain required sizes in pixels etc.).
"""
##linmethod: pointa = 1766, 1782; pointb = 1928, 1916
# Read RGB image
source = 'D:\\2023_04_06_PLMA_HexaDecane_Basler2x_Xp1_24_s11_split____GOODHALO-DidntReachSplit\\D_analysisv4\\PROC_20230724185238'
imgList = [f for f in glob.glob(os.path.join(source, f"rawslicesimage\\*.png"))]
pixellocationLarge = [0, 2200, 4000, 6707]#2170
CLICKEVENT = False
n = 0
#nAllImages = np.arange(0, len(imgList),1)
nAllImages = [0, 50]
print(f"Total amount of images in folder: {len(imgList)}. \nTotal amount of images used: {len(nAllImages)}")
for imgPath in imgList:
    if n in nAllImages:
        img = cv2.imread(imgPath)
        csvList = [f for f in glob.glob(os.path.join(source, f"csv\\*unwrapped.csv"))]

        #imgblack = cv2.imread('C:\\Users\\ReuvekampSW\\Documents\\InterferometryPython\\red square.png')
        imgblack = cv2.imread('C:\\Users\\ReuvekampSW\\PycharmProjects\\InterferometryPython\\red square.png')
        resizedimg = cv2.resize(img, [2400, 1500], interpolation = cv2.INTER_AREA)
        squareSize = 25
        resizedImagBlack = cv2.resize(imgblack, [squareSize, squareSize], interpolation = cv2.INTER_AREA)

        def calcLineEquation(x_coords, y_coords):       #For eq   y = ax +b
            aL = (y_coords[1]-y_coords[0])/(x_coords[1]-x_coords[0])    #a = dy /dx
            bL = y_coords[0] - aL * x_coords[0]                          #b = y(x) - a*x
            return aL, bL
        def calcLineLength(edge1, edge2):   #edge1(x,y)    edge2(x,y)
            return ((edge2[0] - edge1[0])**2 + (edge2[1] - edge1[1])**2 )**0.5
        def readInDataLengthLargeImage(csvList):
            with open(csvList[0]) as f:
                summation = sum(1 for line in f)
            return summation-1

        #P1 and P2 along the draw line of Image in plot.
        #INPUT EDGES OF THE LINE WITH BORDER OF IMAGE IN PLOT AS (P1 = [x,y])
        #Check this in e.g. paint.net with the cursor
        P1 = [466, 206]
        #P2 = [1091, 725]
        P2 = [1892, 1382]           #For hexadecane
        a, b = calcLineEquation([P1[0], P2[0]], [P1[1], P2[1]])
        limits = [466, 1937, 112, 1385]     #xmin xmin ymin ymax of image in plot. Should always be same
        l = calcLineLength(P1, P2)
        print(f"Image in plot: length calculated is {l} from edges")

        #x_coords = [1766, 1782]
        #y_coords = [1928, 1916]
        #aL, bL = calcLineEquation(x_coords, y_coords)
        #print(f"(Large plot). a ={aL}, b = {bL}")
        lLarge = readInDataLengthLargeImage(csvList)    #read in OG image data length of line from counting rows in csv file
        print(f"Length of data in Large image= {lLarge}")
        ratioLines = lLarge / l

        for i, pixelLocation in enumerate(pixellocationLarge):
            pixelLocNew = pixelLocation / ratioLines
            print(f"ratioLines = {ratioLines}, pixelLocLarge = {pixelLocation}, pixelLocNew = {pixelLocNew}")

            c = pixelLocNew     #c = length of line (schuine zijde)
            x_offset = round(c / (np.sqrt(1+a**2)) + 466 - squareSize/2)            #-squareSize/2 to centre it
            pixely = round(a*(x_offset) + b)
            y_offset = pixely
            print(f"pixelLocNew={pixelLocNew} x={x_offset}, y = {y_offset}")
            resizedimg[y_offset:y_offset+resizedImagBlack.shape[0], x_offset:x_offset+resizedImagBlack.shape[1]] = resizedImagBlack
            resizedimg = cv2.putText(resizedimg, f"pix: {pixelLocation}", [2000, 120+i*30], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
            if CLICKEVENT:
                cv2.imshow('image', resizedimg)
                right_clicks = []
                def click_event(event, x, y, flags, params):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        right_clicks.append([x, y])
                    if len(right_clicks) == 2:
                        cv2.destroyAllWindows()

                cv2.setMouseCallback('image', click_event)
                cv2.waitKey(0)
                P1 = np.array(right_clicks[0])
                P2 = np.array(right_clicks[1])
                print(f"Selected coordinates: {P1=}, {P2=}.")
                print(f"Selected coordinates: P1 = [{P1[0]:.0f}, {P1[1]:.0f}], P2 = [{P2[0]:.0f}, {P2[1]:.0f}]")
                cv2.destroyAllWindows()
        if not os.path.exists(os.path.join(source, f"rawslicesimage\\pixelLocation")):
            os.mkdir(os.path.join(source, f"rawslicesimage\\pixelLocation"))
        cv2.imwrite(os.path.join(source, f"rawslicesimage\\pixelLocation\\rawlsiceimageWithPixelLocation{n}.png") , resizedimg)
    n = n+1