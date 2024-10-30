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
# Read RGB image
#source = 'D:\\2023_04_06_PLMA_HexaDecane_Basler2x_Xp1_24_s11_split____GOODHALO-DidntReachSplit\\D_analysisv4\\PROC_20230724185238'     #hexadecane
#source = 'E:\\2023_04_06_PLMA_HexaDecane_Basler2x_Xp1_24_s11_split____GOODHALO-DidntReachSplit\\D_analysisv4\\PROC_20230913122145_condensOnly'     #hexadecane, condens only
#source = "F:\\2023_02_17_PLMA_DoDecane_Basler2x_Xp1_24_S9_splitv2____DECENT_movedCameraEarly\\B_Analysis_V2\\PROC_20230829105238"       #dodecane
#source = "E:\\2023_08_30_PLMA_Basler2x_dodecane_1_29_S2_ClosedCell\\B_Analysis2\\PROC_20230905134930"           #dodecane 2d
#source = "E:\\2023_04_03_PLMA_TetraDecane_Basler2x_Xp1_24_s10_single______DECENT\\A_Analysis\\PROC_20230915102215"  # tetradecane swelling
#source = "E:\\2023_02_13_PLMA_Hexadecane_Basler2x_Xp1_24_S10_split_v2\\Analysis_v2\\PROC_20230919122236_imbed_conds"    #hexadecane v2_EvapConds
#source = "E:\\2023_02_13_PLMA_Hexadecane_Basler2x_Xp1_24_S10_split_v2\\Analysis_v2\\PROC_20230919150913_conds"          #hexadecane v2_conds only
#source = "E:\\2023_09_22_PLMA_Basler2x_hexadecane_1_29S2_split\\B_Analysis\\PROC_20230927135916_imbed"          #hexadecane, imbed
#source = "D:\\2023_09_21_PLMA_Basler2x_tetradecane_1_29S2_split_ClosedCell\\B_Analysis\\PROC_20230927143637_condens"        #tetradecane split, imbed & condens
#source = "E:\\2023_09_22_PLMA_Basler2x_hexadecane_1_29S2_split\\B_Analysis\\PROC_20230927135916_imbed"          #hexadecane, imbed

#source = "E:\\2023_04_06_PLMA_HexaDecane_Basler2x_Xp1_24_s11_split____GOODHALO-DidntReachSplit\\D_analysisv4\\PROC_20230724185238"        #tetradecane split, imbed & condens
source = "M:\\Enqing\\Halo_Zeiss20X\\Img2\\Line1"

nAllImages = [0]        #images to plot line for
pixellocationLarge = range(1249-300, 7957)     #pixellocations to plot for. Can be a single value, or a range (which will draw a line)
showPixelLocationLegend = False         #show in a legend the pixellocation (usefull when only 1 or a few pixels are plotted)
#INPUT EDGES OF THE LINE WITH BORDER OF IMAGE IN PLOT AS (P1 = [x,y])
#Check this in e.g. paint.net with the cursor
#Eqning dataset
P1 = [1477, 111]
P2 = [1936, 1173]

#     #For hexadecane, v1
# P1 = [466, 206]
# P2 = [1892, 1382]
    #For hexadecane, condens only
#P1 = [466, 414]
#P2 = [1933, 418]
    #For dodecane
# P1 = [467, 611]
# P2 = [1932, 302]
    #Dodecane v2
#P1 = [467, 472]
#P2 = [1933, 444]
    #tetradecane
#P1 = [1460, 114]
#P2 = [1684, 1382]
    #hexadecaneV2 evap conds
#P1 = [817, 114]
#P2 = [910, 1382]
    #hexadecanev2 conds only
#P1 = [467, 439]
#P2 = [1933, 475]
    #tetradecane split, imbed
# P1 = [689, 114]
# P2 = [1130, 1381]
    #tetradecane split, condens
# P1 = [461, 360]
# P2 = [1933, 363]
    #hexadecane 09_22 split, imbed
# P1 = [1097, 114]
# P2 = [467, 1040]

# TODO make sure this path is correct as well to the square to be inputted
imgblack = cv2.imread('C:\\Users\\ReuvekampSW\\Documents\\InterferometryPython\\red square.png')
#imgblack = cv2.imread('C:\\Users\\ReuvekampSW\\PycharmProjects\\InterferometryPython\\red square.png')
imgList = [f for f in glob.glob(os.path.join(source, f"rawslicesimage\\*.png"))]
CLICKEVENT = False
n = 0
print(f"Total amount of images in folder: {len(imgList)}. \nTotal amount of images used: {len(nAllImages)}")
for imgPath in imgList:
    if n in nAllImages:
        img = cv2.imread(imgPath)
        csvList = [f for f in glob.glob(os.path.join(source, f"csv\\*unwrapped.csv"))]
        resizedimg = cv2.resize(img, [2400, 1500], interpolation = cv2.INTER_AREA)
        squareSize = 20
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


        a, b = calcLineEquation([P1[0], P2[0]], [P1[1], P2[1]])
        limits = [466, 1937, 112, 1385]     #xmin xmin ymin ymax of image in plot. Should always be same
        l = calcLineLength(P1, P2)
        print(f"Image in plot: length calculated is {l} from edges")

        lLarge = readInDataLengthLargeImage(csvList)    #read in OG image data length of line from counting rows in csv file
        print(f"Length of data in Large image= {lLarge}")
        ratioLines = lLarge / l

        for i, pixelLocation in enumerate(pixellocationLarge):
            pixelLocNew = pixelLocation / ratioLines
            print(f"ratioLines = {ratioLines}, pixelLocLarge = {pixelLocation}, pixelLocNew = {pixelLocNew}")

            c = pixelLocNew     #c = length of line (schuine zijde)
            x_offset = round(c / (np.sqrt(1+a**2)) + (P1[0]))
            pixely = round(a*(x_offset) + b)
            x_offset = round(x_offset - squareSize/2)    #-squareSize/2 to centre the square
            y_offset = round(pixely - squareSize/2)         #-squareSize/2 to centre the square
            print(f"pixelLocNew={pixelLocNew} x={x_offset}, y = {y_offset}")
            resizedimg[y_offset:y_offset+resizedImagBlack.shape[0], x_offset:x_offset+resizedImagBlack.shape[1]] = resizedImagBlack
            if showPixelLocationLegend:
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