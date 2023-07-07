import cv2
import numpy as np
from line_method import click_event, coordinates_on_line
##linmethod: pointa = 1766, 1782; pointb = 1928, 1916
# Read RGB image
img = cv2.imread('F:\\2023_04_06_PLMA_HexaDecane_Basler2x_Xp1_24_s11_split____GOODHALO-DidntReachSplit\\D_analysis_v2\PROC_20230612121104\\rawslicesimage\\rawslicesimage_Basler_a2A5328-15ucBAS__40087133__20230406_131652896_0009_analyzed_.png')
imgblack = cv2.imread('C:\\Users\ReuvekampSW\\Documents\\InterferometryPython\\black square.png')
resizedimg = cv2.resize(img, [2400, 1500], interpolation = cv2.INTER_AREA)
squareSize = 10
resizedImagBlack = cv2.resize(imgblack, [squareSize, squareSize], interpolation = cv2.INTER_AREA)

pixellocationLarge = 4000

#x_offset = 468
#y_offset = 114

a = 0.827846
b = -183.776
limits = [466, 1937, 112, 1385]     #xmin xmin ymin ymax
coordinates, l = coordinates_on_line(a, b, limits)      #INCORRECT VGM. ZELF BEREKEND=1852
print(f"Length of line is {l}. (Image in plot)")

x_coords = [1766, 1928]
y_coords = [1782, 1916]
aL = (y_coords[1]-y_coords[0])/(x_coords[1]-x_coords[0])
bL = y_coords[0] - a * x_coords[0]
limitsL = [0, 5328, 0, 4608]
coordinatesLarge, lLarge = coordinates_on_line(aL, bL, limitsL)
print(f"Length of line is {lLarge}. (Large plot). a ={aL}, b = {bL}")

ratioLines = lLarge / 1852
pixelLocNew = pixellocationLarge / ratioLines
print(f"ratioLines = {ratioLines}, pixelLocLarge = {pixellocationLarge}, pixelLocNew = {pixelLocNew}")
#pixely = round(a*(pixelLocNew+466) + b)

#x_offset = round(pixelLocNew)  + 466
#y_offset = pixely

c = pixelLocNew     #c = length of line (schuine zijde)
#x_offset = round((-a*b + np.sqrt(c**2 + a**2 * c**2 - b**2)) / (1+a**2))
x_offset = round(c / (np.sqrt(1+a**2)) + 466 - squareSize/2)            #-squareSize/2 to centre it
pixely = round(a*(x_offset) + b)
y_offset = pixely
print(f"pixelLocNew={pixelLocNew} x={x_offset}, y = {y_offset}")

resizedimg[y_offset:y_offset+resizedImagBlack.shape[0], x_offset:x_offset+resizedImagBlack.shape[1]] = resizedImagBlack
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
