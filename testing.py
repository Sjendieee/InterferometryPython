import cv2
import numpy as np
from line_method import click_event
##linmethod: pointa = 1766, 1782; pointb = 1928, 1916
# Read RGB image
img = cv2.imread('F:\\2023_04_06_PLMA_HexaDecane_Basler2x_Xp1_24_s11_split____GOODHALO-DidntReachSplit\\D_analysis_v2\PROC_20230612121104\\rawslicesimage\\rawslicesimage_Basler_a2A5328-15ucBAS__40087133__20230406_131652896_0009_analyzed_.png')
imgblack = cv2.imread('C:\\Users\ReuvekampSW\\Documents\\InterferometryPython\\black square.png')

dimensions = img.shape
print(dimensions)

#resizedimg = cv2.resize(img, [1920, 1080], interpolation = cv2.INTER_AREA)
resizedimg = cv2.resize(img, [2400, 1500], interpolation = cv2.INTER_AREA)
print(resizedimg.shape)
print(imgblack.shape)
resizedImagBlack = cv2.resize(imgblack, [10, 10], interpolation = cv2.INTER_AREA)

x_offset = 468
y_offset = 114

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
