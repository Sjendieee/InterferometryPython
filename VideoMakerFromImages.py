import os
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
import glob
import re
from matplotlib.animation import FFMpegWriter
import cv2

def path_in_use():
    """
    Write path to folder in which the analyzed images (and subsequent analysis) are
    :return:
    """
    path = "H:\\2024_05_07_PLMA_Basler15uc_Zeiss5x_dodecane_Xp1_31_S2_WEDGE_2coverslip_spacer_V3"
    path = "D:\\2024-09-04 PLMA dodecane Xp1_31_2 ZeissBasler15uc 5x M3 tilted drop"

    metadata = dict(title='4-panel Movie', artist='Sjendieee')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    return path


def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def ffmpegPath():
    paths_ffmpeg = ['C:\\Users\\ReuvekampSW\\Desktop\\ffmpeg-7.1-essentials_build\\bin\\ffmpeg.exe',  # UT pc & laptop
                    'C:\\Users\\Sander PC\\Desktop\\ffmpeg-7.1-essentials_build\\bin\\ffmpeg.exe'  # thuis pc
                    ]
    for ffmpeg_path in paths_ffmpeg:
        if os.path.exists(ffmpeg_path):
            plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path        #set path to ffmpeg file.
            break
    if not os.path.exists(ffmpeg_path):
        logging.critical("No good path to ffmpeg.exe.\n Correct path, or install from e.g. https://www.gyan.dev/ffmpeg/builds/#git-master-builds")
    return ffmpeg_path

def videoMakerOfImges(imgList, analysisFolder, compression = 100):
    #Read in size of original image for compression later
    referenceFrame = cv2.imread(imgList[0])

    (inputHeight, inputWidth, referenceLayers) = referenceFrame.shape
    outputHeight = round(inputHeight * (compression / 100))
    outputWidth = round(inputWidth * (compression / 100))

    #https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python
    ffmpeg_path = ffmpegPath()
    output_folder = analysisFolder
    video_name = f'Complete_overview_movie.mp4v'
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')       #tried for mp4 - didn't work: https://stackoverflow.com/questions/30103077/what-is-the-codec-for-mp4-videos-in-python-opencv/55596396
    video = cv2.VideoWriter(video_name, 0, 1, (outputWidth, outputHeight))      #output name, codec used, FPS, tuple of dimensions

    for n, img in enumerate(imgList):
        logging.info(f"Processing image {n}/{len(imgList)}")
        img = cv2.resize(cv2.imread(img), (outputWidth, outputHeight), interpolation=cv2.INTER_AREA)
        video.write(img)
    cv2.destroyAllWindows()
    video.release()
    logging.info(f'Finished turning images into a video')

def main():
    path_images = path_in_use()
    analysisFolder = os.path.join(path_images, "Analysis CA Spatial")  # name of output folder of Spatial Contact Analysis

    imgList = [f for f in glob.glob(os.path.join(analysisFolder, f"Complete overview*png"))]        #this grabs all the 4-panel images in folder
    videoMakerOfImges(imgList, analysisFolder, compression = 50)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')     #configuration for printing logging messages. Can be removed safely
    main()
    exit()
