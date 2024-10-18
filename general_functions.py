import statistics
from datetime import datetime
import logging
import cv2
import os
from natsort import natsorted
import re
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from skimage import data, img_as_float, color, exposure
from skimage.restoration import unwrap_phase

def TimeRemaining(arraytimes, left):
    avgtime = statistics.mean(arraytimes)
    timeremaining = left * avgtime
    if timeremaining < 2:
        rem = f"Almost done now ..."
    elif timeremaining < 90:
        rem = f"{round(timeremaining)} seconds"
    else:
        rem = f"{round(timeremaining / 60)} minutes"
    logging.info(f"Estimated time remaining: {rem}")
    return True

def image_resize_percentage(image, scale_percent):
    new_width = int(image.shape[1] * scale_percent / 100)
    new_height = int(image.shape[0] * scale_percent / 100)
    new_dim = (new_width, new_height)
    logging.debug(f"Image ({image.shape[1]} x {image.shape[0]}) resized to ({new_width} x {new_height}.)")
    return cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    logging.debug(f"Image ({w} x {h}) resized to ({dim[0]} x {dim[1]}.)")
    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    # return the resized image
    return resized

def list_images(source, config):
    if os.path.isdir(source):
            images = os.listdir(source)
            folder = source
    else:
        if not os.path.exists(source):
            logging.error(f"File {source} does not exist.")
            exit()
        images = [os.path.basename(source)]
        folder = os.path.dirname(source)
        if config.get("GENERAL", "ANALYSIS_RANGE") != 'False':
            source2 = config.get("GENERAL", "ANALYSIS_RANGE")
            images2 = [os.path.basename(source2)]
            imagestot = os.listdir(folder)
            imagesrange = [images[0]]
            for i in range(imagestot.index(images[0])+1, imagestot.index(images2[0])+1):
                imagesrange.append(imagestot[i])
            images = imagesrange
    imagestrimmed = images[0::config.getint("GENERAL", "ANALYSIS_INTERVAL")]
    imagestrimmed = [img for img in imagestrimmed if
              img.endswith(".tiff") or img.endswith(".png") or img.endswith(".jpg") or img.endswith(
                  ".jpeg") or img.endswith(".bmp")]
    if not images:
        raise Exception(
            f"{datetime.now().strftime('%H:%M:%S')} No images with extension '.tiff', '.png', '.jpg', '.jpeg' or '.bmp' found in selected folder.")
    return natsorted(imagestrimmed), folder, [os.path.join(folder, i) for i in natsorted(imagestrimmed)]

def get_timestamps(config, filenames, filenames_fullpath):
    '''
    For a given list of filenames, it extracts the datetime stamps from the filenames and determines the time difference
    in seconds between those timestamps. If TIMESTAMPS_FROMFILENAME is False, the set INPUT_FPS will be used to
    determine the time difference between images; timestamps will be None in that case.

    It uses standard regex expressions to look for a pattern in the filename string. Then standard c datetime format
    codes to convert that pattern into a datetime object.

    :param config: ConfigParser config object with settings
        GENERAL, TIMESTAMPS_FROMFILENAME    True or False. If False INPUT_FPS is used for deltatime, timestamps=None
        GENERAL, FILE_TIMESTAMPFORMAT_RE    The regex pattern to look for in the filename
        GENERAL, FILE_TIMESTAMPFORMAT       The datetime format code in the regex pattern for conversion
        GENERAL, INPUT_FPS                  If TIMESTAMPS_FROMFILENAME is False, use fps to determine deltatime
    :param filenames: list with filenames as strings, filenames_fullpath: list with fill path filenames as strings
    :return: timestamps, deltatime
    '''


    if config.getboolean("GENERAL", "TIMESTAMPS_FROMFILENAME"):
        timestamps = []
        for f in filenames:
            # match = re.search(r"[0-9]{14}", f)  # we are looking for 14 digit number
            match = re.search(config.get("GENERAL", "FILE_TIMESTAMPFORMAT_RE"), f)  # we are looking for 14 digit number
            if not match:
                logging.error("No 14-digit timestamp found in filename.")
                exit()
            try:
                timestamps.append(datetime.strptime(match.group(0), config.get("GENERAL", "FILE_TIMESTAMPFORMAT")))
            except:
                logging.error("Could not obtain a %d%m%Y%H%M%S timestamp from the 14-digit number in filename.")
                exit()
        deltatime = timestamps_to_deltat(timestamps)
        logging.info("Timestamps read from filenames. Deltatime calculated based on timestamps.")
    elif config.getboolean("GENERAL", "TIMESTAMPS_FROMCREATIONDATE"):
        # read from creation date file property
        timestamps = []
        for f in filenames_fullpath:
            timestamps.append(datetime.fromtimestamp(os.path.getctime(f)))
        deltatime = timestamps_to_deltat(timestamps)
        logging.info("Timestamps read from filenames. Deltatime calculated based on creation time.")
    elif config.getboolean("GENERAL", "TIMESTAMPS_FROMMODIFIEDDATE"):
        # read from creation date file property
        timestamps = []
        for f in filenames_fullpath:
            timestamps.append(datetime.fromtimestamp(os.path.getmtime(f)))
        deltatime = timestamps_to_deltat(timestamps)
        logging.info("Timestamps read from filenames. Deltatime calculated based on creation time.")
    else:
        timestamps = None
        # deltatime = np.arange(0, len(filenames)) * config.getfloat("GENERAL", "INPUT_FPS")
        deltatime = np.ones(len(filenames)) * config.getfloat("GENERAL", "INPUT_FPS")
        logging.warning("Deltatime calculated based on fps.")

    return timestamps, deltatime

def timestamps_to_deltat(timestamps):
    deltat = [0]
    for idx in range(1, len(timestamps)):
        deltat.append((timestamps[idx] - timestamps[idx - 1]).total_seconds())
    return deltat

def check_outputfolder(config):
    folders = {}

    folders['save'] = config.get("SAVING", "SAVEFOLDER")
    if folders['save'] == 'source':
        source = config.get("GENERAL", "source")
        if os.path.isdir(source):
            folder = source
        else:
            folder = os.path.dirname(source)
        folders['save'] = folder

    if not os.path.exists(folders['save']):
        os.mkdir(folders['save'])

    proc = f"PROC_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    if config.getboolean("SAVING", "SAVEFOLDER_CREATESUB"):
        folders['save'] = os.path.join(folders['save'], proc)
        os.mkdir(folders['save'])

    if config.getboolean("SAVING", "SEPARATE_FOLDERS"):
        folders['csv'] = os.path.join(folders['save'], 'csv')
        folders['npy'] = os.path.join(folders['save'], 'npy')
        os.mkdir(folders['csv'])
        os.mkdir(folders['npy'])
        if config.get('GENERAL', 'ANALYSIS_METHOD').lower() == 'surface':
            folders['save_process'] = os.path.join(folders['save'], 'process')
            folders['save_unwrapped3d'] = os.path.join(folders['save'], 'unwrapped3d')
            folders['save_wrapped'] = os.path.join(folders['save'], 'wrapped')
            folders['save_unwrapped'] = os.path.join(folders['save'], 'unwrapped')
            os.mkdir(folders['save_process'])
            os.mkdir(folders['save_unwrapped3d'])
            os.mkdir(folders['save_wrapped'])
            os.mkdir(folders['save_unwrapped'])
        elif config.get('GENERAL', 'ANALYSIS_METHOD').lower() == 'line':
            folders['save_rawslices'] = os.path.join(folders['save'], 'rawslices')
            folders['save_process'] = os.path.join(folders['save'], 'process')
            folders['save_rawslicesimage'] = os.path.join(folders['save'], 'rawslicesimage')
            folders['save_unwrapped'] = os.path.join(folders['save'], 'unwrapped')
            os.mkdir(folders['save_rawslices'])
            os.mkdir(folders['save_process'])
            os.mkdir(folders['save_rawslicesimage'])
            os.mkdir(folders['save_unwrapped'])
    else:
        folders['csv'] = folders['npy'] = folders['save']
        if config.get('GENERAL', 'ANALYSIS_METHOD').lower() == 'surface':
            folders['save_process'] = folders['save_unwrapped3d'] = folders['save_rawslicesimage'] = folders[
                'save_unwrapped'] = folders['save']
        elif config.get('GENERAL', 'ANALYSIS_METHOD').lower() == 'line':
            folders['save_rawslices'] = folders['save_process'] = folders['save_rawslicesimage'] = folders[
                'save_unwrapped'] = folders['save']

    folders['savepath'] = os.path.abspath(folders['save'])

    return folders, proc

def image_preprocessing(config, imagepath):
    im_raw = cv2.imread(imagepath)
    if config.get("IMAGE_PROCESSING", "IMAGE_ROTATE") != 'False':
        import imutils
        im_raw = imutils.rotate(im_raw, angle=config.getint("IMAGE_PROCESSING", "IMAGE_ROTATE"))
        logging.debug('Image rotated.')
    if config.get("IMAGE_PROCESSING", "IMAGE_RESIZE") != 'False':
        im_raw = image_resize_percentage(im_raw, config.getint("IMAGE_PROCESSING", "IMAGE_RESIZE"))
        logging.debug('Image resized.')
    im_gray = cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY)
    if config.get("IMAGE_PROCESSING", "IMAGE_CROP") != 'False':
        crop = [int(e.strip()) for e in config.get('IMAGE_PROCESSING', 'IMAGE_CROP').split(',')]
        im_gray = im_gray[crop[0]:-crop[2], crop[3]:-crop[1]]
        logging.debug(f'Image cropped. Pixels removed from edges: {crop}.')
    if config.getboolean("IMAGE_PROCESSING", "IMAGE_CONTRAST_ENHANCE"):
        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE()
        im_gray = clahe.apply(im_gray)
        logging.debug('Image contrast enhanced.')
    if config.getboolean("IMAGE_PROCESSING", "IMAGE_DENOISE"):
        cv2.fastNlMeansDenoising(im_gray)
        logging.debug('Image denoised.')

    # # TODO temporarily testing to save greyscaled images
    # image = exposure.rescale_intensity(im_gray, out_range=(0, 4 * np.pi))
    # cv2.imwrite(os.path.join("C:\\TEMP_data_for_conversion", "greyimg_scaled.png"), image)
    # image_wrapped = np.angle(np.exp(1j * image))
    # cv2.imwrite(os.path.join("C:\\TEMP_data_for_conversion", "img_wrapped.png"), image)
    # image_unwrapped = unwrap_phase(image_wrapped)
    # cv2.imwrite(os.path.join("C:\\TEMP_data_for_conversion", "img_unwrappedwrapped.png"), image_unwrapped)
    #
    # fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    # ax1, ax2, ax3, ax4 = ax.ravel()
    #
    # fig.colorbar(ax1.imshow(image, cmap='gray', vmin=0, vmax=4 * np.pi), ax=ax1)
    # ax1.set_title('Original')
    # fig.colorbar(ax2.imshow(image_wrapped, cmap='gray', vmin=-np.pi, vmax=np.pi),
    #              ax=ax2)
    # ax2.set_title('Wrapped phase')
    # fig.colorbar(ax3.imshow(image_unwrapped, cmap='gray'), ax=ax3)
    # ax3.set_title('After phase unwrapping')
    # fig.colorbar(ax4.imshow(image_unwrapped - image, cmap='gray'), ax=ax4)
    # ax4.set_title('Unwrapped minus original')
    # plt.show()
    return im_gray, im_raw


def verify_settings(config, stats):
    # plotting on, while more than 1 image
    if len(stats['inputImages']) > 1 and config.getboolean("PLOTTING", "SHOW_PLOTS"):
        logging.warning(f"There are {len(stats['inputImages'])} images to be analyzed and SHOW_PLOTS is True.")
        # TODO prompt with timeout?

    # wavelength should be in nm
    if config.getfloat('GENERAL', 'WAVELENGTH') < 1:
        logging.error('WAVELENGTH should be set in nm, not meters (number is too small).')
        return False

    return True


def conversion_factors(config):
    units = ['nm', 'um', 'mm', 'm', 'pixels']
    conversionsXY = [1e6, 1e3, 1, 1e-3, 1]  # standard unit is um
    conversionsZ = [1, 1e-3, 1e-6, 1e-9, 1]  # standard unit is nm


    # Determine XY conversion factor and unit
    try:
        conversionFactorXY = config.getfloat('LENS_PRESETS', config.get('GENERAL', 'LENS_PRESET'))
        logging.info(f"Lens preset '{config.getfloat('LENS_PRESETS', config.get('GENERAL', 'LENS_PRESET'))}' is used.")
    except ValueError:
        logging.error(f"The set lens preset '{config.get('GENERAL', 'LENS_PRESET')}' is not in LENS_PRESETS.")
        exit()

    unitXY = config.get('GENERAL', 'UNIT_XY')
    if unitXY not in units:
        raise ValueError(f"Desired unit {unitXY} is not valid. Choose, nm, um, mm or m.")
    if unitXY == 'pixels':
         conversionFactorXY = 1
    conversionFactorXY = 1 / conversionFactorXY * conversionsXY[units.index(unitXY)]  # apply unit conversion

    # Determine Z conversion factor and unit
    unitZ = config.get('GENERAL', 'UNIT_Z')
    if unitZ == 'pi':
        conversionFactorZ = 1
    else:
        conversionFactorZ = (config.getfloat('GENERAL', 'WAVELENGTH')) / (2 * config.getfloat('GENERAL', 'REFRACTIVE_INDEX')) / (2 * np.pi)  # 1 period of 2pi = lambda / (4n). /2pi since our wrapped space is in absolute units, not pi
        if unitZ not in units:
            raise ValueError(f"Desired unit {unitZ} is not valid. Choose, nm, um, mm or m.")
        conversionFactorZ = conversionFactorZ * conversionsZ[units.index(unitZ)]  # apply unit conversion

    return conversionFactorXY, conversionFactorZ, unitXY, unitZ


#TODO vgm niet functional
def makeMovie(imgArr):
    frames = []  # for storing the generated images
    fig = plt.figure()
    for i in range(len(imgArr)):
        frames.append([plt.imshow(imgArr[i], animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                    repeat_delay=1000)
    ani.save('movie.mp4')
    return True

#TODO vgm niet functional
def makeMovie2(imgArr, SaveFolder, savename, deltatime):
    video_name = os.path.join(SaveFolder, savename)
    images = imgArr
    frame = (imgArr[0])
    height, width = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width, height))
    n = 0
    for i in images:
        I1 = ImageDraw.Draw(i)
        # Add Text to an image
        I1.text((10, 10), f"{round(deltatime[n])} seconds ", fill=(255, 0, 0))
        video.write(I1)
        n += 1

    cv2.destroyAllWindows()
    video.release()
    return True