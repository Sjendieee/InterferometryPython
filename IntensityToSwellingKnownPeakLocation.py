import logging
import os
import traceback
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from SwellTest import idk, idkPre1stExtremum, idkPostLastExtremum
from SwellingFromAbsoluteIntensity import readDataFromfileV2, saveDataToFile, selectMinimaAndMaxima
from line_method import timeFormat

def heightFromIntensityProfileV3(FLIP, MANUALPEAKSELECTION, PLOTSWELLINGRATIO, SAVEFIG, SEPERATEPLOTTING, USESAVEDPEAKS,
                                 ax0, ax1, cmap, colorGradient, dryBrushThickness, elapsedtime, fig0, fig1, idx, idxx,
                                 intensityProfileZoomConverted, knownHeightArr, knownPixelPosition, normalizeFactor,
                                 range1, range2, source, xshifted, vectorNumber, outwardsLengthVector, unitXY= "mm", extraPartIndroplet=0, knownPeaks = [], knownMinima = []):

    if not os.path.exists(os.path.join(source, f"Swellingimages")):
        os.mkdir(os.path.join(source, f"Swellingimages"))
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})  # print arrays later with only 2 decimals

    if len(knownPeaks) == 0 or len(knownMinima) == 0:
        MANUALPEAKSELECTION = True
    else:
        peaks = knownPeaks
        minima = knownMinima
        MANUALPEAKSELECTION = False

        ######################################################################################################
        ############## Below: calculate height profiles making use of the known maxima & minima ##############
        ######################################################################################################
        minAndMax = np.concatenate([peaks, minima])
        minAndMaxOrderedUnsorted = np.sort(minAndMax)
        minAndMaxOrdered = []
        ###Below: sort the list of minima and maxima such that minima and maxima are alternating.
        ### this requires all min & maxima to be 'correctly found' beforehand:
        for i, ival in enumerate(minAndMaxOrderedUnsorted):
            if i == 0:  # always input first extremum
                minAndMaxOrdered.append(ival)
            else:
                if minAndMaxOrdered[-1] in minima:  # if last value in new adjust list is a minimum:
                    # then next value should be a maximum OR next minimum should be of a lower intensity value (to find absolute minimum)
                    if ival in peaks:
                        minAndMaxOrdered.append(ival)
                    # Check if next extremum has a lower intensity value (and replace if yes):
                    elif intensityProfileZoomConverted[ival] < intensityProfileZoomConverted[minAndMaxOrdered[-1]]:
                        minAndMaxOrdered[-1] = ival
                    # #else, find
                    # for maximum in peaks:
                    #     if maximum > minAndMaxOrdered[-1]:
                    #         minAndMaxOrdered.append(maximum)
                    #         break
                elif minAndMaxOrdered[-1] in peaks:  # if last value in new adjust list is a maximum:
                    # then next value should be a minimum OR next maximum should have a higher intensity value (to find absolute maximum)
                    if ival in minima:
                        minAndMaxOrdered.append(ival)
                    # Check if next extremum has a lower intensity value (and replace if yes):
                    elif intensityProfileZoomConverted[ival] > intensityProfileZoomConverted[minAndMaxOrdered[-1]]:
                        minAndMaxOrdered[-1] = ival

                    # for minimum in minima:
                    #     if minimum > minAndMaxOrdered[-1]:
                    #         minAndMaxOrdered.append(minimum)
                    #         break
                else:
                    print(f"Skipped {minAndMaxOrderedUnsorted[i]}")

    # TODO select regions in plot to find minima and maxima
    if MANUALPEAKSELECTION:  # use manually selected peaks, either from a previous time or select new ones now
        if USESAVEDPEAKS:  # use peaks from a previous time (if they exist)
            #TODO this way of saving will create A LOT of files for different vectors in same image: optimization desired
            if os.path.exists(os.path.join(source, f"SwellingImages\\MinAndMaximaHandpicked{idx}_{vectorNumber}_{outwardsLengthVector}_{extraPartIndroplet}.txt")):
                minAndMaxOrdered = readDataFromfileV2(
                    os.path.join(source, f"SwellingImages\\MinAndMaximaHandpicked{idx}_{vectorNumber}_{outwardsLengthVector}_{extraPartIndroplet}.txt"))
                print(f">Imported saved peaks from 'MinAndMaximaHandpicked{idx}_{vectorNumber}_{outwardsLengthVector}_{extraPartIndroplet}.txt'")
            else:
                try:
                    print(f"No saved peaks yet. Select them now:")
                    minAndMaxOrdered = selectMinimaAndMaxima(np.divide(intensityProfileZoomConverted, normalizeFactor), idx)
                    saveDataToFile(minAndMaxOrdered, os.path.join(source, f"SwellingImages"), f"MinAndMaximaHandpicked{idx}_{vectorNumber}_{outwardsLengthVector}_{extraPartIndroplet}.txt")
                except:
                    logging.error("Some error occured while trying to manually select peaks!")
                    print(traceback.format_exc())
        else:  # select new peaks now
            try:
                minAndMaxOrdered = selectMinimaAndMaxima(np.divide(intensityProfileZoomConverted, normalizeFactor), idx)
                saveDataToFile(minAndMaxOrdered, os.path.join(source, f"SwellingImages"), f"MinAndMaximaHandpicked{idx}_{vectorNumber}_{outwardsLengthVector}_{extraPartIndroplet}.txt")
            except:
                logging.error("Some error occured while trying to manually select peaks!")
                print(traceback.format_exc())
        print(f"Handpicked extrema at: \n"
              f"Indices: {[minAndMaxOrdered]}\n"
              f"Distance : {np.array(xshifted)[minAndMaxOrdered]} in {unitXY}")

    ax0.plot(np.array(xshifted)[minAndMaxOrdered], np.array(intensityProfileZoomConverted)[minAndMaxOrdered], "or", label='picked max- & minima')
    ax0.plot(xshifted, intensityProfileZoomConverted, '.', label=f'Time={timeFormat(elapsedtime)}', color=cmap(colorGradient[idxx]))  # plot the intensity profile

    # if FLIP:
    #     xshifted.reverse()
    #     np.flip(intensityProfileZoomConverted)
    #     minAndMaxOrdered = np.subtract(len(xshifted)-1,  minAndMaxOrdered)
    #     minAndMaxOrdered = np.sort(minAndMaxOrdered)
    # TODO below was set to 0 before?
    hdry = dryBrushThickness
    h = []
    xrange = []
    if len(minAndMaxOrdered) > 1:  # if at least 2 extrema are found
        # evaluate before first extremum: before index 0
        # between all extrema: between indices 0 - (len(extrema)-1)
        # after last extremum: after (len(extrema)-1)
        for i in range(0, len(minAndMaxOrdered) - 1):  # iterating from the first to the first-to-last extremum
            extremum1 = minAndMaxOrdered[i]
            extremum2 = minAndMaxOrdered[i + 1]
            # to calculate profile before first extremum
            if i == 0:  # calculate profile before first extremum
                dataI = np.divide(np.array(intensityProfileZoomConverted)[0:extremum2+1],
                                  normalizeFactor)  # intensity (y) data
                datax = np.array(xshifted)[0:extremum2+1]  # time (x) data
                # Below: calculate heights of[0 : Extremum1]. Resulting h will not start at 0, because index=0 does not start at an extremum, so must be corrected for.
                h_newrange = idkPre1stExtremum(datax, dataI, extremum1,
                                               extremum2)  #TODO check: removed -1 # do some -1 stuff because f how indexes work when parsing
                h_newrange = np.subtract(h_newrange, h_newrange[
                    0])  # substract value at index0 from all heights since the programn assumed the height to start at 0 (but it doesn't since we did not tsart at an extremum)

                # TODO below: this is not necesairy for this analysis I think. At the left, we don't known/need to know the height in advance
                # TODO: Set height at index 0 just to 0, and later

                # adjust entire h_newrange by stitching last value of h_newrange to height of first extremum
                # estimate from known dry height at what thickness the first extremum is.
                # in case of a maximum: 181.1*N
                # in case of minimum: 90.9 + 181.1*N
                if dataI[extremum1] - dataI[0] > 0:  # if etrx1 > data[0], next up is a maximum
                    maximas = np.arange(0, 181.1 * 20, 181.1)
                    diff_maximas = np.abs(np.subtract(maximas, hdry))
                    maxIndex = np.where(diff_maximas == min(diff_maximas))
                    h_1stextremum = maximas[maxIndex]
                else:
                    minima = np.arange(90.9, 90.9 + 181.1 * 20, 181.1)
                    diff_minima = np.abs(np.subtract(minima, hdry))
                    minIndex = np.where(diff_minima == min(diff_minima))
                    h_1stextremum = minima[minIndex]

                print(f"Calculated extremum: {h_1stextremum}")
                # TODO adusting normal code: set h_1st extremum to the last value of the calculated height profile
                # this just makes for a smooth profile, which should then start at h=0?
                h_1stextremum = h_newrange[-1]
                print(f"But using extremum: {h_1stextremum}")
                diff_hExtremumAndFinalnewRange = np.subtract(h_1stextremum, h_newrange[
                    -1])  # substract h of extremum with last value of calculated height
                h_newrange = np.add(h_newrange,
                                    diff_hExtremumAndFinalnewRange)  # add that difference to all calculated heights to stich profiles together
                xrange = np.concatenate([xrange, np.array(xshifted)[0:extremum1]])
                h = np.concatenate([h, h_newrange[:-1]])  # main output if this part: height profile before first extremum WITHOUT the height of the last extremum (which will be added in the next iteration)

            # to calculate profiles in between extrema
            dataI = np.divide(np.array(intensityProfileZoomConverted)[extremum1:extremum2+1],
                              normalizeFactor)  # intensity (y) data
            datax = np.array(xshifted)[extremum1:extremum2+1]  # time (x) data
            h_newrange = np.add(idk(datax, dataI, 0, len(datax) - 1), h_1stextremum + i * 90.9)
            xrange = np.concatenate([xrange, datax[:-1]])        #TODO datax[1:] ??
            h = np.concatenate([h, h_newrange[:-1]])

            # Once the first-to-last maximum is reached, above the profile between first-to-last and last extremum is calculated.
            # Below, the profile after last extremum is calculated
            if i == len(minAndMaxOrdered) - 2:  # -2 because this then happens after effectively the last extremum
                # input data ranging from the first extremum
                dataI = np.divide(np.array(intensityProfileZoomConverted)[0:len(xshifted)],
                                  normalizeFactor)  # intensity (y) data
                datax = np.array(xshifted)[0:len(xshifted)]  # time (x) data
                # Below: calculate heights of[Extremum2:end].
                ###TODO check if (i) or (i+1), beforehand (i+1) worked, now not?
                h_newrange = np.add(idkPostLastExtremum(datax, dataI, extremum1, extremum2),
                                    h_1stextremum + (i + 1) * 90.9)  # TODO removed: "do some -1 stuff with the extremum1&2 because f how indexes work when parsing"
                # xrange = np.concatenate([xrange, datax])
                xrange = np.concatenate([xrange, np.array(xshifted)[extremum2:len(xshifted)]])
                h = np.concatenate([h, h_newrange])

        # once entire height profile is calculated, convert to 'correct' height profile
        if FLIP:
            # first plot the data upside down, to have the height more swollen on the left
            h = -np.subtract(h, max(h))
        # then, correct height with a 'known' height somewhere. Can be dry height in dry region, or from a known height vs. time curve at a pixellocation
        diffh = knownHeightArr[idxx] - h[knownPixelPosition]
        print(
            f"Correcting height with {diffh} nm, because known height= {knownHeightArr[idxx]}, and calculated height= {h[knownPixelPosition]}")
        h = np.add(h, diffh)

        h_ratio = np.divide(h, hdry)
        ax1.set_xlabel(f"Distance of chosen range ({unitXY})")
        if PLOTSWELLINGRATIO:
            ax1.set_ylabel("Swelling ratio (h/h$_{0}$)")
            ax1.plot(xrange, h_ratio, label=f'Time={timeFormat(elapsedtime)}', color=cmap(colorGradient[idxx]))
            ax1.set_title(f"Swelling profiles in pixelrange {range1}:{range2}\nImgNr={idx}, VectorNr={vectorNumber}")
            ax1.set_title(f"Calibrated swelling profiles")
            ax1.plot(xrange[knownPixelPosition], h_ratio[knownPixelPosition], 'ok', markersize=9)
            ax1.plot(xrange[knownPixelPosition], h_ratio[knownPixelPosition], 'o', color=cmap(colorGradient[idxx]))
        else:
            ax1.set_ylabel("Film thickness (nm)")
            ax1.plot(xrange, h, label=f'Time={timeFormat(elapsedtime)}', color=cmap(colorGradient[idxx]))
            # ax1.set_title(f"Height profile at time: {timeFormat(elapsedtime)} in pixelrange {range1}:{range2}")
            ax1.set_title(f"Calibrated height profiles\nImgNr={idx}, VectorNr={vectorNumber}")
            ax1.plot(xrange[knownPixelPosition], h[knownPixelPosition], 'ok', markersize=9)
            ax1.plot(xrange[knownPixelPosition], h[knownPixelPosition], 'o', color=cmap(colorGradient[idxx]))
        ax1.legend(loc='upper right')
        ax1.autoscale(enable=True, axis='x', tight=True)
        ax1.autoscale(enable=True, axis='y', tight=True)
        if PLOTSWELLINGRATIO:
            ax1.set_ylim(bottom=0.9)
        else:
            ax1.set_ylim(bottom=0)
        fig1.tight_layout()

        print(f"Average thickness Left Side Image = {np.mean(h[1:50]):.2f}, Right Side = {np.mean(h[-50:-1]):.2f} (over 50 points)")

        # Saves data in time vs height profile plot so a csv file.
        wrappedPath = os.path.join(source,
                                   f"Swellingimages\\data{timeFormat(elapsedtime)}_anchor{knownPixelPosition}_PureIntensity.csv")
        d = dict(
            {f'xshifted ({unitXY})': xshifted, 'Intensity converted (-)': intensityProfileZoomConverted, f'xrange ({unitXY})': xrange,
             'height (nm)': h, 'Swelling ratio (-)': h_ratio})
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))  # pad shorter colums with NaN's
        df.to_csv(wrappedPath, index=False)
        # np.savetxt(wrappedPath, [p for p in zip_longest('xshifted (mm)', 'Insensity converted (-)', 'xrange (mm)', 'height (nm)', 'Swelling ratio (-)', fillvalue='')], delimiter=',', fmt='%s')
        # np.savetxt(wrappedPath, [p for p in zip_longest(xshifted, intensityProfileZoomConverted, xrange, h, h_ratio, fillvalue='')], delimiter=',', fmt='%s')
    else:
        print(f"No minimum and maximum were found. Only a single extremum")
    ax0.legend(loc='upper right')
    ax0.set_ylabel("Intensity (-)")
    ax0.set_xlabel(f"Distance of chosen range ({unitXY})")
    ax0.set_title(f"Intensity profile\n ImgNr={idx}, VectorNr={vectorNumber}")
    if SAVEFIG:
        ax0.autoscale(enable=True, axis='x', tight=True)
        fig0.savefig(os.path.join(source, f"Swellingimages\\n{idx}_k{vectorNumber}_Intensity.png"), dpi=300)
        fig1.savefig(os.path.join(source, f"Swellingimages\\HeightProfile{timeFormat(elapsedtime)}_k{vectorNumber}.png"), dpi=300)
    if SEPERATEPLOTTING:
        plt.close(fig1)
        plt.close(fig0)
    return h, h_ratio

def main():
    pass

if __name__ == "__main__":
    main()
    exit()