"""
Attempts at quali- & quantitatively calculating/fitting/plotting CA_apparent using the Cox-Voinov equation
𝜃_𝑎^3−𝜃_𝑒𝑞^3=9𝐶𝑎 𝑙𝑛 𝑅/𝑙    with 𝐶𝑎=𝜇 𝑣_(𝑝ℎ𝑖)/𝜎,

AFAIK (best) working functions:
        tiltedDropQualitative()         #using this one to model tilted droplets
        movingDropQualitative_fitting() #Using this one to FIT moving droplets!
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import csv
import json
import scipy.optimize
import traceback

from Testing_ToCollapseCode import manualFitting

def importData():
    # define img nr &
    # define folder path
    imgNr = 32
    folder_path = "G:\\2024_05_07_PLMA_Basler15uc_Zeiss5x_dodecane_Xp1_31_S2_WEDGE_2coverslip_spacer_V3\\Analysis CA Spatial"
    #imgNr = 47
    #folder_path = "D:\\2024-09-04 PLMA dodecane Xp1_31_2 ZeissBasler15uc 5x M3 tilted drop\\Analysis CA Spatial"

    #folder_path = os.path.join(os.getcwd(), "TestData")
    # import CA datafile paths
    file_paths = [os.path.join(folder_path, f"ContactAngleData {imgNr}.csv")]

    # import middle coord
    with open(os.path.join(folder_path, f"Analyzed Data\\{imgNr}_analysed_data.json"), 'r') as file:
        json_data = json.load(file)
    middleCoord = json_data['middleCoords-surfaceArea']

    # import data from csv datafiles
    xcoord = []
    ycoord = []
    CA = []
    for file_path in file_paths:
        with open(file_path, newline='') as csvfile:
            data = csv.reader(csvfile, delimiter=',')
            for row in data:
                try:
                    xcoord.append(int(row[0]))
                    ycoord.append(int(row[1]))
                    CA.append(float(row[2]))
                except:
                    print(f"some error: row info = {row}")
    return xcoord, ycoord, CA

# Plot spatial experimental CA profile vs X,Y-Coords
# TODO uncomment
# fig5, ax5 = plt.subplots(figsize=(9,6))
# im5 = ax5.scatter(xcoord, ycoord, c=CA, cmap='jet', vmin=min(CA), vmax=max(CA))
# ax5.set(xlabel = "X-coord", ylabel = "Y-Coord");
# ax5.set_title(f"Experimental data\nSpatial Apparent Contact Angles Colormap", fontsize = 20)

def coordsToPhi(xArrFinal, yArrFinal, medianmiddleX, medianmiddleY):
    """
    :return phi: range [-pi : pi]
    :return rArray: distance from the middle to the coordinate. UNIT= same as input units (so probably pixel, or e.g. mm)

    phi = 0 at right side -> 0.5pi at top -> 1pi at left = -1pi at left -> -0.5pi at bottom
    example how phi evolves: https://stackoverflow.com/questions/17574424/how-to-use-atan2-in-combination-with-other-radian-angle-systems
    """
    dx = np.subtract(xArrFinal, medianmiddleX)
    dy = np.subtract(yArrFinal, medianmiddleY)
    phi = np.arctan2(dy, dx)  # angle of contour coordinate w/ respect to 12o'clock (radians)
    rArray = np.sqrt(np.square(dx) + np.square(dy))
    return phi, rArray


def convertPhiToazimuthal(phi):
    """
    :param phi: np.array of atan2 radians   [-pi, pi]
    :return np.sin(phi2): np.array of azimuthal angle [-1, 1] rotating clockwise, starting at with 0 at top
    :return phi2: np.array of normal radians [0, 2pi] rotating clockwise starting at top.
    """
    phi1 = 0.5 * np.pi - phi
    phi2 = np.mod(phi1, (
                2.0 * np.pi))  # converting atan2 to normal radians: https://stackoverflow.com/questions/17574424/how-to-use-atan2-in-combination-with-other-radian-angle-system
    return np.sin(phi2), phi2


def fitSpatialCA(xcoord, ycoord, CA, middleCoord):
    # Convert x,y-coords to radial angle (phi)
    phi, rArray = coordsToPhi(xcoord, ycoord, middleCoord[0], 4608 - middleCoord[1])
    azimuthalX, phi_normalRadians = convertPhiToazimuthal(phi)

    # TODO uncomment
    # fig6, ax6 = plt.subplots()
    # ax6.plot(azimuthalX, CA, marker='.')
    # plt.show()

    # Ca is variable over phi, because of Ca = mu * v(phi!) / gamma : in which mu=viscosity, v=velocity (phi dependent + or -), gamma=surface tension
    # dodecane mu=1.34mPas       gamma = 25.55mN/m       #v is to be in m/s
    mu = 1.34 / 1000  # Pa*s
    gamma = 25.55 / 1000  # N/m
    # TODO for now let's say v = 50um/min
    v = 50 * 1E-6 / 60  # m/s
    CapNr = lambda v, mu, gamma: mu * v / gamma

    theta_app = lambda theta_eq, Ca, R, l: (9 * Ca * np.log(R / l) + theta_eq ** 3) ** (1 / 3)

    anglerange = np.linspace(0, 1, 1000)  # 0-1
    angle = np.linspace(-np.pi, np.pi, 1000)        #-pi - pi
    k = 3  # power order for 'steepness' of sin curve

    #    f_theta_eq = lambda anglerange, k: (((0.5+np.sin(anglerange*np.pi-np.pi/2)/2)**((2*(1-anglerange))**k))*2 + 1) * np.pi / 180           #base function - [1-3], so around 2+-1
    #Define 'input' theta_eq values for the Cox-Voinov equation. <- derived from experimental data, min&maxima where friction had least influnec
    #Also variation of theta_eq is not defined as a normal sinus, but with a kink (intended because of non-linear swelling gradient under/outside cover)
    CA_eq_adv = 1.875;
    CA_eq_rec = 1.725  # CA [deg] values observed from spatial CA profile: local max & minima on location where friction should not play big of a role - basically the max & min CA values
    Ca_eq_mid = (CA_eq_adv + CA_eq_rec) / 2  # 'middle' CA value around which we build the sinus-like profile
    Ca_eq_diff = CA_eq_adv - Ca_eq_mid  # difference between middle CA value & the 'eq' ones - for in the sinus-like function
    f_theta_eq = lambda anglerange, k: (((0.5 + np.sin(anglerange * np.pi - np.pi / 2) / 2) ** (
                (2 * (1 - anglerange)) ** k)) * 2 - 1) * np.pi / 180  # base function - [-1,1], so around 0+-1
    theta_eq_cover = np.flip(f_theta_eq(anglerange[:len(anglerange) // 2], 0))  # under cover = less steep
    theta_eq_open = np.flip(f_theta_eq(anglerange[len(anglerange) // 2:], 3))  # open air = steep
    theta_eq = np.concatenate([theta_eq_open, theta_eq_cover]) * 180 / np.pi
    # perform operation to shift CA values to desired CA_eq range
    theta_eq = theta_eq * Ca_eq_diff + Ca_eq_mid

    # Calculate theta_apparent:
    prefactor = 9  # OG = 9
    R = 1E-6  # slip length, 10 micron?                     -macroscopic -
    l = 3E-9  # capillary length, ongeveer              -microscopic
    Ca = 4.8E-8 * np.sin(angle - np.pi / 2)  # OG standard Ca curve: normal sinus between + and - the value.  OG = -1.55E-7 *
    print(f"Calculated R/l: {R/l}, ln(R/l): {np.log(R/l)}")
    print(f"Calculated Cap.Nr's from CA_adv, _mid & _rec: {CapNr(52 * 1E-6 / 60, mu, gamma):.2E}, {CapNr(55 * 1E-6 / 60, mu, gamma):.2E}, {CapNr(140 * 1E-6 / 60, mu, gamma):.2E}")
    print(f"Calculated 9Ca*ln(R/l): {prefactor * max(Ca) * np.log(R / l)}")
    theta_app = ((theta_eq / 180 * np.pi) ** 3 + prefactor * Ca * np.log(R / l)) ** (1 / 3) * 180 / np.pi  # [deg]

    # plot eq & apparent contact angle vs azimuthal angle
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(angle * 180 / np.pi, theta_eq, label=r'$\theta_{eq}$ - no friction')
    ax1.plot(angle * 180 / np.pi, theta_app, '.', label=r'$\theta_{app}$ - with friction')
    ax1.plot(azimuthalX * 180, CA, '.', label=r'$\theta$ - experimental')

    ax1.set(xlabel='Azimuthal angle (deg)', ylabel='Contact angle (deg)',
            title='Example influence hydrolic resistance on apparent contact angle')
    ax1.legend(loc='best')
    # fig1.savefig(f"C:\\TEMP\\CA vs azimuthal Ca={max(Ca)}, x={x}, l={l}.png", dpi=600)
    plt.show()

    # fig7, ax7 = plt.subplots()
    # ax7.plot(anglerange, theta_eq, marker='.')
    # ax7.set(title = 'new CA Eq profile', xlabel = 'azimuthal angle [deg]', ylabel = 'CA angle [deg]')
    # plt.show()

    # plot eq (NO FRICTION) contact angle vs 'spatial X,Y-coordinates'
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    xArrFinal = np.cos(anglerange * np.pi)
    yArrFinal = np.sin(anglerange * np.pi)
    im3 = ax3.scatter([xArrFinal, np.flip(xArrFinal)], [yArrFinal, -np.flip(yArrFinal)],
                      c=[np.flip(theta_eq), theta_eq], cmap='jet', vmin=min(theta_eq), vmax=max(theta_eq))
    ax3.set_xlabel("X-coord");
    ax3.set_ylabel("Y-Coord");
    ax3.set_title(f"Model: No Hydrolic Resistance \nSpatial Equilibrium Contact Angles Colormap", fontsize=20)
    fig3.colorbar(im3)
    # fig3.savefig("C:\\TEMP\\NOhydrolic.png", dpi=600)

    # plot apparent (W/ FRICTION) contact angle vs 'spatial X,Y-coordinates'
    fig4, ax4 = plt.subplots(figsize=(9, 6))
    im4 = ax4.scatter([xArrFinal, np.flip(xArrFinal)], [yArrFinal, -np.flip(yArrFinal)],
                      c=[np.flip(theta_app), theta_app], cmap='jet', vmin=min(theta_app), vmax=max(theta_app))
    ax4.set_xlabel("X-coord");
    ax4.set_ylabel("Y-Coord");
    ax4.set_title(f"Model: Effect of viscous friction\nSpatial Predicted Apparent Contact Angles Colormap", fontsize=20)
    fig4.colorbar(im4)
    # fig4.savefig(f"C:\\TEMP\\YEShydrolic Ca={max(Ca)}, x={x}, l={l}.png", dpi=600)

    return

def fitSpatialCA_simplified(xcoord, ycoord, CA, middleCoord):

    #theta_app = (theta_eq^3 + someFactor)^0.333
    #in which someFactor varies between [-varyingfactor, varyingfactor] (velocity: - -> + in moving direction.). someFactor varies for now with a sinus curve
    f_theta_app_simple = lambda theta_eq, varyingfactor: np.power(np.power(theta_eq, 3) + varyingfactor, 1/3)         #Cox-Voinov equation, in this 'varyingfactor' is the combination of '9Ca ln(R/l)'

    phi, rArray = coordsToPhi(xcoord, ycoord, middleCoord[0], 4608 - middleCoord[1])
    azimuthalX, phi_normalRadians = convertPhiToazimuthal(phi)

    anglerange = np.linspace(0, 1, len(CA))  # 0-1
    angle = np.linspace(-np.pi, np.pi, len(CA))  # -pi - pi

    #Define 'input' theta_eq values for the Cox-Voinov equation. <- derived from experimental data, min&maxima where friction had least influnec
    #Also variation of theta_eq is not defined as a normal sinus, but with a kink (intended because of non-linear swelling gradient under/outside cover)
    CA_eq_adv = 1.875;
    CA_eq_rec = 1.725  # CA [deg] values observed from spatial CA profile: local max & minima on location where friction should not play big of a role - basically the max & min CA values
    Ca_eq_mid = (CA_eq_adv + CA_eq_rec) / 2  # 'middle' CA value around which we build the sinus-like profile
    Ca_eq_diff = CA_eq_adv - Ca_eq_mid  # difference between middle CA value & the 'eq' ones - for in the sinus-like function

    #Function to vary between CA_max&_min with a sinus-like shape.
    #Input: anglerange = a range between [0, 1]
    #       k = int value. Defines steepness of sinus-like curve
    f_theta_eq = lambda anglerange, k: ((
                                    np.power(0.5 + np.sin(anglerange * np.pi - np.pi / 2) / 2,
                                    np.power(2 * (1 - anglerange), k) )
                                    ) * 2 - 1) * np.pi / 180  # base function - [-1,1], so around 0+-1


    theta_eq_cover = np.flip(f_theta_eq(anglerange[:len(anglerange) // 2], 0))  # under cover = less steep
    theta_eq_open = np.flip(f_theta_eq(anglerange[len(anglerange) // 2:], 3))  # open air = steep
    theta_eq = np.concatenate([theta_eq_open, theta_eq_cover]) * 180 / np.pi
    # perform operation to shift CA values to desired CA_eq range
    theta_eq = theta_eq * Ca_eq_diff + Ca_eq_mid

    #fit theta_app function to experimental data
    popt, pcov = scipy.optimize.curve_fit(f_theta_app_simple, theta_eq, CA, bounds = (0, 1E-05))

    # plot eq & apparent contact angle vs azimuthal angle
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(angle * 180 / np.pi, theta_eq, label=r'$\theta_{eq}$ - no friction')
    ax1.plot(angle * 180 / np.pi, f_theta_app_simple(theta_eq, *popt), '.', label=r'$\theta_{app}$ - with friction')
    ax1.plot(azimuthalX * 180, CA, '.', label=r'$\theta$ - experimental')

    ax1.set(xlabel='Azimuthal angle (deg)', ylabel='Contact angle (deg)',
            title='Example influence hydrolic resistance on apparent contact angle')
    ax1.legend(loc='best')
    # fig1.savefig(f"C:\\TEMP\\CA vs azimuthal Ca={max(Ca)}, x={x}, l={l}.png", dpi=600)
    plt.show()

    return

def trial1(xcoord, ycoord, CA, middleCoord):
    fig3, ax3 = plt.subplots(2,2, figsize=(18, 12))
    phi, rArray = coordsToPhi(xcoord, ycoord, middleCoord[0], 4608 - middleCoord[1])
    azimuthalX, phi_normalRadians = convertPhiToazimuthal(phi)

    anglerange = np.linspace(0, 1, len(CA))  # 0-1
    angle = np.linspace(-np.pi, np.pi, len(CA))  # -pi - pi

    #Define 'input' theta_eq values for the Cox-Voinov equation. <- derived from experimental data, min&maxima where friction had least influnec
    #Also variation of theta_eq is not defined as a normal sinus, but with a kink (intended because of non-linear swelling gradient under/outside cover)
    CA_eq_adv = 1.875;
    CA_eq_rec = 1.725  # CA [deg] values observed from spatial CA profile: local max & minima on location where friction should not play big of a role - basically the max & min CA values
    Ca_eq_mid = (CA_eq_adv + CA_eq_rec) / 2  # 'middle' CA value around which we build the sinus-like profile
    Ca_eq_diff = CA_eq_adv - Ca_eq_mid  # difference between middle CA value & the 'eq' ones - for in the sinus-like function

    #Function to vary between CA_max&_min with a sinus-like shape.
    #Input: anglerange = a range between [0, 1]
    #       k = int value. Defines steepness of sinus-like curve
    f_theta_eq = lambda anglerange, k: ((
                                    np.power(0.5 + np.sin(anglerange * np.pi - np.pi / 2) / 2,
                                    np.power(2 * (1 - anglerange), k) )
                                    ) * 2 - 1) * np.pi / 180  # base function - [-1,1], so around 0+-1

    theta_eq_cover = np.flip(f_theta_eq(anglerange[:len(anglerange) // 2], 0))  # under cover = less steep
    theta_eq_open = np.flip(f_theta_eq(anglerange[len(anglerange) // 2:], 3))  # open air = steep
    theta_eq = np.concatenate([theta_eq_open, theta_eq_cover]) * 180 / np.pi
    # perform operation to shift CA values to desired CA_eq range
    theta_eq = theta_eq * Ca_eq_diff + Ca_eq_mid

    # plot eq & apparent contact angle vs azimuthal angle
    ax3[0,0].plot(angle / np.pi, theta_eq, label=r'$\theta_{eq}$ - no friction')
    ax3[0,0].plot(azimuthalX, CA, '.', label=r'$\theta$ - experimental azi')

    ax3[0,0].set(xlabel=r'Azimuthal angle ($\pi$)', ylabel='Contact angle (deg)',
            title='Example influence hydrolic resistance on apparent contact angle')
    ax3[0,0].legend(loc='best')



    varyFactor = np.power(np.array(CA) / 180 * np.pi, 3) - np.power(np.array(theta_eq) / 180 * np.pi, 3)

    #values for R & l: https://www.sciencedirect.com/science/article/pii/S1359029422000139?via%3Dihub
    R = 100E-6  # capillary length. slip length. >10 micron? 2.7mm for water      -macroscopic -
    l = 3E-9  # e.g. 1nm              -microscopic/molecular length scale
    Ca = varyFactor / 9 / np.log(R / l)

    mu = 1.34 / 1000  # Pa*s
    gamma = 25.55 / 1000  # N/m
    local_velocity = Ca * gamma / mu

    phi_sorted, local_velocity_sorted = [list(a) for a in zip(*sorted(zip(phi, local_velocity)))]
    for i in range(1, len(phi_sorted)):
        if phi_sorted[i] <= phi_sorted[i - 1]:
            phi_sorted[i] = phi_sorted[i - 1] + 1e-5
    #Fit velocity profile with a normal sine wave:
    f_local_velocity, func_single, N, _, _ = manualFitting(np.array(phi_sorted), np.array(local_velocity_sorted) * 60 * 1E6, f"C:\\TEMP", [r"Local Velocity", "[$\mu$m/min]"],
                                                     [5], 'manual')

    phi_range = np.linspace(-np.pi, np.pi, len(CA))
    # def func(x, a, b, c):
    #     return a * np.sin(x+b) + c
    #
    # popt, pcov = scipy.optimize.curve_fit(func, phi, local_velocity)

    ax3[0,1].plot(phi, local_velocity * 60 * 1E6, '.')
    #ax3[0,1].plot(phi, func(phi, *popt) * 60 * 1E6, '.')
    ax3[0,1].plot(phi_range, f_local_velocity(phi_range), '.')
    ax3[0,1].set(xlabel = r'radial angle ($\phi$)', ylabel = r'local velocity ($\mu$m/min)', title = 'Local velocity plot')

    Ca_calculated = (np.array(f_local_velocity(phi_range)) * mu / gamma) / (60 * 1E6)
    theta_app_calculated = np.power(np.power(theta_eq / 180 * np.pi, 3) + 9 * Ca_calculated * np.log(R/l), 1/3)
    ax3[0,0].plot(phi_range / np.pi, theta_app_calculated * 180 / np.pi, '.', label=r'$\theta_{app}$ - modelled')
    ax3[0, 0].legend(loc='best')


    xArrFinal = np.cos(phi)
    yArrFinal = np.sin(phi)
    im3 = ax3[1,1].scatter(xArrFinal, yArrFinal, c=local_velocity * 60 * 1E6, cmap='jet', vmin=min(local_velocity * 60 * 1E6), vmax=max(local_velocity * 60 * 1E6))
    ax3[1,1].set_xlabel("X-coord");
    ax3[1,1].set_ylabel("Y-Coord");
    ax3[1,1].set_title(f"Model: spatial local velocity colormap", fontsize=20)
    fig3.colorbar(im3)

    #plot spatial CA image
    im3 = ax3[1,0].scatter(xArrFinal, yArrFinal, c=CA, cmap='jet', vmin=min(CA), vmax=max(CA))
    ax3[1,0].set_xlabel("X-coord");
    ax3[1,0].set_ylabel("Y-Coord");
    ax3[1,0].set_title(f"Model: spatial Contact angle colormap", fontsize=20)
    fig3.colorbar(im3)

    plt.show()


    return



#TODO: fit experimental data to Cox-Voinov
def tiltedDrop(xcoord, ycoord, CA, middleCoord):
    """
    For tilted drops specifically.
    Fit Cox-Voinov Ca_app to experimental data with an inputted theta_eq, Capillary number (Ca(phi)) profile along CL (local velocity depends on phi),
    and R/l value.

    :param xcoord:
    :param ycoord:
    :param CA:
    :param middleCoord:
    :return:
    """
    fig3, ax3 = plt.subplots(2, 2, figsize=(18, 12))
    phi_exp, _ = coordsToPhi(xcoord, ycoord, middleCoord[0], 4608 - middleCoord[1])
    azi_exp, rad_exp = convertPhiToazimuthal(phi_exp)

#TODO from here onwards
    anglerange = np.linspace(0, 1, len(CA))  # 0-1
    angle = np.linspace(-np.pi, np.pi, len(CA))  # -pi - pi

    # Define 'input' theta_eq values for the Cox-Voinov equation. <- derived from experimental data, min&maxima where friction had least influnec
    # Also variation of theta_eq is not defined as a normal sinus, but with a kink (intended because of non-linear swelling gradient under/outside cover)
    CA_eq = 2.75        #eq CA, in case of flat droplet.

    # CA_eq_adv = 1.875;
    # CA_eq_rec = 1.725  # CA [deg] values observed from spatial CA profile: local max & minima on location where friction should not play big of a role - basically the max & min CA values
    # Ca_eq_mid = (CA_eq_adv + CA_eq_rec) / 2  # 'middle' CA value around which we build the sinus-like profile
    # Ca_eq_diff = CA_eq_adv - Ca_eq_mid  # difference between middle CA value & the 'eq' ones - for in the sinus-like function

    # Function to vary between CA_max&_min with a sinus-like shape.
    # Input: anglerange = a range between [0, 1]
    #       k = int value. Defines steepness of sinus-like curve
    f_theta_eq = lambda anglerange, k: ((
                                            np.power(0.5 + np.sin(anglerange * np.pi - np.pi / 2) / 2,
                                                     np.power(2 * (1 - anglerange), k))
                                        ) * 2 - 1) * np.pi / 180  # base function - [-1,1], so around 0+-1

    theta_eq_cover = np.flip(f_theta_eq(anglerange[:len(anglerange) // 2], 0))  # under cover = less steep
    theta_eq_open = np.flip(f_theta_eq(anglerange[len(anglerange) // 2:], 3))  # open air = steep
    theta_eq = np.concatenate([theta_eq_open, theta_eq_cover]) * 180 / np.pi
    # perform operation to shift CA values to desired CA_eq range
    theta_eq = theta_eq * Ca_eq_diff + Ca_eq_mid

    # plot eq & apparent contact angle vs azimuthal angle
    ax3[0, 0].plot(angle / np.pi, theta_eq, label=r'$\theta_{eq}$ - no friction')
    ax3[0, 0].plot(azimuthalX, CA, '.', label=r'$\theta$ - experimental azi')

    ax3[0, 0].set(xlabel=r'Azimuthal angle ($\pi$)', ylabel='Contact angle (deg)',
                  title='Example influence hydrolic resistance on apparent contact angle')
    ax3[0, 0].legend(loc='best')

    varyFactor = np.power(np.array(CA) / 180 * np.pi, 3) - np.power(np.array(theta_eq) / 180 * np.pi, 3)

    #values for R & l: https://www.sciencedirect.com/science/article/pii/S1359029422000139?via%3Dihub
    R = 100E-6  # capillary length. slip length. >10 micron? 2.7mm for water      -macroscopic -
    l = 3E-9  # e.g. 1nm              -microscopic/molecular length scale
    Ca = varyFactor / 9 / np.log(R / l)

    mu = 1.34 / 1000  # Pa*s
    gamma = 25.55 / 1000  # N/m
    local_velocity = Ca * gamma / mu

    phi_sorted, local_velocity_sorted = [list(a) for a in zip(*sorted(zip(phi, local_velocity)))]
    for i in range(1, len(phi_sorted)):
        if phi_sorted[i] <= phi_sorted[i - 1]:
            phi_sorted[i] = phi_sorted[i - 1] + 1e-5
    # Fit velocity profile with a normal sine wave:
    f_local_velocity, func_single, N, _, _ = manualFitting(np.array(phi_sorted),
                                                           np.array(local_velocity_sorted) * 60 * 1E6, f"C:\\TEMP",
                                                           [r"Local Velocity", "[$\mu$m/min]"],
                                                           [5], 'manual')

    phi_range = np.linspace(-np.pi, np.pi, len(CA))
    # def func(x, a, b, c):
    #     return a * np.sin(x+b) + c
    #
    # popt, pcov = scipy.optimize.curve_fit(func, phi, local_velocity)

    ax3[0, 1].plot(phi, local_velocity * 60 * 1E6, '.')
    # ax3[0,1].plot(phi, func(phi, *popt) * 60 * 1E6, '.')
    ax3[0, 1].plot(phi_range, f_local_velocity(phi_range), '.')
    ax3[0, 1].set(xlabel=r'radial angle ($\phi$)', ylabel=r'local velocity ($\mu$m/min)', title='Local velocity plot')

    Ca_calculated = (np.array(f_local_velocity(phi_range)) * mu / gamma) / (60 * 1E6)
    theta_app_calculated = np.power(np.power(theta_eq / 180 * np.pi, 3) + 9 * Ca_calculated * np.log(R / l), 1 / 3)
    ax3[0, 0].plot(phi_range / np.pi, theta_app_calculated * 180 / np.pi, '.', label=r'$\theta_{app}$ - modelled')
    ax3[0, 0].legend(loc='best')

    xArrFinal = np.cos(phi)
    yArrFinal = np.sin(phi)
    im3 = ax3[1, 1].scatter(xArrFinal, yArrFinal, c=local_velocity * 60 * 1E6, cmap='jet',
                            vmin=min(local_velocity * 60 * 1E6), vmax=max(local_velocity * 60 * 1E6))
    ax3[1, 1].set_xlabel("X-coord");
    ax3[1, 1].set_ylabel("Y-Coord");
    ax3[1, 1].set_title(f"Model: spatial local velocity colormap", fontsize=20)
    fig3.colorbar(im3)

    # plot spatial CA image
    im3 = ax3[1, 0].scatter(xArrFinal, yArrFinal, c=CA, cmap='jet', vmin=min(CA), vmax=max(CA))
    ax3[1, 0].set_xlabel("X-coord");
    ax3[1, 0].set_ylabel("Y-Coord");
    ax3[1, 0].set_title(f"Model: spatial Contact angle colormap", fontsize=20)
    fig3.colorbar(im3)

    plt.show()
    return

#WORKING!
def tiltedDropQualitative():
    """
    Model theta_app for a tilted droplet with 'known' values: input constant thetha_eq, R & L. Vary Ca(phi)
    𝜃_𝑎^3−𝜃_𝑒𝑞^3=9𝐶𝑎 𝑙𝑛 𝑅/𝑙
    with 𝐶𝑎=𝜇 𝑣_(𝑝ℎ𝑖)/𝜎
    :return:
    """
    nr_of_datapoints = 1000
    theta_eq_deg = np.ones(nr_of_datapoints) * 1.23            #deg
    v0 = 700 * 1E-6 / 60  #[m/s] assume same velocity at advancing and receding, just in opposite direction

    mu = 1.34 / 1000  # Pa*s
    gamma = 25.55 / 1000  # N/m
    R = 100E-6  # slip length, 10 micron?                     -macroscopic
    l = 2E-9  # capillary length, ongeveer              -micro/nanoscopic

    theta_eq_rad = theta_eq_deg / 180 * np.pi

    phi = np.linspace(-np.pi, np.pi, nr_of_datapoints)          #angle of CL position. 0 at 3'o clock, pi at 9'o clock. +pi/2 at 12'o clock, -pi/2 at 6'o clock.

    #assuming v(0) = -v(pi)
    #velocity_local = np.cos(phi) * v0        #v0 at advancing, -v0 at receding
    #correcting for v(0) < -v(pi)
    v_adv = 86 * 1E-6 / 60  #[m/s] assume same velocity at advancing and receding, just in opposite direction       [70]    (right side)
    v_rec = 40 * 1E-6 / 60  #[m/s] assume same velocity at advancing and receding, just in opposite direction      [150]    (left side)

    def targetExtremumCA_to_inputVelocity(CA_app: float, CA_eq: float, sigma: float, mu:float, R:float, l:float):
        """
        Tilted droplets:
        Calculate what the (maximum) velocity must be for the inputted CA_app at the advancing or receding location
        for tilted droplets.
        :param CA_app: target maximum or minimum CA at adv./rec. respectively [rad]
        :param CA_eq: input CA_eq [rad]
        :return: velocity [m/s] or [um/min]
        """
        velocity = ((np.power(CA_app, 3) - np.power(CA_eq, 3)) * sigma) / (9*mu * np.log(R/l))
        return velocity * 60 / 1E-6

    print(f"Calculated velocities are: {targetExtremumCA_to_inputVelocity(np.array([1.48, 1.07]) / 180 * np.pi, 1.23 / 180 * np.pi, gamma, mu, R, l)}")

    print(f"For water, CA_eq=60 velocities are: {targetExtremumCA_to_inputVelocity(np.array([63]) / 180 * np.pi, 60 / 180 * np.pi, 72/1000, 1.0016/1000, R, l)}")


    velocity_local = np.array([np.cos(phi_l) * v_adv if abs(phi_l) < np.pi/2 else np.cos(phi_l) * v_rec for phi_l in phi])
    fig2, ax2 = plt.subplots(1,2, figsize= (15, 9.6/1.5))
    ax2[0].plot(phi, velocity_local * 60 / 1E-6, linewidth=7)
    ax2[0].set(xlabel='radial angle [rad]', ylabel=f'local velocity [$\mu$m/min]', title='Input local velocity profile \n(adjusted for difference between adv. and rec. speed)')

    Ca_local = mu * velocity_local / gamma
    print(f"Ca = {max(Ca_local)}")
    if 0 > np.power(min(theta_eq_rad), 3) + (9 * min(Ca_local) * np.log(R/l)):
        logging.error(f"Some calculated CA's will be NaN: min(Ca) = {min(Ca_local)} +  min(theta_eq_rad) = {min(theta_eq_rad)} will be negative:"
        f"{np.power(min(theta_eq_rad), 3) + (9 * min(Ca_local) * np.log(R / l))}, ()^1/3 = {np.power(np.power(min(theta_eq_rad), 3) + 9 * min(Ca_local) * np.log(R/l), 1/3)} ")
    theta_app_calculated = np.power(np.power(theta_eq_rad, 3) + 9 * Ca_local * np.log(R/l), 1/3)
    theta_app_calculated_deg = theta_app_calculated * 180 / np.pi

    ax2[1].plot(phi, theta_app_calculated_deg, color='darkorange', linewidth=7)
    ax2[1].set(xlabel='radial angle [rad]', ylabel='CA$_{app}$ [deg]', title='Calculated apparent contact angle profile')
    fig2.tight_layout()

    R_drop = 1  # [mm]
    x_coord = R_drop * np.cos(phi)
    y_coord = R_drop * np.sin(phi)
    fig1, ax1 = plt.subplots(figsize= (12, 9.6))
    im1 = ax1.scatter(x_coord, y_coord, s = 45, c=theta_app_calculated_deg, cmap='jet', vmin=min(theta_app_calculated_deg), vmax=max(theta_app_calculated_deg))
    ax1.set_xlabel("X-coord"); ax1.set_ylabel("Y-Coord");
    ax1.set_title(f"Spatial Contact Angles Colormap Tilted Droplet\n Quantitative description", fontsize=20)
    fig1.colorbar(im1)
    fig1.tight_layout()

    fig2.savefig(os.path.join('C:\\Users\\ReuvekampSW\\Downloads', 'temp1.png'), dpi=600)
    fig1.savefig(os.path.join('C:\\Users\\ReuvekampSW\\Downloads', 'temp2.png'), dpi=600)

    plt.show()
    return

def movingDropQualitative():
    """
    Model theta_app for a moving droplet with 'known' values:
    input constant, R & L. Vary thetha_eq(phi) & Ca(phi).
    𝜃_𝑎^3−𝜃_𝑒𝑞^3=9𝐶𝑎 𝑙𝑛 𝑅/𝑙
    with 𝐶𝑎=𝜇 𝑣_(𝑝ℎ𝑖)/𝜎
    :return:
    """
    fig1, ax1 = plt.subplots(2, 2, figsize= (12, 9.6))
    nr_of_datapoints = 1000         #must be a multiple of 4!!
    if not nr_of_datapoints % 4 == 0:
        nr_of_datapoints = nr_of_datapoints + (4-(nr_of_datapoints % 4))
        logging.warning(f"nr_of_datapoints is not a multiple of 4: changing it to = {nr_of_datapoints}")

    # Define 'input' theta_eq values for the Cox-Voinov equation. <- derived from experimental data, min&maxima where friction had least influnec
    # Also variation of theta_eq is not defined as a normal sinus, but with a kink (intended because of non-linear swelling gradient under/outside cover)
    CA_eq_adv = 1.20;
    CA_eq_rec = 1.8  # CA [deg] values observed from spatial CA profile: local max & minima on location where friction should not play big of a role - basically the max & min CA values

    #Input velocities at advancing & receding point.
    v_adv = 150 * 1E-6 / 60  #[m/s] assume same velocity at advancing and receding, just in opposite direction       [70]    (right side)
    v_rec = 150 * 1E-6 / 60  #[m/s] assume same velocity at advancing and receding, just in opposite direction      [150]    (left side)

    mu = 1.34 / 1000  # Pa*s
    gamma = 25.55 / 1000  # N/m
    R = 100E-6  # slip length, 10 micron?                     -macroscopic
    l = 2E-9  # capillary length, ongeveer              -micro/nanoscopic

    #print(f"For water, CA_eq=60 velocities are: {targetExtremumCA_to_inputVelocity(np.array([63]) / 180 * np.pi, 60 / 180 * np.pi, 72/1000, 1.0016/1000, R, l)}")

    #Input target advancing and receding CA (outer moving droplet) to calculate required local velocities
    v_adv, v_rec = targetExtremumCA_to_inputVelocity(np.array([1.47, 1.43]) / 180 * np.pi, np.array([CA_eq_adv, CA_eq_rec]) / 180 * np.pi, gamma, mu, R, l)
    print(f"Calculated velocities are for max&min: {targetExtremumCA_to_inputVelocity(np.array([1.47, 1.43]) / 180 * np.pi, np.array([CA_eq_adv, CA_eq_rec]) / 180 * np.pi, gamma, mu, R, l)}")

    ratio_wettablitygradient = 0.5  #0=fully covered, 0.5=50:50, 1=fully open
    anglerange1 = np.linspace(0, 0.5, round(nr_of_datapoints/2*ratio_wettablitygradient))  # 0-1        #for open side
    anglerange2 = np.linspace(0.5, 1, round(nr_of_datapoints/2*(1-ratio_wettablitygradient)))  # 0-1    #for closed side

    Ca_eq_mid = (CA_eq_adv + CA_eq_rec) / 2  # 'middle' CA value around which we build the sinus-like profile
    Ca_eq_diff = CA_eq_adv - Ca_eq_mid  # difference between middle CA value & the 'eq' ones - for in the sinus-like function
    #Function to vary between CA_max&_min with a sinus-like shape.
    #Input: anglerange = a range between [0, 1]
    #       k = int value. Defines steepness of sinus-like curve
    f_theta_eq = lambda anglerange, k: ((
                                    np.power(0.5 + np.sin(anglerange * np.pi - np.pi / 2) / 2,
                                    np.power(2 * (1 - anglerange), k) )
                                    ) * 2 - 1) * np.pi / 180  # base function - [-1,1], so around 0+-1

    theta_eq_cover = f_theta_eq(anglerange2, 0)  # under cover = less steep
    theta_eq_open = f_theta_eq(anglerange1, 3)  # open air = steep
    theta_eq = np.concatenate([theta_eq_open, theta_eq_cover, np.flip(theta_eq_cover), np.flip(theta_eq_open)]) * 180 / np.pi
    # perform operation to shift CA values to desired CA_eq range
    theta_eq_deg = theta_eq * Ca_eq_diff + Ca_eq_mid
    theta_eq_rad = theta_eq_deg / 180 * np.pi

    phi = np.linspace(-np.pi, np.pi, nr_of_datapoints)          #angle of CL position. 0 at 3'o clock, pi at 9'o clock. +pi/2 at 12'o clock, -pi/2 at 6'o clock.

    ax1[0,0].plot(phi, theta_eq_deg, color='darkorange', linewidth=7)
    ax1[0,0].set(xlabel=f'radial angle [rad]', ylabel='equilibrium contact angle [deg]', title='Input CA$_{eq}$ variable over radial angle')

    velocity_local = np.array([np.cos(phi_l) * v_adv if abs(phi_l) < np.pi/2 else np.cos(phi_l) * v_rec for phi_l in phi])
    #fig2, ax2 = plt.subplots(1,2, figsize= (15, 9.6/1.5))
    ax1[0,1].plot(phi, velocity_local * 60 / 1E-6, linewidth=7)
    ax1[0,1].set(xlabel='radial angle [rad]', ylabel=f'local velocity [$\mu$m/min]', title='Input local velocity profile \n(adjusted for difference between adv. and rec. speed)')

    Ca_local = mu * velocity_local / gamma
    print(f"Ca = {max(Ca_local)}")
    inBetweenClaculation = np.power(theta_eq_rad, 3) + 9 * Ca_local * np.log(R/l)
    if any(inBetweenClaculation < 0):
        logging.error(f"Some calculated CA's will be NaN")
    theta_app_calculated = np.power(np.power(theta_eq_rad, 3) + 9 * Ca_local * np.log(R/l), 1/3)
    theta_app_calculated_deg = theta_app_calculated * 180 / np.pi

    def trial2(fx, gx, A, phi):
        CA_app = np.power(np.power(fx, 3) + A*np.power(gx, 3), 1/3)

        dfx = np.concatenate([np.diff(fx), [0]])
        dgx = np.concatenate([np.diff(gx), [0]])
        part1 = np.power(fx, 2) * dfx / np.power(A*gx + np.power(fx, 3), 2/3)
        part2 = A*dgx / (3 * np.power(A*gx + np.power(fx, 3), 2/3))

        tot = part1+part2
        peaks,_ = scipy.signal.find_peaks(-abs(tot), width = 5)        #minimal peak width  = 5 datapoints to remove artifacts from stitching curves
        np.set_printoptions(precision=2)
        print(f"extremum CA predicted= {(CA_app[peaks] * 180 / np.pi)}"
              f"\ndeg at phi = {phi[peaks]}rad")
        np.set_printoptions(precision=8)

        fig, ax = plt.subplots(figsize= (12, 9.6))
        ax.plot(phi, -abs(tot))
        ax.plot(phi[peaks], -abs(tot[peaks]), '.')
        return (CA_app[peaks] * 180 / np.pi)            #return values of

    #TODO test this:
    trial2(theta_eq_rad, velocity_local, 9*mu*np.log(R/l) / gamma, phi)

    ax1[1,0].plot(phi, theta_app_calculated_deg, color='darkorange', linewidth=7)
    ax1[1,0].set(xlabel='radial angle [rad]', ylabel='CA$_{app}$ [deg]', title='Calculated apparent contact angle profile')
    fig1.tight_layout()

    R_drop = 1  # [mm]
    x_coord = R_drop * np.cos(phi)
    y_coord = R_drop * np.sin(phi)
    #fig1, ax1 = plt.subplots(figsize= (12, 9.6))
    im1 = ax1[1,1].scatter(x_coord, y_coord, s = 45, c=theta_app_calculated_deg, cmap='jet', vmin=min(theta_app_calculated_deg), vmax=max(theta_app_calculated_deg))
    ax1[1,1].set(xlabel="X-coord", ylabel="Y-Coord", title=f"Spatial Contact Angles Colormap Tilted Droplet\n Quantitative description");
    fig1.colorbar(im1)
    fig1.tight_layout()

    fig1.savefig(os.path.join('C:\\Users\\ReuvekampSW\\Downloads', 'temp1.png'), dpi=600)
    #fig1.savefig(os.path.join('C:\\Users\\ReuvekampSW\\Downloads', 'temp2.png'), dpi=600)

    plt.show()
    return

def movingDropQualitative_fitting():
    """
    Model the CA_app(phi) along a MOVING DROPLET contact line, using only some experimental constants (R,l,𝜇,𝜎) &
    EXPERIMENTALLY OBSERVED 'CA_adv,rec' at the outer drop positions, & 'CA_max,min' at the local extrema !

    Based an the Cox-Voinov formula, 𝜃_𝑎^3−𝜃_𝑒𝑞^3=9𝐶𝑎 𝑙𝑛 𝑅/𝑙    with 𝐶𝑎=𝜇 𝑣_(𝑝ℎ𝑖)/𝜎,
    from these 4 values, an Optimizer function determines the 'best' CA_eq_adv,rec & wettability steepness factors.

    To do that, the calculating_CA_app(...) function is used to:
    -First, from the inputted CA_eq_adv,rec + experimental CA_app_adv,rec, the v_adv,rec are calculated.
    These are then used to create a v(phi) profile along the CL using a normal sin function.
    -The inputted CA_eq_adv,rec are used to create the entire CA_eq(phi) profile using the inputted wettability gradient & steepness factors.

    Knowing CA_eq(phi), v(phi) and (R,l,𝜇,𝜎), we calculate CA_app(phi)

    BE MINDFULL of the programn finding & comparing the correct CA values in the optimizer: It SHOULD compare the EXPERIMENTAL CA_max,min local extrema values with
    the CALCULATED  CA_max,min values, but the latter were sometimes not found properly (if the profile did e.g. not have a local min/max). Implemented fix seems to work,
    but ALWAYS CHECK afterwards if the local max&minima are correct.

    TODO improvements: fix when wettability_gradient =! 0.5. Currently only that one seems to work properly. E.g. 0.3 tends
        to give NaN CA values at the transition location.
    :return:
    """

    exp_CAs_advrec = [1.64, 1.67]       #Hard set: Experimentally observed CA's at the advancing & receding outer points of the droplet [deg]
    target_localextrema_CAs = [1.92, 1.58]  #Hard set: Experimental CA's at local max/minimum (from left->right on droplet)     [deg]
    #TODO this one w/ different values is not working too well yet..
    wettability_gradient = 0.5    # 0=fully covered, 0.5=50:50, 1=fully open

    # Define 'input' theta_eq values for the Cox-Voinov equation. <- derived from experimental data, min&maxima where friction had least influnec
    # Also variation of theta_eq is not defined as a normal sinus, but with a kink (intended because of non-linear swelling gradient under/outside cover)
    CA_eq_adv = 1.20;  # initial guesses
    CA_eq_rec = 1.8  # CA [deg] values observed from spatial CA profile: local max & minima on location where friction should not play big of a role - basically the max & min CA values


    mu = 1.34 / 1000  # Pa*s
    gamma = 25.55 / 1000  # N/m
    R = 100E-6  # slip length, 10 micron?                     -macroscopic
    l = 2E-9  # capillary length, ongeveer              -micro/nanoscopic
    nr_of_datapoints = 2000 #must be a multiple of 4!!
    phi = np.linspace(-np.pi, np.pi, nr_of_datapoints)  # angle of CL position. 0 at 3'o clock, pi at 9'o clock. +pi/2 at 12'o clock, -pi/2 at 6'o clock.

    fig1, ax1 = plt.subplots(2, 2, figsize= (12, 9.6))
    if not nr_of_datapoints % 4 == 0:
        nr_of_datapoints = nr_of_datapoints + (4-(nr_of_datapoints % 4))
        logging.warning(f"nr_of_datapoints is not a multiple of 4: changing it to = {nr_of_datapoints}")

    iterations = []
    def callback(xk, convergence=None):
        """Function called at each iteration."""
        iterations.append(xk)
        print(f"Current iteration: CA_eq_adv={xk[0]:.4f}, CA_eq_rec={xk[1]:.4f}, power cover {xk[2]:.4f}, power open air {xk[3]:.4f}")

    #Older minimize optimizer approach. Sometimes got stuck in local minima (or code was not bug-free yet)
    # sol = scipy.optimize.minimize(              #.minimize
    #     optimizeInputCA,
    #     [CA_eq_adv, CA_eq_rec],
    #     # upper&lowerbounds. For advancing, v>!0, thus CA_app>CA_eq. Thus,
    #     #For receding, v<!0, thus CA_eq > CA_app
    #     bounds=((0.5, exp_CAs_advrec[0]),
    #             (exp_CAs_advrec[1], 3)),
    #     args = (exp_CAs_advrec, phi, target_localextrema_CAs, mu, gamma, R, l, nr_of_datapoints),
    #     method='Powell',
    #     callback=callback)

    OPTIMIZE = False         #True: use optimizer to find best CA_eq_adv,rec & wettability steepnesses. False: manual input (for quick data checking)
    if OPTIMIZE:
        sol = scipy.optimize.differential_evolution(
            optimizeInputCA,
            bounds=((0.5, exp_CAs_advrec[0]),       #advancing  upper&lower bound
                    (exp_CAs_advrec[1], 3),        #receding  upper&lower bound
                    (0, 10),        #steepness of wettability gradient  - covered part
                    (0, 10)),       #steepness of wettability gradient  - open part
            # bounds=((1.2, 1.25),  # advancing  upper&lower bound
            #         (2.2, 2.3),  # receding  upper&lower bound
            #         (0.5, 10),  # steepness of wettability gradient  - covered part
            #         (0.5, 8)),  # steepness of wettability gradient  - open part
            args = (exp_CAs_advrec, phi, target_localextrema_CAs, mu, gamma, R, l, nr_of_datapoints, wettability_gradient),
            callback=callback)
        print(f"sol = {sol.x}")
        print(sol)
        calculated_CA_eq_adv_rec = np.array(sol.x)[0:2]

        calc_vel = targetExtremumCA_to_inputVelocity(np.array(exp_CAs_advrec)/180*np.pi, np.array(calculated_CA_eq_adv_rec)/180*np.pi, gamma, mu, R, l)
        theta_app_calculated, velocity_local, theta_eq_rad = calculating_CA_app(sol.x, exp_CAs_advrec, phi, mu, gamma, R, l, nr_of_datapoints, wettability_gradient)
    else:
        print(f"NOT OPTIMIZING: USING MANUAL INPUT TO SHOW & CALCULATE CA_app")
        #Input required/desired CA_eq_adv,rec angles & wettability steepness factors below
        calculated_CA_eq_adv_rec = [1.57, 2.08]
        calc_vel = targetExtremumCA_to_inputVelocity(np.array(exp_CAs_advrec)/180*np.pi, np.array(calculated_CA_eq_adv_rec)/180*np.pi, gamma, mu, R, l)
        theta_app_calculated, velocity_local, theta_eq_rad = calculating_CA_app(calculated_CA_eq_adv_rec + [6.76, 1.18],
                                                                                exp_CAs_advrec, phi, mu, gamma, R, l,
                                                                                nr_of_datapoints, wettability_gradient,
                                                                                np.array([32, -275])*(1E-6 / 60))

    print(f"Corresponding velocities are = {np.array(calc_vel)/(1E-6 / 60)} mu/min")
    theta_app_calculated_deg = theta_app_calculated * 180 / np.pi

    #TODO can be removed: for showing peaks found in CA_app only
    A = 9 * mu * np.log(R / l) / gamma
    calc_local_extrema_values(theta_eq_rad, velocity_local, A, phi, PLOTTING=True)

    ## Input CA_eq profile
    ax1[0,0].plot(phi, theta_eq_rad * 180 / np.pi, color='blue', linewidth=7)
    ax1[0,0].set(xlabel='radial angle [rad]', ylabel='CA$_{eq}$ [deg]', title='Input CA$_{eq}$ profile')

    ##  Input velocity_local profile
    ax1[0,1].plot(phi, velocity_local / (1E-6 / 60), color='blue', linewidth=7)
    ax1[0,1].set(xlabel='radial angle [rad]', ylabel='velocity [$\mu$m/min]', title='Input local velocity profile')

    ## calculated  CA_App profile 2D
    peaks,_ = scipy.signal.find_peaks(theta_app_calculated_deg, width = 5)
    minima, _ = scipy.signal.find_peaks(-theta_app_calculated_deg, width = 5)
    ax1[1,0].plot(phi, theta_app_calculated_deg, color='darkorange', linewidth=7)
    ax1[1,0].plot(phi[peaks[0]], theta_app_calculated_deg[peaks[0]], '.', color='blue', markersize=10, label=r'CA$_{target, max}$=CA$_{calc}$=' + f'{target_localextrema_CAs[0]:.2f}')
    ax1[1, 0].plot(phi[minima[0]], theta_app_calculated_deg[minima[0]], '.', color='blue', markersize=10, label=r'CA$_{target, min}$=CA$_{calc}$=' + f'{target_localextrema_CAs[1]:.2f}')
    ax1[1,0].set(xlabel='radial angle [rad]', ylabel='CA$_{app}$ [deg]', title='Calculated apparent contact angle profile')
    ax1[1,0].legend(loc='best')
    fig1.tight_layout()

    R_drop = 1  # [mm]
    x_coord = R_drop * np.cos(phi)
    y_coord = R_drop * np.sin(phi)
    #fig1, ax1 = plt.subplots(figsize= (12, 9.6))
    im1 = ax1[1,1].scatter(x_coord, y_coord, s = 45, c=theta_app_calculated_deg, cmap='jet', vmin=min(theta_app_calculated_deg), vmax=max(theta_app_calculated_deg))
    ax1[1,1].set(xlabel="X-coord", ylabel="Y-Coord", title=f"Spatial Contact Angles Colormap Tilted Droplet\n Quantitative description");
    fig1.colorbar(im1)
    fig1.tight_layout()


    fig1.savefig(os.path.join('C:\\Users\\ReuvekampSW\\Downloads', 'temp1.png'), dpi=600)
    #fig1.savefig(os.path.join('C:\\Downloads', 'temp1.png'), dpi=600)

    plt.show()
    return

def calc_local_extrema_values(fx, gx, A, phi, PLOTTING=False):
    """
    Calculate & return values of local extrema by finding where d𝜃_𝑎/dϕ = 0.

    calculates Apparent Contact Angle over entire contact line range with:
    𝜃_𝑎= (𝜃_𝑒𝑞^3 + 9𝐶𝑎 𝑙𝑛 𝑅/𝑙)^1/3

    Then, it's derivate to zero is the 'index' location of local extrema:
    sum =   f(x)^2 f′(x) / (A g(x) + f(x)^3) ^ 2/3)
            + Ag′(x) / (3 (A g(x) + f(x)^3) ^ 2/3)
        == 0

    :param fx: given CA_eq profile  [rad]
    :param gx: given velocity profile   [m/s]
    :param A: combination of some constants: 9*𝜇/𝜎 * 𝑙𝑛(𝑅/𝑙)
    :param phi: radial position
    :return: CA_extrema values [deg], corresponding radial position [rad]
    """
    CA_app = np.power(np.power(fx, 3) + A * gx, 1/3)

    dfx = np.concatenate([np.diff(fx), [0]])
    dgx = np.concatenate([np.diff(gx), [0]])
    part1 = np.power(fx, 2) * dfx / np.power(A * gx + np.power(fx, 3), 2 / 3)
    part2 = A * dgx / (3 * np.power(A * gx + np.power(fx, 3), 2 / 3))

    tot = part1 + part2
    peaks, _ = scipy.signal.find_peaks(-abs(tot), width=5)  # minimal peak width  = 5 datapoints to remove artifacts from stitching curves
    # np.set_printoptions(precision=2)
    # print(f"extremum CA predicted= {(CA_app[peaks] * 180 / np.pi)}"
    #       f"\ndeg at phi = {phi[peaks]}rad")
    # np.set_printoptions(precision=8)
    if PLOTTING:
        fig, ax = plt.subplots(2,2)
        ax[0,0].plot(phi, fx * 180 / np.pi)
        ax[0,0].set(title='CA_eq profile')
        ax[1,0].plot(phi, gx / (1E-6 /60))
        ax[1,0].set(title='velocity profile')
        ax[0,1].plot(phi, CA_app * 180 / np.pi)
        ax[0,1].plot(phi[peaks], CA_app[peaks]* 180 / np.pi, '.', markersize=5)
        ax[0,1].set(title='CA_app profile calculated')
        ax[1,1].plot(tot)
        ax[1,1].plot(peaks, tot[peaks], '.', markersize=5)
        ax[1,1].set(title='derivate sum')
    return (CA_app[peaks] * 180 / np.pi), phi[peaks]  # return values of the peaks

def optimizeInputCA(CAs_input, exp_CAs_advrec, phi, target_localextrema_CAs, mu, gamma, R, l, nr_of_datapoints, wettability_gradient):
    """
    Optimizer function for determining the 'best' CA_eq_adv,rec & wettability steepness factors for MOVING DROPLETS RIGHT with
    a given input of:

    :param CAs_input: 4 values!  CA_eq_adv&rec AND wettability profile steepness factors under cover&openair.
            ^THESE values will be fitted for in the optimizer.
    :param exp_CAs_advrec: EXPERIMENTALLY OBSERVED CA's at the outer advancing & receding position
            ^THESE are used to calculate the velocity of the droplet at the outer advancing & receding position
    :param phi: radial angle range [-pi, pi]
    :param target_localextrema_CAs:  EXPERIMENTALLY OBSERVED CA's at the local maximum and minimum, respectively. SO INPUT=from left to right in droplet.
            ^THESE are the target valuse for the optimizer.
    :param mu:          viscosity       [Pa*s]
    :param gamma:       surface tension  [N/m]
    :param R:           macroscopic cutoff length   [m]
    :param l:           micro/nanoscopic cutoff length [m]
    :param nr_of_datapoints: nr. of datapoints of input.
    :param wettability_gradient: value 'size' of the respective wettability gradients under plate and in open air: 0=fully covered, 0.5=50:50, 1=fully open
    :return: difference between calculated local min/max & experimental target min/max. Lower=better.
    """
    theta_app_calculated, velocity_local, theta_eq_rad = calculating_CA_app(CAs_input, exp_CAs_advrec, phi, mu, gamma, R, l, nr_of_datapoints, wettability_gradient)

    local_extrema_calculated, phi_local_extrema = calc_local_extrema_values(theta_eq_rad, velocity_local, 9 * mu * np.log(R / l) / gamma, phi)  # i0=local max, i1= local min        (i2 right max, i3 local min, i4 local max)
    local_extrema_calculated_temp = local_extrema_calculated
    phi_local_extrema_temp = phi_local_extrema
    local_extrema_calculated = []
    phi_local_extrema = []
    error_to_giveback = 1e6
    if len(local_extrema_calculated_temp) > 5:      #if less than 6 local max/minima found, give large error: solution will not contain desired local max/minim
        for n, CA in enumerate(local_extrema_calculated_temp):              #remove local extrema from list at 9&3 `o clock
            #if abs(phi_local_extrema_temp[n]) > np.pi/5 and abs(phi_local_extrema_temp[n]) < (4*np.pi/5):
            local_extrema_calculated.append(CA)
            phi_local_extrema.append(phi_local_extrema_temp[n])
        try:
            difference_target_calculated = [target_localextrema_CAs[0] - local_extrema_calculated[0],           # difference experimentally observed local min/max vs. modelled
                                            target_localextrema_CAs[1] - local_extrema_calculated[2]]
        except:
            print(
                f"too few extrema are found from the trial2 function: min&max CA's used are {min(theta_eq_rad) * 180 / np.pi:.2f} & {max(theta_eq_rad) * 180 / np.pi:.2f} deg\n with velocities min&max: {min(velocity_local) * 1E6 / 60:.3e}, {max(velocity_local) * 1E6 / 60:.3e} um/min.")
            if len(local_extrema_calculated) == 0:
                difference_target_calculated = [error_to_giveback, error_to_giveback]
            elif len(local_extrema_calculated) == 1:
                difference_target_calculated = [target_localextrema_CAs[0] - local_extrema_calculated[0], error_to_giveback]
    else:
        difference_target_calculated = [error_to_giveback, error_to_giveback]
    return abs(difference_target_calculated[0]) + abs(difference_target_calculated[1])

def calculating_CA_app(CAs_eq_advrec_input, exp_CAs_advrec, phi, mu, gamma, R, l, nr_of_datapoints, ratio_wettablitygradient = 0.5, calculateVelocies = []):
    """
    Calculating function of the CA_app along the contact line from the following input values.
    -First, from the inputted CA_eq_adv,rec + experimental CA_app_adv,rec, the v_adv,rec are calculated.
    These are then used to create a v(phi) profile along the CL using a normal sin function.
    -The inputted CA_eq_adv,rec are used to create the entire CA_eq(phi) profile using the inputted wettability gradient & steepness factors.

    With these two profile in place, the resulting CA_app(phi) profile is calculated along the CL.

    :param CAs_eq_advrec_input: 4 values!  CA_eq_adv&rec AND wettability profile steepness factors under cover&openair.
            ^THESE values will be fitted for in the optimizer.
    :param exp_CAs_advrec: EXPERIMENTALLY OBSERVED CA's at the outer advancing & receding position
            ^THESE are used to calculate the velocity of the droplet at the outer advancing & receding position
    :param phi: radial angle range [-pi, pi]
    :param target_localextrema_CAs:  EXPERIMENTALLY OBSERVED CA's at the local maximum and minimum, respectively. SO INPUT=from left to right in droplet.
            ^THESE are the target valuse for the optimizer.
    :param mu:          viscosity       [Pa*s]
    :param gamma:       surface tension  [N/m]
    :param R:           macroscopic cutoff length   [m]
    :param l:           micro/nanoscopic cutoff length [m]
    :param nr_of_datapoints: nr. of datapoints of input.
    :param wettability_gradient: value 'size' of the respective wettability gradients under plate and in open air: 0=fully covered, 0.5=50:50, 1=fully open
    :return: CA_app(phi), v(phi), CA_eq(phi)    all along contact line position
    """

    # fitting inputs: first 2 are adv&rec eq. contact angle. second 2 are for steepness of wettability gradient.
    CA_eq_adv, CA_eq_rec, steepnessWettability_cover, steepnessWettability_open = CAs_eq_advrec_input


    exp_CA_adv, exp_CA_rec = exp_CAs_advrec


    # Input velocities at advancing & receding point.
    #v_adv = 150 * 1E-6 / 60  # [m/s] assume same velocity at advancing and receding, just in opposite direction       [70]    (right side)
    #v_rec = 150 * 1E-6 / 60  # [m/s] assume same velocity at advancing and receding, just in opposite direction      [150]    (left side)

    if len(calculateVelocies) == 0: #If no given input velocities, calculate them from the input experimental CA_adv,rec
                                    #This is the prefered way - and gives physically correct&matching CA_adv,rec angles
        # Input target advancing and receding CA (outer moving droplet) to calculate required local velocities
        v_adv, v_rec = targetExtremumCA_to_inputVelocity(np.array([exp_CA_adv, exp_CA_rec]) / 180 * np.pi,
                                                         np.array([CA_eq_adv, CA_eq_rec]) / 180 * np.pi,
                                                         gamma, mu, R, l)
        v_rec = -v_rec      #swap sign: input in formula below should be positive (it'll make it negative) #TODO not the nicest - change if needed
    else: #else, use the given input velocities for
        if not len(calculateVelocies) == 2:
            logging.critical(f"Wrong amount of in put velocities. Input 2 values: [v_adv, v_rec] in m/s")
        v_adv = calculateVelocies[0]
        v_rec = -calculateVelocies[1]       #swap sign: input in formula below should be positive (it'll make it negative)
    #print(f"Velocities in between: adv {v_adv/(1E-6 / 60):.2f} $\mu$/min, rec {v_rec/(1E-6 / 60):.2f} $\mu$/min.")

    anglerange1 = np.linspace(0, 0.5,
                              round(nr_of_datapoints / 2 * ratio_wettablitygradient))  # 0-1        #for open side
    anglerange2 = np.linspace(0.5, 1,
                              round(nr_of_datapoints / 2 * (1 - ratio_wettablitygradient)))  # 0-1    #for closed side
    Ca_eq_mid = (CA_eq_adv + CA_eq_rec) / 2  # 'middle' CA value around which we build the sinus-like profile
    Ca_eq_diff = CA_eq_adv - Ca_eq_mid  # difference between middle CA value & the 'eq' ones - for in the sinus-like function
    # Function to vary between CA_max&_min with a sinus-like shape.
    # Input: anglerange = a range between [0, 1]
    #       k = int value. Defines steepness of sinus-like curve
    f_theta_eq = lambda anglerange, k: ((
                                            np.power(0.5 + np.sin(anglerange * np.pi - np.pi / 2) / 2,
                                                     np.power(2 * (1 - anglerange), k))
                                        ) * 2 - 1) * np.pi / 180  # base function - [-1,1], so around 0+-1
    theta_eq_cover = f_theta_eq(anglerange2, steepnessWettability_cover)  # under cover = less steep         #0      or even 3
    theta_eq_open = f_theta_eq(anglerange1, steepnessWettability_open)  # open air = steep                  #3      or even 7
    theta_eq = np.concatenate(
        [theta_eq_open, theta_eq_cover, np.flip(theta_eq_cover), np.flip(theta_eq_open)]) * 180 / np.pi
    # perform operation to shift CA values to desired CA_eq range
    theta_eq_deg = theta_eq * Ca_eq_diff + Ca_eq_mid
    theta_eq_rad = theta_eq_deg / 180 * np.pi

    velocity_local = np.array(
        [np.cos(phi_l) * v_adv if abs(phi_l) < np.pi / 2 else np.cos(phi_l) * v_rec for phi_l in phi])

    Ca_local = mu * velocity_local / gamma
    inBetweenClaculation = np.power(theta_eq_rad, 3) + 9 * Ca_local * np.log(R / l)
    if any(inBetweenClaculation < 0):
        logging.error(f"Some calculated CA's will be NaN")
    theta_app_calculated = np.power(np.power(theta_eq_rad, 3) + 9 * Ca_local * np.log(R / l), 1 / 3)

    return theta_app_calculated, velocity_local, theta_eq_rad


def targetExtremumCA_to_inputVelocity(CA_app: float, CA_eq: float, sigma: float, mu:float, R:float, l:float):
    """
    Calculate what the (maximum) velocity must be for the inputted CA_app at the advancing or receding location
    for any droplet. Simply Cox_voinov rewritten.
    :param CA_app: target maximum or minimum CA at adv./rec. respectively [rad]
    :param CA_eq: input CA_eq [rad]
    :return: velocity [m/s] or [um/min]
    """
    velocity = ((np.power(CA_app, 3) - np.power(CA_eq, 3)) * sigma) / (9 * mu * np.log(R/l))
    return velocity

def testingQualitativeDescription():
    angle = np.linspace(0, np.pi, 1000)
    # theta_eq = (np.sin(angle-np.pi/2) + 2) * np.pi / 180

    #Ca = -1.55E-7 * np.sin(angle - np.pi / 2)  # OG standard Ca curve: normal sinus between + and - the value
    Ca = -5E-8 * np.sin(angle - np.pi / 2)  # OG standard Ca curve: normal sinus between + and - the value

    for Ca in [-1E-8 * np.sin(angle - np.pi / 2), -5E-8 * np.sin(angle - np.pi / 2), -1E-7 * np.sin(angle - np.pi / 2), -5E-7 * np.sin(angle - np.pi / 2)]:
        #Figure: input capillary number (Ca) vs. phi
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(angle * 180 / np.pi, Ca)
        ax1.set(xlabel='Contact angle', ylabel='Capillary number')

        x = 10E-6  # slip length, 10 micron?                     -macroscopic
        l = 2E-9  # capillary length, ongeveer              -micro/nanoscopic
        for x in [10E-6, 100E-6, 1000E-6]:
            anglerange = np.linspace(0, 1, 1000)
            k = 3
            # Ca = -1E-6 * ((((0.5+np.sin(anglerange*np.pi-np.pi/2)/2)**((2*(1-anglerange))**k)))*2 - 1)
            theta_eq = (((0.5 + np.sin(anglerange * np.pi - np.pi / 2) / 2) ** (
                        (2 * (1 - anglerange)) ** k)) * 2 + 1) * np.pi / 180
            prefactor = 9  # OG = 9
            theta_app = (theta_eq ** 3 + prefactor * Ca * np.log(x / l)) ** (1 / 3)
            print(f"{x / l}")
            print(f"{prefactor * np.log(x / l)}")
            # xOverL = 0.01
            # theta_app = (theta_eq**3 + 9*Ca*np.log(xOverL))**(1/3)



            CA_max_i = np.argmax(theta_app)
            CA_min_i = np.argmin(theta_app)

            #Figure: plot w/ both CA_eq (non friction) & CA_app (friction)
            fig1, ax1 = plt.subplots(figsize=(9, 6))
            ax1.plot(angle * 180 / np.pi, theta_eq * 180 / np.pi, label=r'$\theta_{eq}$ - no friction')
            ax1.plot(angle * 180 / np.pi, theta_app * 180 / np.pi, '.', label=r'$\theta_{app}$ - with friction')
            ax1.set(xlabel='Azimuthal angle (deg)', ylabel='Contact angle (deg)',
                    title='Example influence hydrolic resistance on apparent contact angle\n'
                          f'x={x:.1E}, l={l:.1E}, x/l={x/l:.1E}, Ca=[{min(Ca):.1E} - {max(Ca):.1E}]\n'
                          f'CA_max at {angle[CA_max_i]* 180 / np.pi:.1f}deg & CA_min at {angle[CA_min_i]* 180 / np.pi:.1f}deg')
            ax1.legend(loc='best')
            #fig1.savefig(f"C:\\Downloads\\CA vs azimuthal Ca={max(Ca):.1E}, x={x:.1E}, l={l:.1E}.png", dpi=600)
            fig1.savefig(f"C:\\Users\\ReuvekampSW\\Downloads\\CA vs azimuthal Ca={max(Ca):.1E}, x={x:.1E}, l={l:.1E}.png", dpi=600)
            #

            theta_eq = theta_eq * 180 / np.pi
            theta_app = theta_app * 180 / np.pi

            # #Figure: spatial colormap: CA_eq
            # fig3, ax3 = plt.subplots(figsize=(9, 6))
            # xArrFinal = np.cos(angle)
            # yArrFinal = np.sin(angle)
            # im3 = ax3.scatter([xArrFinal, np.flip(xArrFinal)], [yArrFinal, -np.flip(yArrFinal)], c=[theta_eq, np.flip(theta_eq)], cmap='jet', vmin=min(theta_eq), vmax=max(theta_eq))
            # ax3.set_xlabel("X-coord");
            # ax3.set_ylabel("Y-Coord");
            # ax3.set_title(f"Model: No Hydrolic Resistance \nSpatial Equilibrium Contact Angles Colormap", fontsize=20)
            # fig3.colorbar(im3)
            # fig3.savefig("C:\\Downloads\\NOhydrolic.png", dpi=600)
            # #
            #
            # #Figure: spatial colomap: CA_app
            # fig4, ax4 = plt.subplots(figsize=(9, 6))
            # im4 = ax4.scatter([xArrFinal, np.flip(xArrFinal)], [yArrFinal, -np.flip(yArrFinal)], c=[theta_app, np.flip(theta_app)], cmap='jet', vmin=min(theta_app), vmax=max(theta_app))
            # ax4.set_xlabel("X-coord"); ax4.set_ylabel("Y-Coord");
            # ax4.set_title(f"Model: Effect of viscous friction\nSpatial Predicted Apparent Contact Angles Colormap", fontsize=20)
            # fig4.colorbar(im4)
            # fig4.savefig(f"C:\\Downloads\\YEShydrolic Ca={max(Ca)}, x={x}, l={l}.png", dpi=600)
            # #

    plt.show()

def main():
    try:
        #testingQualitativeDescription()

        #tiltedDropQualitative()        #using this one to model tilted droplets
        #movingDropQualitative()
        movingDropQualitative_fitting() #Using this one to FIT moving droplets!

        #xcoord, ycoord, CA = importData()
        #fitSpatialCA(xcoord, ycoord, CA, middleCoord)

        #fitSpatialCA_simplified(xcoord, ycoord, CA, middleCoord)
        #trial1(xcoord, ycoord, CA, middleCoord)

        angle = np.linspace(0, np.pi, 1000)
        # theta_eq = (np.sin(angle-np.pi/2) + 2) * np.pi / 180

        # Ca = -1.55E-7 * np.sin(angle-np.pi/2)  #OG standard Ca curve: normal sinus between + and - the value
        # fig1, ax1 = plt.subplots(figsize=(6,4))
        # ax1.plot(angle*180/np.pi, Ca)
        # ax1.set(xlabel = 'Contact angle', ylabel = 'Capillary number')
    except:
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
    exit()