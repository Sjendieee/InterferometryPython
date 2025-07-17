"""
based on Fitting_spatical_CA_profiles_CoxVoinov.py
adjusted such that it takes as input:
CA_adv,CA_rec , v_adv, v_rec and some typical input parameters l,R,gamma,viscosity

Calculate CA_eq on both front & back and use average as input for presumed 'constant wettability profile'
Set up velocity(phi) profile, and calculate following CA_app(phi) profile.
Dump to pickle file for import as fit.
"""

import logging
import traceback

import dill
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime




def tiltedDropQualitative_fitting():
    """
    Model theta_app for a tilted droplet with 'known' values: input constant thetha_eq, thetha_adv & theta_rec, R & L.
    Vary Ca(phi) by calculating the v_adv & v_rec from the CA_adv,rec

    𝜃_𝑎^3−𝜃_𝑒𝑞^3=9𝐶𝑎 𝑙𝑛 𝑅/𝑙
    with 𝐶𝑎=𝜇 𝑣_(𝑝ℎ𝑖)/𝜎
    :return:
    """
    path = 'C:\\Users\\ReuvekampSW\\Downloads'
    if not os.path.exists(path):
        path = 'C:\\Downloads'
    if not os.path.exists(path):
        logging.critical("Path does not exist: put a correct one in")
        exit()
    logging.info(f"Dumping data to the following directory:\n {path}")


    nr_of_datapoints = 2000
    ##### INPUT######
    CA_EQ_CONSTANT = False      #true: use averga CA_eq front&back. False: calculate CA_eq font&back, vary linearly between them
    IMPORT_VEL_PROFILE = True
    #theta_eq_deg =  1.78 # Eq Contact angle along entire contact line (for tilted drops should be constant) [deg]
    theta_adv_deg, theta_rec_deg = 1.97, 1.54   #measured CA_adv & CA_rec at the outer positions of the droplet [deg]
    v_adv = 85E-6 / 60
    v_rec = -140E-6 / 60

    mu = 1.34 / 1000  # Pa*s
    gamma = 25.35 / 1000  # N/m
    R = 20E-6  # slip length / capillary length, 10 micron to 1.9mm               -macroscopic
    l = 2E-9  # about 1-2 nm              -micro/nanoscopic
    ##### END INPUT######
    phi = np.linspace(-np.pi, np.pi, nr_of_datapoints)  # angle of CL position. 0 at 3'o clock, pi at 9'o clock. +pi/2 at 12'o clock, -pi/2 at 6'o clock.
    CA_apps = np.array([theta_adv_deg, theta_rec_deg]) / 180 * np.pi



    if IMPORT_VEL_PROFILE:
        with open('C:\\Users\\ReuvekampSW\\Downloads\\velocityProfile_20250715144135.pkl', 'rb') as new_filename:
            data = dill.load(new_filename)
        logging.info("IMPORTING VELOCITY PROFILE from external pickle file")
        velocity_local = np.array(data(phi))
        v_adv, v_rec = data(np.array([0, np.pi]))
        vels = np.array([v_adv, v_rec])
    else:
        vels = np.array([v_adv, v_rec])
        velocity_local = np.array(
        [np.cos(phi_l) * v_adv if abs(phi_l) < np.pi / 2 else np.cos(phi_l) * -v_rec for phi_l in phi])

    print(f"Calculated velocities adv&rec position are = {np.array([v_adv, v_rec]) / (1E-6 / 60)} mu/min")

    CA_eqs = np.power(np.power(CA_apps, 3) - vels * 9 * mu * np.log(R / l) / gamma, 1/3)   #OG cox-voinov
    # CA_eqs = np.power(np.power(CA_apps, 3) - 9 * (vels * mu / gamma + vels * 1E15 / gamma)  * (np.log(R / l)) , 1/3)   #cox-voinov + Ca_effective

    if CA_EQ_CONSTANT:
        CA_eq = np.mean(CA_eqs)
        theta_eq_deg = CA_eq * 180/ np.pi
        print(f"Calculated CA_eqs = {CA_eqs*180/np.pi} deg\n"
                     f"Calculated CA_eq avg = {theta_eq_deg} deg")
        theta_eq_rad, theta_adv_rad, theta_rec_rad = np.array([theta_eq_deg, theta_adv_deg, theta_rec_deg]) / 180 * np.pi
        theta_eq_deg_arr = np.ones(nr_of_datapoints) * theta_eq_deg
        theta_eq_rad_arr = theta_eq_deg_arr / 180 * np.pi

    else:
        ratio_wettablitygradient = 0.5  # 0=fully covered, 0.5=50:50, 1=fully open
        anglerange1 = np.linspace(0, 0.5, round(nr_of_datapoints / 2 * ratio_wettablitygradient))  # 0-1        #for open side
        anglerange2 = np.linspace(0.5, 1, round(nr_of_datapoints / 2 * (1 - ratio_wettablitygradient)))  # 0-1    #for closed side

        Ca_eq_mid = (CA_eqs[0] + CA_eqs[1]) / 2  # 'middle' CA value around which we build the sinus-like profile
        Ca_eq_diff = CA_eqs[0] - Ca_eq_mid  # difference between middle CA value & the 'eq' ones - for in the sinus-like function
        # Function to vary between CA_max&_min with a sinus-like shape.
        # Input: anglerange = a range between [0, 1]
        #       k = int value. Defines steepness of sinus-like curve
        f_theta_eq = lambda anglerange, k: ((
                                                np.power(0.5 + np.sin(anglerange * np.pi - np.pi / 2) / 2,
                                                         np.power(2 * (1 - anglerange), k))
                                            ) * 2 - 1) * np.pi / 180  # base function - [-1,1], so around 0+-1

        theta_eq_cover = f_theta_eq(anglerange2, 0)  # under cover = less steep
        theta_eq_open = f_theta_eq(anglerange1, 0)  # open air = steep
        theta_eq = np.concatenate([theta_eq_open, theta_eq_cover, np.flip(theta_eq_cover), np.flip(theta_eq_open)]) * 180 / np.pi
        # perform operation to shift CA values to desired CA_eq range
        theta_eq_rad_arr = theta_eq * Ca_eq_diff + Ca_eq_mid
        theta_eq_deg_arr = theta_eq_rad_arr * 180 / np.pi



    ########### plotting of data ##########
    # fig2, ax2 = plt.subplots(1, 2, figsize=(15, 9.6 / 1.5))
    #
    # # Input velocity profile
    # ax2[0].plot(phi, velocity_local * 60 / 1E-6, linewidth=7)
    # ax2[0].set(xlabel='radial angle [rad]', ylabel=f'local velocity [$\mu$m/min]',
    #            title='Input local velocity profile \n(adjusted for difference between adv. and rec. speed)')
    #

    Ca_local = mu * velocity_local / gamma
    #Ca_effective =  mu * velocity_local / gamma +  1E15 * velocity_local / gamma
    print(f"Ca = {max(Ca_local)}")
    if any(0 > np.power(theta_eq_rad_arr, 3) + (9 * min(Ca_local) * np.log(R / l))):
        logging.error(
            f"Some calculated CA's will be NaN: min(Ca) = {min(Ca_local)} +  min(theta_eq_rad) = {min(theta_eq_rad_arr)} will be negative:"
            f"{np.power(min(theta_eq_rad_arr), 3) + (9 * min(Ca_local) * np.log(R / l))}, ()^1/3 = {np.power(np.power(min(theta_eq_rad_arr), 3) + 9 * min(Ca_local) * np.log(R / l), 1 / 3)} ")
    theta_app_calculated = np.power(np.power(theta_eq_rad_arr, 3) + 9 * Ca_local * np.log(R / l), 1 / 3)   #OG CoxVoinov
    #theta_app_calculated = np.power(np.power(theta_eq_rad_arr, 3) + 9 * Ca_effective * np.log(R / l), 1 / 3)    #Ca_effective

    theta_app_calculated_deg = theta_app_calculated * 180 / np.pi
    #
    # # Caluclated CA apparent normal plot
    # ax2[1].plot(phi, theta_app_calculated_deg, color='darkorange', linewidth=7)
    # ax2[1].set(xlabel='radial angle [rad]', ylabel='CA$_{app}$ [deg]',
    #            title='Calculated apparent contact angle profile')
    # fig2.tight_layout()
    #
    # R_drop = 1  # [mm]
    # x_coord = R_drop * np.cos(phi)
    # y_coord = R_drop * np.sin(phi)
    #
    # # Seperate scatterplot
    # fig1, ax1 = plt.subplots(figsize=(12, 9.6))
    # im1 = ax1.scatter(x_coord, y_coord, s=45, c=theta_app_calculated_deg, cmap='jet',
    #                   vmin=min(theta_app_calculated_deg), vmax=max(theta_app_calculated_deg))
    # ax1.set_xlabel("X-coord");
    # ax1.set_ylabel("Y-Coord");
    # ax1.set_title(f"Spatial Contact Angles Colormap Tilted Droplet\n Quantitative description", fontsize=20)
    # fig1.colorbar(im1)
    # fig1.tight_layout()
    #
    # fig2.savefig(os.path.join('C:\\Users\\ReuvekampSW\\Downloads', 'temp1.png'), dpi=600)
    # fig1.savefig(os.path.join('C:\\Users\\ReuvekampSW\\Downloads', 'temp2.png'), dpi=600)
    #
    # plt.show()
    #
    ##########
    fig1, ax1 = plt.subplots(2, 2, figsize= (12, 9.6))

    ## Input CA_eq profile
    ax1[0,0].plot(phi, theta_eq_rad_arr * 180 / np.pi, color='blue', linewidth=7)
    ax1[0,0].set(xlabel='radial angle [rad]', ylabel='CA$_{eq}$ [deg]', title='Input CA$_{eq}$ profile')

    ##  Input velocity_local profile
    ax1[0,1].plot(phi, velocity_local / (1E-6 / 60), color='blue', linewidth=7)
    ax1[0,1].set(xlabel='radial angle [rad]', ylabel='velocity [$\mu$m/min]', title='Input local velocity profile')

    ## calculated  CA_App profile 2D
    ax1[1,0].plot(phi, theta_app_calculated_deg, color='darkorange', linewidth=7)
    ax1[1,0].set(xlabel='radial angle [rad]', ylabel='CA$_{app}$ [deg]', title='Calculated apparent contact angle profile')
    #ax1[1,0].legend(loc='best')
    fig1.tight_layout()

    R_drop = 1  # [mm]
    x_coord = R_drop * np.cos(phi)
    y_coord = R_drop * np.sin(phi)
    #fig1, ax1 = plt.subplots(figsize= (12, 9.6))
    im1 = ax1[1,1].scatter(x_coord, y_coord, s = 45, c=theta_app_calculated_deg, cmap='jet', vmin=min(theta_app_calculated_deg), vmax=max(theta_app_calculated_deg))
    ax1[1,1].set(xlabel="X-coord", ylabel="Y-Coord", title=f"Spatial Contact Angles Colormap Tilted Droplet\n Quantitative description");
    fig1.colorbar(im1)
    fig1.tight_layout()

    fig1.savefig(os.path.join(path, 'tiltedDrop.png'), dpi=600)
    plt.show()

    with open(os.path.join(path, f"CoxVoinovFit_tiltedDrop_{datetime.now().strftime('%Y%m%d%H%M%S')}.pickle"), 'wb') as internal_filename:
        # Dump data in order: Calculated CA [rad], velocity profile [m/s], input CA_eq profile [rad], solution fit where
        # sol[0 & 1] CA_eq adv & rec,    sol[2&3] wettability gradient factor covered&open part
        pickle.dump([theta_app_calculated, velocity_local, theta_eq_rad_arr], internal_filename)

    return


def main():
    tiltedDropQualitative_fitting()
    return


if __name__ == '__main__':
    try:
        main()
    except:
        logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')  # configuration for printing logging messages. Can be removed safely
        print(traceback.format_exc())
    exit()