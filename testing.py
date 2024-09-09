import logging
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt


def showPlot(SHOWPLOTS_SHORT):
    if SHOWPLOTS_SHORT == 0:
        pass
    elif SHOWPLOTS_SHORT == 1:
        plt.show(block=False)
        plt.pause(3)
    elif SHOWPLOTS_SHORT == 2:
        plt.show()
    else:
        logging.critical(f"Wrong 'SHOWPLOTS_SHORT' value. Plotting will probably go wrong now.")
    plt.close('all')

def showPlot(display_mode, figures):
    """
    Display one or more plots with the specified display mode.

    Parameters:
    - display_mode: A string that specifies the display mode. It can be:
        - 'none': Do not display the plots.
        - 'timed': Display the plots for 3 seconds.
        - 'manual': Display the plots until manually closed.
    - figures: A list of matplotlib figure objects to be displayed.
    """
    if display_mode == 'none':
        return

    for fig in figures:
        fig.show()

    if display_mode == 'timed':
        plt.pause(3)
        for fig in figures:
            plt.close(fig)
    elif display_mode == 'manual':
        plt.show()
    else:
        raise ValueError("Invalid display_mode. Use 'none', 'timed', or 'manual'.")


def showPlot2(display_mode, figures):
    """
    Display one or more plots with the specified display mode.

    Parameters:
    - display_mode: A string that specifies the display mode. It can be:
        - 'none': Do not display the plots.
        - 'timed': Display the plots for 3 seconds.
        - 'manual': Display the plots until manually closed.
    - figures: A list of matplotlib figure objects to be displayed.
    """
    if display_mode == 'none':
        return

    if display_mode == 'timed':
        for fig in figures:
            fig.show()
        plt.pause(3)  # Display plots for 3 seconds
        for fig in figures:
            plt.close(fig)
    elif display_mode == 'manual':
        # Show only the specified figures
        for fig in figures:
            fig.show()
            # Display the figures until manually closed
            #plt.show(fig)
    else:
        raise ValueError("Invalid display_mode. Use 'none', 'timed', or 'manual'.")


def showPlot3(figures, mode='manual', duration=3, show_index=None):
    """
    Displays specific plots based on the mode and index provided.

    Parameters:
    - figures: List of figure objects.
    - mode: 'timed' to display for a fixed duration or 'manual' to keep open until closed.
    - duration: Time in seconds to display plots if mode is 'timed'.
    - show_index: Index of the plot to be displayed. Only applicable for the 'manual' mode.
    """

    if mode == 'timed':
        # Display all figures for a fixed duration
        for fig in figures:
            fig.show()  # Show the figure
        plt.pause(duration)
        plt.close('all')

    elif mode == 'manual':
        if show_index is not None and 0 <= show_index < len(figures):
            # Hide all figures first
            for i, fig in enumerate(figures):
                if i != show_index:
                    fig.canvas.manager.window.hide()  # Hide the figure window
            # Show the selected figure
            figures[show_index].canvas.manager.window.show()  # Show the figure window
            figures[show_index].canvas.draw()
            plt.show(block=True)  # Keep the plot open until closed manually
        else:
            print("Invalid index provided for manual display.")


def main():
    x = [1,2,3,4,5]
    y = [1,2,3,4,5]
    z = [5,4,3,2,1]
    SHOWPLOTS_SHORT = 1

    fig1, ax1 = plt.subplots()
    ax1.plot(x,y)
    fig2, ax2 = plt.subplots()
    ax2.plot(x,z)
    fig4, ax4 = plt.subplots()
    ax4.plot(x, [9,7,5,4,2])

    showPlot3([fig1,fig2, fig4], 'manual', show_index=1)
    #display_plots([fig1,fig2], 'timed')

    fig3, ax3 = plt.subplots()
    ax3.plot(x,np.array(y)+np.array(z), '*')
    #showPlot2('timed',[fig3])


if __name__ == "__main__":
    main()
    exit()