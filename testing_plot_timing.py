"""
Testing showing of only desired plot, untill clicked away
For now, the methods in showPlot_V3 & showPlot_V4 work as desired for the following purposes!
showPlot_V3 = not showing, timed & open untill clicked away (but non closed GUI loops)
showPlot_V4 = for 1 figure, in which data is to be selected (closed gui loop, showing ONLY that figure)
"""
import matplotlib.pyplot as plt
import time
from matplotlib.widgets import RectangleSelector
import numpy as np

def showPlot_v1(display_mode : str, figures : list):
    """
    Display one or more plots with the specified display mode.
    Parameters:
    - display_mode: A string that specifies the display mode. It can be:
    :param display_mode: A string that specifies the display mode. It can be:
        - 'none': Do not display the plots.
        - 'timed': Display the plots for 3 seconds.
        - 'manual': Display the plots until manually closed.
    - figures: A list of matplotlib figure objects to be displayed.
    :param figures: A list of matplotlib figure objects to be displayed.
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
    return

def showPlot_v2(display_mode : str, figures : list):
    """
    Display one or more plots with the specified display mode.
    Parameters:
    :param display_mode: A string that specifies the display mode. It can be:
        - 'none': Do not display the plots.
        - 'timed': Display the plots for 3 seconds.
        - 'manual': Display the plots until manually closed.
    :param figures: A list of matplotlib figure objects to be displayed.
    """
    if display_mode == 'none':
        return

    figs_min = []
    figs_interest = []
    print(plt.get_fignums())
    for i in plt.get_fignums():
        fig = plt.figure(i)
        if not fig in figures:
            figs_min.append(fig)
            plt.close(fig)
        else:
            figs_interest.append(fig)

    if display_mode == 'timed':
        for fig in figs_interest:
            plt.figure(fig.number)
            plt.show(block=False)
            plt.pause(3)
            plt.close(fig)

    elif display_mode == 'manual':
        for fig in figs_interest:
            plt.figure(fig.number)
            plt.show(block=True)
    else:
        raise ValueError("Invalid display_mode. Use 'none', 'timed', or 'manual'.")

    for fig in figs_min:        #reopen closed figures
        plt.figure(fig)
    return

def showPlot_v3(display_mode: str, figures: list):
    """
    Display one or more plots with the specified display mode.
    Parameters:
    - display_mode: A string that specifies the display mode. It can be:
    :param display_mode: A string that specifies the display mode. It can be:
        - 'none': Do not display the plots.
        - 'timed': Display the plots for 3 seconds.
        - 'manual': Display the plots until manually closed.
    - figures: A list of matplotlib figure objects to be displayed.
    :param figures: A list of matplotlib figure objects to be displayed.
    """

    if display_mode == 'none':
        return

    figs_min = []
    figs_interest = []
    print(plt.get_fignums())
    for i in plt.get_fignums():
        fig = plt.figure(i)
        if not fig in figures:
            figs_min.append(fig)
        else:
            figs_interest.append(fig)

    if display_mode == 'timed':     #Stay in loop for 3 seconds to stop code from executing further
        for fig in figs_interest:
            fig.show()
            fig.waitforbuttonpress(3)
            plt.close(fig)

    elif display_mode == 'manual':
        for fig in figs_interest:
            fig.show()

    else:
        raise ValueError("Invalid display_mode. Use 'none', 'timed', or 'manual'.")
    return



def showPlot_v4(display_mode: str, fig, ax, x_data : np.ndarray, y_data : np.ndarray):
    """
    THIS SEEMS TO WORK JUST FINE:
    only show desired figure with fig.show(), then keep code loop closed after creating highlighter object untill
    the figure is closed manually

    :param display_mode:
    :param fig:
    :param ax:
    :param x_data:
    :param y_data:
    :return:
    """
    if display_mode == 'manual_interact':
        fig.show()
        closed = [False]
        def on_close(event):
            closed[0] = True

        # Connect the close event to the figure
        fig.canvas.mpl_connect('close_event', on_close)

        highlighter = Highlighter(ax, x_data, y_data)

        # Run a loop to block until the figure is closed
        while not closed[0]:
            fig.canvas.flush_events()

        selected_regions = highlighter.mask


class HighlighterV2:
    def __init__(self, ax, x, y):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.x, self.y = x, y
        self.mask = np.zeros(x.shape, dtype=bool)
        self._highlight = ax.scatter([], [], s=200, color='yellow', zorder=10)

        # Initialize RectangleSelector and set it to be active
        self.selector = RectangleSelector(
            ax, self.on_select, useblit=True, interactive=True
        )
        self.selector.set_active(True)

    def on_select(self, event1, event2):
        """Callback for RectangleSelector; updates mask and highlights points."""
        self.mask |= self.inside(event1, event2)
        xy = np.column_stack([self.x[self.mask], self.y[self.mask]])
        self._highlight.set_offsets(xy)
        self.canvas.draw()

    def inside(self, event1, event2):
        """Returns a boolean mask of points inside the rectangle defined by event1 and event2."""
        x0, x1 = sorted([event1.xdata, event2.xdata])
        y0, y1 = sorted([event1.ydata, event2.ydata])
        mask = ((self.x > x0) & (self.x < x1) & (self.y > y0) & (self.y < y1))
        return mask

class Highlighter(object):
    def __init__(self, ax, x, y):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.x, self.y = x, y
        self.mask = np.zeros(x.shape, dtype=bool)
        self._highlight = ax.scatter([], [], s=200, color='yellow', zorder=10)
        self.selector = RectangleSelector(ax, self, useblit=True)

    def __call__(self, event1, event2):
        self.mask |= self.inside(event1, event2)
        xy = np.column_stack([self.x[self.mask], self.y[self.mask]])
        self._highlight.set_offsets(xy)
        self.canvas.draw()

    def inside(self, event1, event2):
        """Returns a boolean mask of the points inside the rectangle defined by
        event1 and event2."""
        # Note: Could use points_inside_poly, as well
        x0, x1 = sorted([event1.xdata, event2.xdata])
        y0, y1 = sorted([event1.ydata, event2.ydata])
        mask = ((self.x > x0) & (self.x < x1) &
                (self.y > y0) & (self.y < y1))
        return mask

def main():
    fig1, ax1 = plt.subplots()
    ax1.plot([1, 2, 3], [1, 4, 9], label="Figure 1")
    ax1.legend()
    fig1.suptitle("Figure 1")

    fig2, ax2 = plt.subplots()
    ax2.plot([1, 2, 3], [1, 2, 3], label="Figure 2")
    ax2.legend()
    fig2.suptitle("Figure 2")

    fig3, ax3 = plt.subplots()
    ax3.plot([1, 2, 3], [9, 5, 1], label="Figure 3")
    ax3.legend()
    fig3.suptitle("Figure 3")

    #showPlot_v3('manual_interact', [fig2])

    fig, ax = plt.subplots(figsize=(10, 10))
    #y = np.array([9, 5, 1, 1, 1, 5, 6, 2, 7])
    y = [9, 5, 1, 1, 1, 5, 6, 2, 7]
    x = np.arange(0, len(y))
    ax.scatter(x, y)
    showPlot_v4('manual_interact', fig, ax, x, y)
    print("halla")
    #plt.show()

if __name__ == '__main__':
    main()
    quit()