"""
Testing showing of only desired plot, untill clicked away
"""
import matplotlib.pyplot as plt

def showPlot(display_mode : str, figures : list):
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

    elif display_mode == 'manual':
        for fig in figs_interest:
            plt.figure(fig.number)
            plt.show(block=True)
    else:
        raise ValueError("Invalid display_mode. Use 'none', 'timed', or 'manual'.")


    for fig in figs_min:        #reopen closed figures
        plt.figure(fig)



    # if display_mode == 'timed':
    #     for fig in figures:
    #         fig.show()
    #     plt.pause(3)
    #     for fig in figures:
    #         plt.close(fig)
    # elif display_mode == 'manual':
    #     for fig in figures:
    #         fig.show()
    #     plt.pause(15)    #some very long time
    #     for fig in figures:
    #         plt.close(fig)


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

    print(plt.get_fignums())
    showPlot('timed', [fig2])
    print("halla")
    plt.show()

if __name__ == '__main__':
    main()
    quit()