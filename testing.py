#For plotting of the 3-d height profiles. Import a pickle file with coordinates & z height saved
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cbook, cm
from matplotlib.colors import LightSource
from matplotlib.animation import FFMpegWriter

def main():
    ffmpegPath = 'C:\\Users\\Sander PC\\Desktop\\ffmpeg-7.1-essentials_build\\bin\\ffmpeg.exe'      #thuis PC
    metadata2 = dict(title='3D Height Movie2', artist='Sjendieee')
    writer2 = FFMpegWriter(fps=15, metadata=metadata2)
    #from z-axis= [1400um -> 1.4mm]
    #from viewinit (30, 130) -> (0, 130)
    #from (zlim = [0, 1.50]) -> (zlim = [0, 0.020])

    zmaxrange = np.linspace(1400, 1.4, 50)
    viewrange = np.linspace(30, 0, 50)

    conversionXY = 0.0005446623093681918    #pixels
    cmap_minval = 0
    cmap_maxval = 1
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    data = pickle.load(open("plot3d_data.pickle", 'rb'))
    with writer2.saving(fig, "h_profile_realistic.mp4", 300):
        for i in [0, 1]:#range(0,len(data[0])):
            x = np.array(data[0][i]) * conversionXY  #convert pixels -> mm
            y = np.array(data[1][i]) * conversionXY #convert pixels -> mm
            z = np.array(data[2][i]) / 1000 / 1000 #convert nm->um -> mm
            ax.scatter(x,y,z, c=z, cmap='jet', edgecolor='none')
            if min(z) < cmap_minval:
                cmap_minval = min(z)
            if max(z) > cmap_maxval:
                cmap_maxval = max(z)
        #ax.set(xlabel = 'X-Coord', ylabel = 'Y-Coord', zlabel = 'Height (nm)', title = f'Spatial Height Profile Colormap')
        ax.set(xlabel = 'x-distance (mm)', ylabel = 'y-distance (mm)', zlabel = 'Height (mm)', title = f'Spatial Height Profile Colormap')
        ax.set(zlim = [0, zmaxrange[i]])
        #cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm = plt.Normalize(cmap_minval, cmap_maxval), cmap = 'jet'), label='height (nm)', orientation='vertical')
        ax.view_init(viewrange[i], 130)
        plt.show()
        writer2.grab_frame()
        ax.clear()

if __name__ == "__main__":
    main()
    exit()