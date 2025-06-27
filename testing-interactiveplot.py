"""
GPT generated code. function: Create an interactive graph, which allows the user to slide between many different
'graphs', ech generated upon sliding to it (to save memory)
"""
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# Simulate 7000 datasets (replace this with your real data)
n_graphs = 7000
x = np.linspace(0, 10, 100)
data = [np.sin(x + i * 0.1) for i in range(n_graphs)]  # list of y-arrays

# Initialize figure
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)  # space for slider

# Initial plot
line, = ax.plot(x, data[0])
ax.set_title(f"Graph 0 / {n_graphs}")

# Add slider
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Graph Index', 0, n_graphs - 1, valinit=0, valfmt='%0.0f')

# Update function
def update(val):
    idx = int(slider.val)
    line.set_ydata(data[idx])
    ax.set_title(f"Graph {idx} / {n_graphs}")
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()