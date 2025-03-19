'''
This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or
any information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.

PACIFIC NORTHWEST NATIONAL LABORATORY
operated by
BATTELLE
for the
UNITED STATES DEPARTMENT OF ENERGY
under Contract DE-AC05-76RL01830
'''

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import math
import numpy as np

def get_colorscale(n_colors):
    print("getting colors")
    color_indices = list(range(n_colors))
    normalized_indices = np.linspace(0, 1, n_colors+1)
    cmap = mc.ListedColormap(plt.cm.tab20.colors[:n_colors])
    small_rgb = [mc.to_rgb(mc.rgb2hex(cmap(index))) for index in color_indices]
    big_rgb = [(math.floor(255*rgb[0]), math.floor(255*rgb[1]), math.floor(255*rgb[2])) for rgb in small_rgb]
    colorscale = []
    for index in color_indices:
        norm_index = normalized_indices[index]
        color = f"rgb{big_rgb[index]}"
        colorscale.append([norm_index, color])
        if index <=n_colors:
            next_index = normalized_indices[index+1]
            colorscale.append([next_index, color])
    return colorscale

def get_color_from_index(n_colors, index):
    print(f"getting specific color from index {index}")
    color_indices = list(range(n_colors))
    cmap = mc.ListedColormap(plt.cm.tab20.colors[:n_colors])
    small_rgb = [mc.to_rgb(mc.rgb2hex(cmap(index))) for index in color_indices]
    big_rgb = [(math.floor(255*rgb[0]), math.floor(255*rgb[1]), math.floor(255*rgb[2])) for rgb in small_rgb]
    # print(f"rgb values are {big_rgb}")
    # return big_rgb[index]
    return big_rgb
