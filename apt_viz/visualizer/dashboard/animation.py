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

import plotly.graph_objects as go
from .data import collect_z_slice
import pandas as pd
import numpy as np
import io
from PIL import Image
import glob
import plotly.io as pio
from dashboard.colors import get_colorscale, get_color_from_index


def fetch_animation(df):
    """
    Creates the 2d animation that allows the user to sweep through the slices of the sample (along the z direction) by either playing the animation, or manually adjusting a slider.
    This is the one that is currently used.
    """

    x_min = df['midpoint_x'].min()
    x_max = df['midpoint_x'].max()
    y_min = df['midpoint_y'].min()
    y_max = df['midpoint_y'].max()
    z_min = df['midpoint_z'].min()

    print(f"x: [{x_min}, {x_max}], y: [{y_min}, {y_max}]")
    z_vals = df['midpoint_z'].unique()
    z_vals.sort()

    clusters = df['community'].unique()
    clusters.sort()

    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    fig_dict["layout"]["xaxis"] = {"range": [x_min, x_max], "title": "x coordinate"}
    fig_dict["layout"]["yaxis"] = {"range": [y_min, y_max], "title": "y coordinate"}
    fig_dict["layout"]["hovermode"] = "closest"
    fig_dict["layout"]["legend"] = {"title": {"text": "Community"}}
    fig_dict["layout"]["width"] = 700
    fig_dict["layout"]["height"] = 680
    #fig_dict["layout"]["colorscale"] = {"sequential": "Turbo"}
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 500, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": 0,
                                                                        "easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Z Layer:",
            "suffix": f"",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 0, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    # assuming that the cluster numbers aren't random. [0, 1, 2] -> 3 clusters
    total_clusters = max(clusters) + 1
    colorscale = get_colorscale(total_clusters)
    color_lst = get_color_from_index(total_clusters, 0)

    # Data Time!
    z_val = z_min
    for a_cluster in clusters:
        dataset_by_z = collect_z_slice(z_val, df)
        dataset_by_z_and_cluster = dataset_by_z[dataset_by_z["community"] == a_cluster]

        data_dict = {
            "x": list(dataset_by_z_and_cluster['midpoint_x']),
            "y": list(dataset_by_z_and_cluster['midpoint_y']),
            "mode": "markers",
            "text": list(dataset_by_z_and_cluster['community']),
            "marker": {
                "sizemode": "area",
                "size": 24,
                # "color": df['community'],
                # "colorscale": colorscale
            },
            "name": str(a_cluster)
        }
        fig_dict["data"].append(data_dict)
    # dataset_by_z = collect_z_slice(z_val, df)
    # fig_dict["data"] = go.Scatter(mode='markers', x=dataset_by_z['midpoint_x'], y=dataset_by_z['midpoint_y'],
    #                            marker={"color": dataset_by_z['kmeans'],
    #                                    "colorscale": "Viridis"})

    for z_ind, z_val in enumerate(z_vals):
        frame = {"data": [], "name": str(z_val)}
        for a_cluster in clusters:
            dataset_by_z = collect_z_slice(z_val, df)
            dataset_by_z_and_cluster = dataset_by_z[dataset_by_z["community"] == a_cluster]

            data_dict = {
                "x": list(dataset_by_z_and_cluster['midpoint_x']),
                "y": list(dataset_by_z_and_cluster['midpoint_y']),
                "mode": "markers",
                "text": list(dataset_by_z_and_cluster['community']),
                "marker": {
                    "sizemode": "area",
                    "size": 24,
                    # "color": df['community'],
                    # "colorscale": colorscale
                },
                "name": str(a_cluster)
            }
            frame["data"].append(data_dict)
        # dataset_by_z = collect_z_slice(z_val, df)
        # frame["data"] = go.Scatter(mode='markers', x=dataset_by_z['midpoint_x'], y=dataset_by_z['midpoint_y'],
        #                            marker={"color": dataset_by_z['kmeans'],
        #                                    "colorscale": "Viridis"})
        fig_dict["frames"].append(frame)
        slider_step = {"args": [
            [z_val],
            {"frame": {"duration": 150, "redraw": False},
             "mode": "immediate",
             "transition": {"duration": 0}}
        ],
            "label": z_ind,
            "method": "animate"
        }
        sliders_dict["steps"].append(slider_step)

    fig_dict["layout"]["sliders"] = [sliders_dict]

    fig = go.Figure(fig_dict)

    fig.update_layout(
        plot_bgcolor="rgb(255, 244, 207)"
    )

    return fig

#
# def plotly_fig2array(fig):
#     #convert Plotly fig to  an array
#     fig_bytes = fig.to_image(format="png")
#     buf = io.BytesIO(fig_bytes)
#     img = Image.open(buf)
#     return np.asarray(img)
#
#
# def save_frames(fig):
#     images = []
#     for frame in range(5):
#         print(f"on frame {frame} of {len(fig.frames)}")
#         frame_fig = go.Figure(fig.frames[frame].data, frames=fig.frames, layout=fig.layout)
#         frame_fig.write_image(f"../../animated_z_slices/{frame}.png")
#     return
#
#
# def create_gif(folder):
#     images = []
#     for f in glob.iglob(folder):
#         images.append(np.asarray(Image.open(f)))
#     images[0].save('../../animated_z_slices.gif',
#                    save_all=True, append_images=images[1:], optimize=False, duration=3000, loop=1)
#     return
#
#
# if __name__ == "__main__":
#     pio.kaleido.scope.mathjax = None
#     df = pd.read_csv("../../data/clustered_data.csv")
#     fig = fetch_animation(df)
#     save_frames(fig)
#     create_gif("../../animated_z_slices/*")
#     z_vals = df['midpoint_z'].unique()
#     z_vals.sort()
#
