#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   vis.py
#        \author   chenghuige
#          \date   2023-06-28 16:27:41.810108
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import *

# https://www.kaggle.com/code/leonidkulyk/eda-aslfr-animated-visualization

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

def map_new_to_old_style(sequence):
  types = []
  landmark_indexes = []
  for column in list(sequence.columns)[1:544]:
    parts = column.split("_")
    if len(parts) == 4:
      types.append(parts[1] + "_" + parts[2])
    else:
      types.append(parts[1])

    landmark_indexes.append(int(parts[-1]))

  data = {
      "frame": [],
      "type": [],
      "landmark_index": [],
      "x": [],
      "y": [],
      "z": []
  }

  for index, row in sequence.iterrows():
    data["frame"] += [int(row.frame)] * 543
    data["type"] += types
    data["landmark_index"] += landmark_indexes

    for _type, landmark_index in zip(types, landmark_indexes):
      data["x"].append(row[f"x_{_type}_{landmark_index}"])
      data["y"].append(row[f"y_{_type}_{landmark_index}"])
      data["z"].append(row[f"z_{_type}_{landmark_index}"])

  return pd.DataFrame.from_dict(data)


# assign desired colors to landmarks
def assign_color(row):
  if row == 'face':
    return 'red'
  elif 'hand' in row:
    return 'dodgerblue'
  else:
    return 'green'


# specifies the plotting order
def assign_order(row):
  if row.type == 'face':
    return row.landmark_index + 101
  elif row.type == 'pose':
    return row.landmark_index + 30
  elif row.type == 'left_hand':
    return row.landmark_index + 80
  else:
    return row.landmark_index


def visualise2d_landmarks(parquet_df, title=""):
  connections = [
      [
          0,
          1,
          2,
          3,
          4,
      ],
      [0, 5, 6, 7, 8],
      [0, 9, 10, 11, 12],
      [0, 13, 14, 15, 16],
      [0, 17, 18, 19, 20],
      [38, 36, 35, 34, 30, 31, 32, 33, 37],
      [40, 39],
      [52, 46, 50, 48, 46, 44, 42, 41, 43, 45, 47, 49, 45, 51],
      [42, 54, 56, 58, 60, 62, 58],
      [41, 53, 55, 57, 59, 61, 57],
      [54, 53],
      [
          80,
          81,
          82,
          83,
          84,
      ],
      [80, 85, 86, 87, 88],
      [80, 89, 90, 91, 92],
      [80, 93, 94, 95, 96],
      [80, 97, 98, 99, 100],
  ]

  parquet_df = map_new_to_old_style(parquet_df)
  frames = sorted(set(parquet_df.frame))
  first_frame = min(frames)
  parquet_df['color'] = parquet_df.type.apply(lambda row: assign_color(row))
  parquet_df['plot_order'] = parquet_df.apply(lambda row: assign_order(row),
                                              axis=1)
  first_frame_df = parquet_df[parquet_df.frame == first_frame].copy()
  first_frame_df = first_frame_df.sort_values(["plot_order"
                                              ]).set_index('plot_order')

  frames_l = []
  for frame in frames:
    filtered_df = parquet_df[parquet_df.frame == frame].copy()
    filtered_df = filtered_df.sort_values(["plot_order"
                                          ]).set_index("plot_order")
    traces = [
        go.Scatter(x=filtered_df['x'],
                   y=filtered_df['y'],
                   mode='markers',
                   marker=dict(color=filtered_df.color, size=9))
    ]

    for i, seg in enumerate(connections):
      trace = go.Scatter(
          x=filtered_df.loc[seg]['x'],
          y=filtered_df.loc[seg]['y'],
          mode='lines',
      )
      traces.append(trace)
    frame_data = go.Frame(data=traces, traces=[i for i in range(17)])
    frames_l.append(frame_data)

  traces = [
      go.Scatter(x=first_frame_df['x'],
                 y=first_frame_df['y'],
                 mode='markers',
                 marker=dict(color=first_frame_df.color, size=9))
  ]
  for i, seg in enumerate(connections):
    trace = go.Scatter(x=first_frame_df.loc[seg]['x'],
                       y=first_frame_df.loc[seg]['y'],
                       mode='lines',
                       line=dict(color='black', width=2))
    traces.append(trace)
  fig = go.Figure(data=traces, frames=frames_l)

  fig.update_layout(
      width=500,
      height=800,
      scene={
          'aspectmode': 'data',
      },
      updatemenus=[{
          "buttons": [
              {
                  "args": [
                      None, {
                          "frame": {
                              "duration": 100,
                              "redraw": True
                          },
                          "fromcurrent": True,
                          "transition": {
                              "duration": 0
                          }
                      }
                  ],
                  "label": "&#9654;",
                  "method": "animate",
              },
              {
                  "args": [[None], {
                      "frame": {
                          "duration": 0,
                          "redraw": False
                      },
                      "mode": "immediate",
                      "transition": {
                          "duration": 0
                      }
                  }],
                  "label": "&#9612;&#9612;",
                  "method": "animate",
              },
          ],
          "direction": "left",
          "pad": {
              "r": 100,
              "t": 100
          },
          "font": {
              "size": 20
          },
          "type": "buttons",
          "x": 0.1,
          "y": 0,
      }],
  )
  camera = dict(up=dict(x=0, y=-1, z=0), eye=dict(x=0, y=0, z=2.5))
  fig.update_layout(title_text=title, title_x=0.5)
  fig.update_layout(scene_camera=camera, showlegend=False)
  fig.update_layout(
      xaxis=dict(visible=False),
      yaxis=dict(visible=False),
  )
  fig.update_yaxes(autorange="reversed")

  fig.show()
  
visualize = visualise2d_landmarks


def get_phrase(df, file_id, sequence_id):
  return df[np.logical_and(df.file_id == file_id,
                           df.sequence_id == sequence_id)].phrase.iloc[0]
  

