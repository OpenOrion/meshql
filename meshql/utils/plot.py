import numpy as np
import numpy.typing as npt
from typing import Sequence, Union
from plotly import graph_objects as go
from meshql.utils.types import NumpyFloat

def add_plot(coords: npt.NDArray[NumpyFloat], fig: go.Figure=go.Figure(), label: str="Plot"):
    dim = 3 if np.all(coords[:,2]) else 2
    if dim == 3:
        fig.add_scatter3d(
            x=coords[:,0],
            y=coords[:,1],
            z=coords[:,2],
            name=label,
            mode="lines+markers",
            marker=dict(size=1)       
        )
    else:
        fig.add_scatter(
            x=coords[:,0],
            y=coords[:,1],
            name=label,
            fill="toself",
            mode="lines+markers",
            marker=dict(size=1)       
        )

    return fig
