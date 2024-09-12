import numpy as np
import numpy.typing as npt
from typing import Sequence, Union, cast
from plotly import graph_objects as go
from meshql.utils.cq_linq import CQLinq
from meshql.utils.types import NumpyFloat
from meshql.utils.shapes import get_sampling
from cadquery.cq import CQObject
import cadquery as cq


def add_plot(
    coords: npt.NDArray[NumpyFloat], fig: go.Figure = go.Figure(), label: str = "Plot"
):
    dim = 3 if np.all(coords[:, 2]) else 2
    if dim == 3:
        fig.add_scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            name=label,
            mode="lines+markers",
            marker=dict(size=1),
        )
    else:
        fig.add_scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            name=label,
            fill="toself",
            mode="lines+markers",
            marker=dict(size=1),
        )

    return fig


def plot_cq(
    target: Union[
        cq.Workplane,
        CQObject,
        Sequence[CQObject],
        Sequence[Sequence[CQObject]],
    ],
    title: str = "Plot",
    samples_per_spline: int = 50,
    ctx=None,
):
    from meshql.gmsh.entity import CQEntityContext

    ctx = cast(CQEntityContext, ctx)

    fig = go.Figure(layout=go.Layout(title=go.layout.Title(text=title)))
    if isinstance(target, cq.Workplane):
        edge_groups = [[edge] for edge in CQLinq.select(target, "edge")]
    elif isinstance(target, CQObject):
        edge_groups = [[edge] for edge in CQLinq.select(target, "edge")]
    elif isinstance(target, Sequence) and isinstance(target[0], CQObject):
        edge_groups = [cast(Sequence[CQObject], target)]

    else:
        target = np.cast(Sequence[Sequence], target)
        edge_groups = cast(Sequence[Sequence[CQObject]], target)

    for i, edges in enumerate(edge_groups):

        edge_name = f"Edge{ctx.select(edges[0]).tag}" if ctx else f"Edge{i}"
        sampling = get_sampling(0, 1, samples_per_spline, False)
        coords = np.concatenate([np.array([vec.toTuple() for vec in edge.positions(sampling)], dtype=NumpyFloat) for edge in edges])  # type: ignore
        add_plot(coords, fig, edge_name)

    fig.layout.yaxis.scaleanchor = "x"  # type: ignore
    fig.show()
