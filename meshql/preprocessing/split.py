import numpy as np
import cadquery as cq
from typing import Callable, Literal, Optional, Sequence, Union, cast
from meshql.ql import GeometryQL
from meshql.selector import Selection
from meshql.utils.cq import CQUtils
from meshql.utils.cq_linq import CQLinq, SetOperation
from meshql.utils.split import MultiFaceAxis, SnapType, SplitUtils
from meshql.utils.types import (
    Axis,
    LineTuple,
    OrderedSet,
    VectorSequence,
    to_2d_array,
    to_array,
    to_vec,
)
from jupyter_cadquery import show
from copy import deepcopy


class Split:
    def __init__(self, ql: GeometryQL) -> None:
        self.ql = ql
        self.pending_splits = list[list[cq.Face]]()

    def apply(self, refresh: bool = False):
        split_faces = [
            split_face
            for split_face_group in self.pending_splits
            for split_face in split_face_group
        ]
        self.ql._workplane = SplitUtils.split_workplane(self.ql._workplane, split_faces)
        self.pending_splits = []
        if refresh:
            self.refresh()
        return self

    def refresh(self):
        self.ql._ctx.region_groups = None
        self.face_edge_groups = CQLinq.groupBy(self.ql._workplane, "face", "edge")

    def push(self, split_shapes: Union[cq.Shape, Sequence[cq.Shape]]):
        split_shapes = (
            [split_shapes] if isinstance(split_shapes, cq.Shape) else list(split_shapes)
        )
        self.pending_splits += [split_shapes]
        return self

    def show(self, theme: Literal["light", "dark"] = "light"):
        show(
            self.ql._workplane.newObject(
                self.pending_splits[-1] if len(self.pending_splits) else []
            ),
            theme=theme,
        )
        return self

    def from_plane(
        self,
        base_pnt: VectorSequence = (0, 0, 0),
        angle: VectorSequence = (0, 0, 1),
        sizing: Literal["maxDim", "infinite"] = "maxDim",
    ):
        split_face = SplitUtils.make_plane_split_face(
            self.ql._workplane, base_pnt, angle, sizing
        )
        self.push(split_face)
        return self

    def _select_from(
        self,
        select_type: Union[GeometryQL, Selection],
        region_set_operation: Optional[SetOperation] = None,
    ):
        ql = ql if isinstance(select_type, GeometryQL) else self.ql
        selection = (
            select_type._selection
            if isinstance(select_type, GeometryQL)
            else select_type
        )
        copied_selection = deepcopy(selection)
        if region_set_operation:
            copied_selection.region_set_operation = region_set_operation
        return ql.select(copied_selection, "edge")

    def from_ratios(
        self,
        start_select: Union[GeometryQL, Selection],
        end_select: Union[GeometryQL, Selection],
        ratios: list[float],
        dir: Literal["away", "towards", "both"],
        snap: SnapType = False,
        angle_offset: VectorSequence = (0, 0, 0),
    ):
        # if snap != False and len(self.pending_splits) > 0:
        #     self.apply(refresh=True)
        # elif (start_ql.type or end_ql.type) and self.ql._ctx.group_types is None:
        #     self.refresh()
        self.apply(refresh=True)

        offset = to_vec(np.radians(list(angle_offset)))
        start_ql = self._select_from(start_select, region_set_operation="intersection")
        start_paths = CQLinq.sortByConnect(start_ql._workplane.vals())
        start_edges = [path.edge for path in start_paths]

        end_ql = self._select_from(end_select, region_set_operation="intersection")
        end_paths = CQLinq.sortByConnect(end_ql._workplane.vals())
        end_edges = [path.edge for path in end_paths]

        assert len(start_edges) > 0, "no selection for start present"
        assert len(end_edges) > 0, "no selection for end present"

        if len(start_edges) > 1:
            start_cw = CQUtils.is_clockwise(start_edges[0], start_edges[1])
            start_edges = start_edges if start_cw else reversed(start_edges)
        
        if len(end_edges) > 1:
            end_cw = CQUtils.is_clockwise(end_edges[0], end_edges[1])
            end_edges = end_edges if end_cw else reversed(end_edges)
        
        start_wire = cq.Wire.assembleEdges(start_edges)
        end_wire = cq.Wire.assembleEdges(end_edges)

        for ratio in ratios:
            start_point = start_wire.positionAt(ratio)
            end_point = end_wire.positionAt(
                0.5 - ratio if ratio <= 0.5 else 1.5 - ratio
            )
            edge = cq.Edge.makeLine(start_point, end_point)

            if (
                start_ql._selection.type
                and start_ql._selection.type == end_ql._selection.type
            ):
                target = self.ql._ctx.region_groups[start_ql._selection.type]
            else:
                target = self.ql._workplane

            nearest_face = cast(
                cq.Face, CQLinq.find_nearest(target, edge, shape_type="face")
            )
            normal_vec = (
                CQUtils.normalize(nearest_face.normalAt((start_point + end_point) / 2))
                + offset
            )
            projected_edge = edge.project(nearest_face, normal_vec).Edges()[0]
            assert isinstance(projected_edge, cq.Edge), "Projected edge is single edge"
            self.from_edge(projected_edge, normal_vec, dir, snap)

        return self

    def group(self, on_split: Callable[["GeometryQL"], "Split"]):
        self.apply(refresh=True)
        self.pending_splits = on_split(self.ql).pending_splits
        return self

    def from_normals(
        self,
        select: Optional[GeometryQL] = None,
        dir: Optional[Literal["away", "towards", "both"]] = None,
        axis: Union[MultiFaceAxis, list[MultiFaceAxis]] = "avg",
        snap: SnapType = False,
        angle_offset: VectorSequence = (0, 0, 0),
    ):
        self.apply(refresh=True)

        selected_ql = self._select_from(select, region_set_operation="difference")

        # if snap != False and len(self.pending_splits) > 0:
        #     self.apply(refresh=True)
        # elif selection.type and self.ql.group_types is None:
        #     self.refresh()

        offset = to_vec(np.radians(list(angle_offset)))
        filtered_edges = selected_ql._workplane.vals()
        assert len(filtered_edges) > 0, "No edges found for selection"
        split_faces = list[cq.Shape]()
        snap_edges = OrderedSet[cq.Edge]()
        for edge in filtered_edges:
            faces = self.face_edge_groups[edge]
            split_face = None
            for _axis in axis if isinstance(axis, list) else [axis]:
                if not isinstance(_axis, str) and dir is None:
                    dir = "towards"
                else:
                    dir = (
                        dir or "away"
                        if selected_ql._selection.type == "interior"
                        else "towards"
                    )

                normal_vec = CQUtils.get_normal_vec(faces, cast(Axis, _axis), offset)
                curr_split_face = SplitUtils.make_edge_split_face(
                    self.ql._workplane, edge, normal_vec, dir, snap, snap_edges
                )
                if split_face is None:
                    split_face = curr_split_face
                else:
                    split_face = split_face.fuse(curr_split_face)
            assert split_face, "No split face found"
            split_faces.append(split_face)

        self.push(split_faces)
        return self

    def from_anchor(
        self,
        anchor: Union[list[VectorSequence], VectorSequence] = (0, 0, 0),
        angle: Union[list[VectorSequence], VectorSequence] = (0, 0, 0),
        snap_tolerance: Optional[float] = None,
        until: Literal["next", "all"] = "next",
    ):
        anchors = [anchor] if isinstance(anchor, tuple) else anchor
        angles = [angle] if isinstance(angle, tuple) else angle
        assert len(anchors) == len(angles), "anchors and dirs must be the same length"

        edges = []
        for anchor, angle in zip(anchors, angles):
            split_face = SplitUtils.make_plane_split_face(
                self.ql._workplane, anchor, angle
            )
            if until == "next":
                intersected_vertices = (
                    self.ql._workplane.intersect(cq.Workplane(split_face))
                    .vertices()
                    .vals()
                )
                intersect_vertex = CQLinq.find_nearest(
                    intersected_vertices, to_vec(anchor), snap_tolerance
                )
                assert intersect_vertex, "No intersecting vertex found"
                edges.append((anchor, intersect_vertex.Center().toTuple()))
            else:
                edges.append(split_face)
        self.from_lines(edges)
        return self

    def from_pnts(self, pnts: Sequence[VectorSequence]):
        pnt_vecs = [to_vec(pnt) for pnt in pnts]
        split_face = cq.Face.makeFromWires(cq.Wire.makePolygon(pnt_vecs))
        self.push(split_face)
        return self

    def from_edge(
        self,
        edge: cq.Edge,
        axis: Axis = "Z",
        dir: Literal["away", "towards", "both"] = "both",
        snap: SnapType = False,
    ):
        split_face = SplitUtils.make_edge_split_face(
            self.ql._workplane, edge, axis, dir, snap
        )
        self.push(split_face)
        return self

    def from_lines(
        self,
        lines: Union[list[LineTuple], LineTuple],
        axis: Axis = "Z",
        dir: Literal["away", "towards", "both"] = "both",
    ):
        max_dim = self.ql._workplane.findSolid().BoundingBox().DiagonalLength * 10

        if isinstance(lines, tuple):
            edges_pnts = np.array(
                [to_2d_array(lines), to_2d_array(lines)], dtype=np.float64
            )
        elif isinstance(lines, list) and len(lines) == 1:
            edges_pnts = np.array(
                [to_2d_array(lines[0]), to_2d_array(lines[0])], dtype=np.float64
            )
        else:
            edges_pnts = np.array(
                [to_2d_array(line) for line in lines], dtype=np.float64
            )
        normal_vector = to_array(CQUtils.normalize(to_vec(axis)))

        if dir in ("both", "towards"):
            edges_pnts[0] += max_dim * normal_vector
        if dir in ("both", "away"):
            edges_pnts[-1] -= max_dim * normal_vector

        side1 = edges_pnts[:, 0].tolist()
        side2 = edges_pnts[:, 1].tolist()
        wire_pnts = [side1[0], *side2, *side1[1:][::-1]]
        self.from_pnts(wire_pnts)
        return self
