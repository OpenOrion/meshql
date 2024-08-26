import numpy as np
import cadquery as cq
from cadquery.cq import CQObject
from typing import Callable, Iterable, Literal, Optional, Sequence, Union, cast
from meshql.selector import Selectable, Selection
from meshql.utils.cq import CQCache, CQExtensions, CQGroupTypeString, CQLinq
from meshql.utils.shapes import get_sampling
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

SplitAt = Literal["end", "per"]
SnapType = Union[bool, Literal["closest"], float]
MultiFaceAxis = Union[Axis, Literal["avg", "face1", "face2"]]


def normalize(vec: cq.Vector):
    return vec / vec.Length


class Split(Selectable):
    def __init__(
        self,
        workplane: cq.Workplane,
        use_cache: bool = False,
        max_dim: Optional[float] = None,
        tol: Optional[float] = None,
    ) -> None:
        self.workplane = self.initial_workplane = workplane
        self.type_groups: dict[CQGroupTypeString, OrderedSet[CQObject]] = {}

        self.use_cache = use_cache
        self.tol = tol
        self.pending_splits = list[list[cq.Face]]()
        self.max_dim = (
            max_dim or workplane.findSolid().BoundingBox().DiagonalLength * 10
        )

    def apply(self, refresh: bool = False):
        split_faces = [
            split_face
            for split_face_group in self.pending_splits
            for split_face in split_face_group
        ]
        self.workplane = split_workplane(self.workplane, split_faces, self.use_cache)
        self.pending_splits = []
        if refresh:
            self.refresh_type_groups()
        return self

    def refresh_type_groups(self):
        self.type_groups = CQLinq.groupByTypes(
            self.workplane, self.max_dim, self.tol, self.type_groups
        )
        self.face_edge_groups = CQLinq.groupBy(self.workplane, "face", "edge")

    def push(self, split_shapes: Union[cq.Shape, Sequence[cq.Shape]]):
        split_shapes = (
            [split_shapes] if isinstance(split_shapes, cq.Shape) else list(split_shapes)
        )
        self.pending_splits += [split_shapes]
        return self

    def show(self, theme: Literal["light", "dark"] = "light"):
        assert len(self.pending_splits) > 0, "No split faces to show"
        show(self.workplane.newObject(self.pending_splits[-1]), theme=theme)
        return self

    def from_plane(
        self,
        base_pnt: VectorSequence = (0, 0, 0),
        angle: VectorSequence = (0, 0, 1),
        sizing: Literal["maxDim", "infinite"] = "maxDim",
    ):
        split_face = get_plane_split_face(self.workplane, base_pnt, angle, sizing)
        self.push(split_face)
        return self

    def from_ratios(
        self,
        start,
        end: Selection,
        ratios: list[float],
        dir: Literal["away", "towards", "both"],
        snap: SnapType = False,
        angle_offset: VectorSequence = (0, 0, 0),
    ):
        if snap != False and len(self.pending_splits) > 0:
            self.apply(refresh=True)
        elif (start.type or end.type) and len(self.type_groups) == 0:
            self.refresh_type_groups()

        offset = to_vec(np.radians(list(angle_offset)))
        start_selection = start.select(self, "edge", is_intersection=True)
        end_selection = end.select(self, "edge", is_intersection=True)

        assert len(start_selection) > 0, "no selection for start present"
        assert len(end_selection) > 0, "no selection for end present"

        start_wire = cq.Wire.assembleEdges(cast(list[cq.Edge], start_selection))
        end_wire = cq.Wire.assembleEdges(cast(list[cq.Edge], end_selection))

        for ratio in ratios:
            start_point = start_wire.positionAt(ratio)
            end_point = end_wire.positionAt(
                0.5 - ratio if ratio <= 0.5 else 1.5 - ratio
            )
            edge = cq.Edge.makeLine(start_point, end_point)

            if start.type and start.type == end.type:
                target = self.type_groups[start.type]
            else:
                target = self.workplane

            nearest_face = cast(
                cq.Face, CQLinq.find_nearest(target, edge, shape_type="face")
            )
            normal_vec = (
                normalize(nearest_face.normalAt((start_point + end_point) / 2)) + offset
            )
            projected_edge = edge.project(nearest_face, normal_vec).Edges()[0]
            assert isinstance(projected_edge, cq.Edge), "Projected edge is single edge"
            self.from_edge(projected_edge, normal_vec, dir, snap)

        return self

    def group(self, on_split: Callable[["Split"], "Split"]):
        self.apply(refresh=True)
        self.pending_splits = on_split(self).pending_splits
        return self

    def from_normals(
        self,
        selection: Selection,
        dir: Optional[Literal["away", "towards", "both"]] = None,
        axis: Union[MultiFaceAxis, list[MultiFaceAxis]] = "avg",
        snap: SnapType = False,
        angle_offset: VectorSequence = (0, 0, 0),
        is_initial: bool = False,
    ):
        if snap != False and len(self.pending_splits) > 0:
            self.apply(refresh=True)
        elif selection.type and len(self.type_groups) == 0:
            self.refresh_type_groups()

        offset = to_vec(np.radians(list(angle_offset)))
        filtered_edges = cast(
            Sequence[cq.Edge],
            selection.select(self, "edge", is_initial=is_initial, is_exclusive=True),
        )
        assert len(filtered_edges) > 0, "No edges found for selection"
        split_faces = list[cq.Shape]()
        snap_edges = OrderedSet[cq.Edge]()
        for edge in filtered_edges:
            faces = cast(OrderedSet[cq.Face], self.face_edge_groups[edge])
            split_face = None
            for _axis in axis if isinstance(axis, list) else [axis]:
                if not isinstance(_axis, str) and dir is None:
                    dir = "towards"
                else:
                    dir = dir or "away" if selection.type == "interior" else "towards"

                normal_vec = get_normal_vec(faces, cast(Axis, _axis), offset)
                curr_split_face = get_edge_split_face(
                    self.workplane, edge, normal_vec, dir, snap, snap_edges
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
            split_face = get_plane_split_face(self.workplane, anchor, angle)
            if until == "next":
                intersected_vertices = (
                    self.workplane.intersect(cq.Workplane(split_face)).vertices().vals()
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
        split_face = get_edge_split_face(self.workplane, edge, axis, dir, snap)
        self.push(split_face)
        return self

    def from_lines(
        self,
        lines: Union[list[LineTuple], LineTuple],
        axis: Axis = "Z",
        dir: Literal["away", "towards", "both"] = "both",
    ):
        if isinstance(lines, tuple):
            edges_pnts = np.array([to_2d_array(lines), to_2d_array(lines)], dtype=np.float64)
        elif isinstance(lines, list) and len(lines) == 1:
            edges_pnts = np.array([to_2d_array(lines[0]), to_2d_array(lines[0])], dtype=np.float64)
        else:
            edges_pnts = np.array([to_2d_array(line) for line in lines], dtype=np.float64)
        normal_vector = to_array(normalize(to_vec(axis)))

        if dir in ("both", "towards"):
            edges_pnts[0] += self.max_dim * normal_vector
        if dir in ("both", "away"):
            edges_pnts[-1] -= self.max_dim * normal_vector

        side1 = edges_pnts[:, 0].tolist()
        side2 = edges_pnts[:, 1].tolist()
        wire_pnts = [side1[0], *side2, *side1[1:][::-1]]
        self.from_pnts(wire_pnts)
        return self


def split_workplane(
    workplane: cq.Workplane, split_faces: Sequence[cq.Shape], use_cache: bool = True
):
    shape_combo = [*workplane.vals(), *split_faces]
    cache_exists = CQCache.get_cache_exists(shape_combo) if use_cache else False
    cache_file_name = CQCache.get_file_name(shape_combo) if use_cache else ""
    if use_cache and cache_exists:
        shape = CQCache.import_brep(cache_file_name)
    else:
        for split_face in split_faces:
            workplane = workplane.split(split_face)
        shape = CQExtensions.fuse_shapes(workplane.vals())
        if use_cache:
            CQCache.export_brep(shape, cache_file_name)
    return cq.Workplane(shape)


def get_split_face_from_edges(edge1: cq.Edge, edge2: cq.Edge, is_line_end: bool = True):
    if (edge2.endPoint().Center() - edge1.endPoint().Center()).Length < (
        edge2.startPoint().Center() - edge1.endPoint().Center()
    ).Length:
        near_pnt1, near_pnt2 = edge2.endPoint(), edge2.startPoint()
    else:
        near_pnt1, near_pnt2 = edge2.startPoint(), edge2.endPoint()

    split_edges = [
        edge1,
        cq.Edge.makeLine(edge1.endPoint(), near_pnt1),
        cq.Edge.makeLine(near_pnt1, near_pnt2) if is_line_end else edge2,
        cq.Edge.makeLine(near_pnt2, edge1.startPoint()),
    ]
    try:
        return cq.Face.makeFromWires(cq.Wire.assembleEdges(split_edges))
    except:
        return cq.Face.makeNSidedSurface(split_edges, [])


def get_plane_split_face(
    workplane: cq.Workplane,
    base_pnt: VectorSequence = (0, 0, 0),
    angle: VectorSequence = (0, 0, 1),
    sizing: Literal["maxDim", "infinite"] = "maxDim",
):
    maxDim = workplane.findSolid().BoundingBox().DiagonalLength * 10
    base_pnt_vec = to_vec(base_pnt)
    angle_vec = to_vec(np.radians(list(angle)))
    if sizing == "maxDim":
        return cq.Face.makePlane(maxDim, maxDim, base_pnt_vec, angle_vec)
    else:
        return cq.Face.makePlane(None, None, base_pnt_vec, angle_vec)


def get_normal_vec(
    faces: OrderedSet[cq.Face],
    axis: Optional[Union[Axis, Literal["avg", "face1", "face2"]]],
    offset: cq.Vector = cq.Vector(0, 0, 0),
):
    if axis is None:
        axis = "face1" if len(faces) == 1 else "avg"

    if axis == "avg":
        average_normal = np.average(
            [face.normalAt().toTuple() for face in faces], axis=0
        )
        norm_vec = cq.Vector(tuple(average_normal)) + offset
    elif axis == "face1":
        norm_vec = list(faces)[0].normalAt()
    elif axis == "face2":
        norm_vec = list(faces)[1].normalAt()
    else:
        norm_vec = to_vec(axis)
    return normalize(norm_vec + offset)


def get_edge_split_face(
    workplane: cq.Workplane,
    edge: cq.Edge,
    axis: Axis = "Z",
    dir: Literal["away", "towards", "both"] = "both",
    snap: SnapType = False,
    snap_edges=OrderedSet[cq.Edge](),
):
    maxDim = workplane.findSolid().BoundingBox().DiagonalLength * 10
    normal_vector = normalize(to_vec(axis))
    towards_edge = edge.translate(normal_vector * maxDim)
    away_edge = edge.translate(-normal_vector * maxDim)
    if dir == "both":
        towards_split_face = get_split_face_from_edges(edge, towards_edge)
        away_split_face = get_split_face_from_edges(edge, away_edge)
        split_face = towards_split_face.fuse(away_split_face)
    elif dir in ("towards", "away"):
        split_face = get_split_face_from_edges(
            edge, towards_edge if dir == "towards" else away_edge
        )

    if snap != False:
        snap_tolerance = snap if isinstance(snap, float) else None
        intersected_edges = workplane.intersect(cq.Workplane(split_face)).edges().vals()
        if len(intersected_edges) > 0:
            closest_intersection_edge = CQLinq.find_nearest(intersected_edges, edge)
            assert closest_intersection_edge, "No close intersecting edge found"
            snap_edge = cast(
                cq.Edge,
                CQLinq.find_nearest(
                    workplane,
                    closest_intersection_edge,
                    snap_tolerance,
                    excluded=[edge],
                ),
            )
            if snap_edge and snap_edge not in snap_edges:
                snap_edges.add(snap_edge)
                split_face = get_split_face_from_edges(
                    edge, snap_edge, is_line_end=False
                )
                return split_face

    return split_face
