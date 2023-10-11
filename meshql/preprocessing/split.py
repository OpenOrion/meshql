import numpy as np
import cadquery as cq
from cadquery.cq import CQObject
from typing import Callable, Iterable, Literal, Optional, Sequence, Union, cast
from meshql.utils.cq import CQCache, CQExtensions, CQLinq, Selection
from meshql.utils.shapes import get_sampling
from meshql.utils.types import Axis, LineTuple, OrderedSet, VectorSequence, to_2d_array, to_array, to_vec
from jupyter_cadquery import show

SplitAt = Literal["end", "per"]
MultiFaceAxis = Union[Axis, Literal["avg", "face1", "face2"]]
def normalize(vec: cq.Vector):
    return vec / vec.Length

class Split:
    def __init__(self, workplane: cq.Workplane, split_at: SplitAt, use_cache: bool = False) -> None:
        self.workplane = self.initial_workplane = workplane
        self.split_at = split_at
        self.use_cache = use_cache
        self.split_groups = list[list[cq.Face]]()
        self.maxDim = workplane.findSolid().BoundingBox().DiagonalLength * 10

    def refresh(self, use_raycast=False):
        self.type_groups = CQLinq.groupByTypes(self.workplane, use_raycast=use_raycast)
        self.face_edge_groups = CQLinq.groupBy(self.workplane, "face", "edge")

    def apply_split(self, split_shapes: Union[cq.Shape, Sequence[cq.Shape]]):
        split_shapes = [split_shapes] if isinstance(split_shapes, cq.Shape) else list(split_shapes)
        self.split_groups += [split_shapes]
        if self.split_at == "per":
            self.workplane = split_workplane(self.workplane, split_shapes, self.use_cache)

    def finalize(self):
        if self.split_at == "end":
            split_faces = [face for split_face_group in self.split_groups for face in split_face_group]
            self.workplane = split_workplane(self.workplane, split_faces, self.use_cache)
        return self

    def show(self, theme: Literal["light", "dark"] = "light"):
        assert len(self.split_groups) > 0, "No split faces to show"
        show(self.workplane.newObject(self.split_groups[-1]), theme=theme)
        return self

    def from_plane(
        self,
        base_pnt: VectorSequence = (0,0,0), 
        angle: VectorSequence = (0,0,1),
        sizing: Literal["maxDim", "infinite"] = "maxDim"
    ):
        split_face = get_plane_split_face(self.workplane, base_pnt, angle, sizing)
        self.apply_split(split_face)
        return self
    
    def from_ratios(
        self,
        start: Selection,
        end: Selection,
        ratios: list[float],
        dir: Literal["away", "towards", "both"],
        axis: Optional[Axis] = None,
        snap_tolerance: Optional[float] = None,
        angle_offset: VectorSequence = (0,0,0),
    ):
        self.refresh(use_raycast=True)
        offset = to_vec(np.radians(list(angle_offset)))
        start_wire = cq.Wire.assembleEdges(cast(list[cq.Edge], start.select(self, "edge", is_intersection=True)))
        end_wire = cq.Wire.assembleEdges(cast(list[cq.Edge], end.select(self, "edge", is_intersection=True)))

        for ratio in ratios:
            start_point = start_wire.positionAt(ratio)
            end_point = end_wire.positionAt(0.5-ratio if ratio <=0.5 else 1.5-ratio)
            edge = cq.Edge.makeLine(start_point, end_point)

            # if axis:
            #     dir = dir or "towards"
            #     normal_vec = to_vec(axis) + offset
            # else:
            if start.type and start.type == end.type:
                target = self.type_groups[start.type]
            else:
                target = self.workplane

            nearest_face = cast(cq.Face, CQLinq.find_nearest(target, edge, select_type="face"))
            normal_vec = normalize(nearest_face.normalAt((start_point + end_point)/2)) + offset
            projected_edge = edge.project(nearest_face, normal_vec).Edges()[0]
            assert isinstance(projected_edge, cq.Edge), "Projected edge is single edge"
            self.from_edge(projected_edge, normal_vec, dir, snap_tolerance)

        return self


    def from_normals(
        self, 
        selection: Selection,
        dir: Optional[Literal["away", "towards", "both"]] = None,
        axis: Union[MultiFaceAxis, list[MultiFaceAxis]] = "avg",
        snap_tolerance: Optional[float] = None,
        angle_offset: VectorSequence = (0,0,0),
        is_initial: bool = False,
    ):
        self.refresh(use_raycast=True)
        offset = to_vec(np.radians(list(angle_offset)))
        dir = dir or "away" if selection.type == "interior" else "towards"        
        filtered_edges = cast(Sequence[cq.Edge], selection.select(self, "edge", is_initial=is_initial, is_exclusive=True))

        split_faces = list[cq.Shape]()
        snap_edges = OrderedSet[cq.Edge]()
        for edge in filtered_edges:
            faces = self.face_edge_groups[edge]
            split_face = None
            for _axis in (axis if isinstance(axis, list) else [axis]):
                normal_vec = self._get_normal_vec(faces, cast(Axis, _axis), offset)
                curr_split_face = get_edge_split_face(self.workplane, edge, normal_vec, dir, snap_tolerance, snap_edges)
                if split_face is None:
                    split_face = curr_split_face
                else:
                    split_face = split_face.fuse(curr_split_face)
            assert split_face, "No split face found"
            split_faces.append(split_face)

        self.apply_split(split_faces)
        return self

    def from_anchor(
        self, 
        anchor: Union[list[VectorSequence], VectorSequence] = (0,0,0), 
        angle: Union[list[VectorSequence], VectorSequence] = (0,0,0),
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
                intersect_vertex = CQExtensions.split_intersect(self.workplane, anchor, split_face, snap_tolerance)
                assert intersect_vertex, "No intersecting vertex found"
                edges.append((anchor, intersect_vertex.toTuple()))
            else:
                edges.append(split_face)
        self.from_lines(edges)
        return self
    
    def from_pnts(self, pnts: Sequence[VectorSequence]):
        pnt_vecs = [to_vec(pnt) for pnt in pnts]
        split_face = cq.Face.makeFromWires(cq.Wire.makePolygon(pnt_vecs))
        self.apply_split(split_face)
        return self

    def from_edge(
        self,
        edge: cq.Edge, 
        axis: Axis = "Z",
        dir: Literal["away", "towards", "both"] = "both",
        snap_tolerance: Optional[float] = None,
    ):
        split_face = get_edge_split_face(self.workplane, edge, axis, dir, snap_tolerance)
        self.apply_split(split_face)
        return self

    def from_lines(
        self, 
        lines: Union[list[LineTuple], LineTuple], 
        axis: Axis = "Z",
        dir: Literal["away", "towards", "both"] = "both",
    ):
        if isinstance(lines, tuple):
            edges_pnts = np.array([to_2d_array(lines), to_2d_array(lines)])
        elif isinstance(lines, list) and len(lines) == 1:
            edges_pnts = np.array([to_2d_array(lines[0]), to_2d_array(lines[0])])
        else: 
            edges_pnts = np.array([to_2d_array(line) for line in lines])
        normal_vector = to_array(normalize(to_vec(axis)))

        if dir in ("both", "towards"):
            edges_pnts[0] += self.maxDim * normal_vector
        if dir in ("both", "away"):        
            edges_pnts[-1] -= self.maxDim * normal_vector

        side1 = edges_pnts[:, 0].tolist()
        side2 = edges_pnts[:, 1].tolist()
        wire_pnts = [side1[0], *side2, *side1[1:][::-1]] 
        self.from_pnts(wire_pnts)
        return self

    def _get_normal_vec(
        self, 
        faces: OrderedSet[cq.Face],
        axis: Optional[Union[Axis, Literal["avg", "face1", "face2"]]],
        offset: cq.Vector = cq.Vector(0,0,0)
    ):
        if axis is None:
            axis = "face1" if len(faces) == 1 else "avg"

        if axis == "avg":
            average_normal = np.average([face.normalAt().toTuple() for face in faces], axis=0)
            norm_vec = cq.Vector(tuple(average_normal)) + offset
        elif axis == "face1":
            norm_vec = list(faces)[0].normalAt()
        elif axis == "face2":
            norm_vec = list(faces)[1].normalAt()
        else:
            norm_vec = to_vec(axis)
        return normalize(norm_vec + offset)

def split_workplane(workplane: cq.Workplane, split_faces: Sequence[cq.Shape], use_cache: bool = True):
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
    if (
        (edge2.endPoint().Center() - edge1.endPoint().Center()).Length < 
        (edge2.startPoint().Center() - edge1.endPoint().Center()).Length
    ):
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
    base_pnt: VectorSequence = (0,0,0), 
    angle: VectorSequence = (0,0,1),
    sizing: Literal["maxDim", "infinite"] = "maxDim"    
):
    maxDim = workplane.findSolid().BoundingBox().DiagonalLength * 10
    base_pnt_vec = to_vec(base_pnt)
    angle_vec = to_vec(np.radians(list(angle)))
    if sizing == "maxDim":
        return cq.Face.makePlane(maxDim, maxDim, base_pnt_vec, angle_vec)
    else:
        return cq.Face.makePlane(None, None, base_pnt_vec, angle_vec)

def get_edge_split_face(
        workplane: cq.Workplane,
        edge: cq.Edge, 
        axis: Axis = "Z",
        dir: Literal["away", "towards", "both"] = "both",
        snap_tolerance: Optional[float] = None,
        snap_edges = OrderedSet[cq.Edge]()
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
            split_face = get_split_face_from_edges(edge, towards_edge if dir == "towards" else away_edge)

        if snap_tolerance:
            intersected_edges = workplane.intersect(cq.Workplane(split_face)).edges().vals()
            if len(intersected_edges) > 0:
                closest_intersection_edge = CQLinq.find_nearest(intersected_edges, edge)
                snap_edge = cast(cq.Edge, CQLinq.find_nearest(workplane, closest_intersection_edge, snap_tolerance, excluded=[edge]))
                if snap_edge and snap_edge not in snap_edges:
                    snap_edges.add(snap_edge)
                    split_face = get_split_face_from_edges(edge, snap_edge, is_line_end=False)
                    return split_face

        return split_face

