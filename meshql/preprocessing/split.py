import numpy as np
import cadquery as cq
from cadquery.cq import VectorLike
from typing import Callable, Iterable, Literal, Optional, Sequence, Union, cast
from meshql.utils.cq import CQCache, CQExtensions, CQLinq
from meshql.utils.types import LineTuple, OrderedSet, VectorTuple
from jupyter_cadquery import show

Axis = Union[Literal["X", "Y", "Z"], VectorTuple, cq.Vector]
def get_normal_from_axis(axis: Axis):
    if isinstance(axis, str):
        normal = cq.Vector([1 if axis == "X" else 0, 1 if axis == "Y" else 0, 1 if axis == "Z" else 0])        
    elif isinstance(axis, tuple):
        normal = cq.Vector(axis)
    else:
        normal = axis
    return normal / normal.Length

def norm_line_tuple(line: LineTuple):
    pnt1 = tuple(float(v) for v in ((*line[0], 0) if len(line[0]) == 2 else line[0]))
    pnt2 = tuple(float(v) for v in ((*line[1], 0) if len(line[1]) == 2 else line[1]))
    return (pnt1, pnt2)

class Split:
    def __init__(self, workplane: cq.Workplane, use_cache: bool = False) -> None:
        self.curr_workplane = workplane
        self.use_cache = use_cache
        self.split_faces: OrderedSet[cq.Face] = OrderedSet()

    def apply_split(self, split_faces: Union[cq.Face, Sequence[cq.Face]]):
        split_faces = [split_faces] if isinstance(split_faces, cq.Face) else split_faces
        self.split_faces.update(split_faces)
        self.curr_workplane = split_workplane(self.curr_workplane, split_faces, self.use_cache)

    def show(self):
        show(self.curr_workplane.newObject(self.split_faces))
        return self

    def from_plane(
        self,
        base_pnt: VectorLike = (0,0,0), 
        angle: VectorTuple = (0,0,1),
        is_max_dim_sizing: bool = False
    ):
        if is_max_dim_sizing:
            maxDim = self.curr_workplane.findSolid().BoundingBox().DiagonalLength * 10.0
            split_face = cq.Face.makePlane(maxDim, maxDim, base_pnt, tuple(np.radians(angle)))
        else:
            split_face = cq.Face.makePlane(None, None, base_pnt, tuple(np.radians(angle)))
        self.apply_split(split_face)
        return self
    
    def from_faces(
        self, 
        face_type: Literal['interior', 'exterior'],
        face_filter: Optional[Callable[[cq.Face], bool]] = None,
        snap_tolerance: Optional[float] = None,
        dir: Optional[Literal["away", "towards", "both"]] = None,
        axis: Optional[Union[Literal["X", "Y", "Z"], VectorTuple, cq.Vector]] = None,
        angle_offset: VectorTuple = (0,0,0),
        use_raycast=False
    ):
        offset = cq.Vector(tuple(np.radians(angle_offset)))
        
        type_groups = CQLinq.groupByTypes(self.curr_workplane, only_faces=True, use_raycast=use_raycast)
        dir = dir or ("away" if face_type == "interior" else "towards")
        face_edge_groups: dict[cq.Edge, set[cq.Face]] = {}
        
        faces = filter(face_filter, cast(Iterable[cq.Face], type_groups[face_type])) if face_filter else type_groups[face_type]
        for face in list(type_groups[face_type]):
            assert isinstance(face, cq.Face)
            edges = CQLinq.select(face, "edge")
            for edge in edges:
                assert isinstance(edge, cq.Edge)
                if edge not in face_edge_groups:
                    face_edge_groups[edge] = set()
                face_edge_groups[edge].add(face)
        split_faces = []

        snap_edges = OrderedSet[cq.Edge]()
        for edge, faces in face_edge_groups.items():
            if axis is None:
                average_normal = np.average([face.normalAt().toTuple() for face in faces], axis=0)
                normal_vec = cq.Vector(tuple(average_normal)) + offset
            else:
                normal_vec = get_normal_from_axis(axis)
            edge_vec = (cq.Vector(edge.endPoint().toTuple()) - cq.Vector(edge.startPoint().toTuple()))/edge.Length()
            # is_parallel = edge_vec.dot(average_normal_vec) == 0

            if 1-abs(edge_vec.z) < 0.01:
            # if is_parallel:
                split_face = get_split_face_from_edge(self.curr_workplane, edge, normal_vec, dir, snap_tolerance, snap_edges)
                split_faces.append(split_face)
        # show(self.curr_workplane.newObject(split_faces))
        self.apply_split(split_faces)
        return self

    def from_anchor(
        self, 
        anchor: Union[list[VectorTuple], VectorTuple] = (0,0,0), 
        angle: Union[list[VectorTuple], VectorTuple] = (0,0,0),
        snap_tolerance: Optional[float] = None,
        until: Literal["next", "all"] = "next",
    ):
        anchors = [anchor] if isinstance(anchor, tuple) else anchor
        angles = [angle] if isinstance(angle, tuple) else angle
        assert len(anchors) == len(angles), "anchors and dirs must be the same length"

        edges = []
        for anchor, angle in zip(anchors, angles):
            split_face = self.from_plane(anchor, angle)
            if until == "next":
                intersect_vertex = CQExtensions.split_intersect(self.curr_workplane, anchor, split_face, snap_tolerance)
                edges.append((anchor, intersect_vertex.toTuple())) # type: ignore
            else:
                edges.append(split_face)
        self.from_lines(edges)
        return self
    
    def from_pnts(self, pnts: Sequence[VectorLike]):
        split_face = cq.Face.makeFromWires(cq.Wire.makePolygon(pnts))
        self.apply_split(split_face)
        return self

    def from_edge(
        self,
        edge: cq.Edge, 
        axis: Union[Literal["X", "Y", "Z"], VectorTuple, cq.Vector] = "Z",
        dir: Literal["away", "towards", "both"] = "both",
        snap_tolerance: Optional[float] = None,
    ):
        split_face = get_split_face_from_edge(self.curr_workplane, edge, axis, dir, snap_tolerance)
        self.apply_split(split_face)
        return self

    def from_lines(
        self, 
        lines: Union[list[LineTuple], LineTuple], 
        axis: Axis = "Z",
        dir: Literal["away", "towards", "both"] = "both",
    ):
        if isinstance(lines, tuple):
            edges_pnts = np.array([norm_line_tuple(lines), norm_line_tuple(lines)])
        elif isinstance(lines, list) and len(lines) == 1:
            edges_pnts = np.array([norm_line_tuple(lines[0]), norm_line_tuple(lines[0])])

        else: 
            edges_pnts = [norm_line_tuple(line) for line in lines]
        maxDim = self.curr_workplane.findSolid().BoundingBox().DiagonalLength * 10.0
        normal_vector = np.array(get_normal_from_axis(axis).toTuple())

        if dir in ("both", "towards"):
            edges_pnts[0] += maxDim * normal_vector
        if dir in ("both", "away"):        
            edges_pnts[-1] -= maxDim * normal_vector

        side1 = edges_pnts[:, 0].tolist()
        side2 = edges_pnts[:, 1].tolist()
        wire_pnts = [side1[0], *side2, *side1[1:][::-1]] 
        self.from_pnts(wire_pnts)
        return self

def split_workplane(workplane: cq.Workplane, split_faces: Sequence[cq.Face], use_cache: bool = True):
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
    split_edges = [
        edge1,
        cq.Edge.makeLine(edge1.endPoint(), edge2.endPoint()),
        *[cq.Edge.makeLine(edge2.startPoint(), edge2.endPoint()) if is_line_end else edge2],
        cq.Edge.makeLine(edge2.startPoint(), edge1.startPoint()),
    ]

    try:
        return cq.Face.makeFromWires(cq.Wire.assembleEdges(split_edges))
    except:
        return cq.Face.makeNSidedSurface(split_edges, [])


def get_split_face_from_edge(
        workplane: cq.Workplane,
        edge: cq.Edge, 
        axis: Union[Literal["X", "Y", "Z"], VectorTuple, cq.Vector] = "Z",
        dir: Literal["away", "towards", "both"] = "both",
        snap_tolerance: Optional[float] = None,
        snap_edges = OrderedSet[cq.Edge]()
    ):
        maxDim = workplane.findSolid().BoundingBox().DiagonalLength
        normal_vector = get_normal_from_axis(axis)        
        towards_edge = edge.translate(normal_vector * maxDim)
        away_edge = edge.translate(-normal_vector * maxDim)

        if dir == "both":
            split_face = get_split_face_from_edges(towards_edge, away_edge)
        elif dir in ("towards", "away"):
            split_face = get_split_face_from_edges(edge, towards_edge if dir == "towards" else away_edge)

        if snap_tolerance:
            intersected_edges = workplane.intersect(cq.Workplane(split_face)).edges().vals()
            if len(intersected_edges) > 0:
                closest_intersection_edge = CQExtensions.find_nearest(intersected_edges, edge)
                snap_edge = cast(cq.Edge, CQExtensions.find_nearest(workplane, closest_intersection_edge, snap_tolerance, excluded=[edge]))
                if snap_edge and snap_edge not in snap_edges:
                    snap_edges.add(snap_edge)
                    split_face = get_split_face_from_edges(edge, snap_edge, is_line_end=False)
                    return split_face

        return split_face

