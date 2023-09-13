import numpy as np
import cadquery as cq
from cadquery.cq import VectorLike
from typing import Literal, Optional, Sequence, Union
from meshql.utils.cq import CQCache, CQExtensions, CQLinq
from meshql.utils.types import LineTuple, VectorTuple
from jupyter_cadquery import show
Axis = Union[Literal["X", "Y", "Z"], VectorTuple, cq.Vector]
def get_normal_from_axis(axis: Axis):
    if isinstance(axis, str):
        return cq.Vector([1 if axis == "X" else 0, 1 if axis == "Y" else 0, 1 if axis == "Z" else 0])        
    elif isinstance(axis, tuple):
        return cq.Vector(axis)
    return axis

def norm_line_tuple(line: LineTuple):
    pnt1 = tuple(float(v) for v in ((*line[0], 0) if len(line[0]) == 2 else line[0]))
    pnt2 = tuple(float(v) for v in ((*line[1], 0) if len(line[1]) == 2 else line[1]))
    return (pnt1, pnt2)

class Split:
    @staticmethod
    def from_plane(
        base_pnt: VectorLike = (0,0,0), 
        angle: VectorTuple = (0,0,0), 
    ):
        return cq.Face.makePlane(None, None, base_pnt, tuple(np.radians(angle)))

    @staticmethod
    def from_faces(
        workplane: cq.Workplane, 
        face_type: Literal['interior', 'exterior'],
        snap_tolerance: Optional[float] = None,
        angle_offset: VectorTuple = (0,0,0),
    ):
        offset = cq.Vector(tuple(np.radians(angle_offset)))
        type_groups = CQLinq.groupByTypes(workplane, only_faces=True, check_splits=False)
        if False:
            yield None
        dir = "away" if face_type == "interior" else "towards"
        face_edge_groups: dict[cq.Edge, set[cq.Face]] = {}

        for face in type_groups[face_type]:
            assert isinstance(face, cq.Face)
            edges = CQLinq.select(face, "edge")
            for edge in edges:
                assert isinstance(edge, cq.Edge)
                if edge not in face_edge_groups:
                    face_edge_groups[edge] = set()
                face_edge_groups[edge].add(face)

        for edge, faces in face_edge_groups.items():
            average_normal = np.average([face.normalAt().toTuple() for face in faces], axis=0)
            average_normal_vec = cq.Vector(tuple(average_normal)) + offset
            edge_vec = cq.Vector(edge.endPoint().toTuple()) - cq.Vector(edge.startPoint().toTuple())
            is_parallel = edge_vec.dot(average_normal_vec) == 0
            if is_parallel:
                yield Split.from_edge(workplane, edge, average_normal_vec, dir, snap_tolerance)


    @staticmethod
    def from_anchor(
        workplane: cq.Workplane, 
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
            split_face = Split.from_plane(anchor, angle)
            if until == "next":
                intersect_vertex = CQExtensions.split_intersect(workplane, anchor, split_face, snap_tolerance)
                edges.append((anchor, intersect_vertex.toTuple())) # type: ignore
            else:
                edges.append(split_face)
        return Split.from_lines(workplane, edges)
    
    @staticmethod
    def from_pnts(pnts: Sequence[VectorLike]):
        return cq.Face.makeFromWires(cq.Wire.makePolygon(pnts))

    @staticmethod
    def from_edge(
        workplane: cq.Workplane,
        edge: cq.Edge, 
        axis: Union[Literal["X", "Y", "Z"], VectorTuple, cq.Vector] = "Z",
        dir: Literal["away", "towards", "both"] = "both",
        snap_tolerance: Optional[float] = None,
    ):
        scaled_edge = CQExtensions.scale(edge, z=10)
        maxDim = workplane.findSolid().BoundingBox().DiagonalLength * 10.0
        normal_vector = get_normal_from_axis(axis)        

        max_dim_edge = scaled_edge.translate(normal_vector * maxDim) if dir in ("both", "towards") else scaled_edge
        min_dim_edge = scaled_edge.translate(-normal_vector * maxDim) if dir in ("both", "away") else scaled_edge

        
        split_face = cq.Face.makeFromWires(cq.Wire.assembleEdges([
            min_dim_edge,
            cq.Edge.makeLine(min_dim_edge.endPoint(), max_dim_edge.endPoint()),
            max_dim_edge,
            cq.Edge.makeLine(max_dim_edge.startPoint(), min_dim_edge.startPoint()),
        ]))

        # TODO: for some reason this snap tolerance is not working
        if snap_tolerance:
            intersect_vertex = CQExtensions.split_intersect(workplane, edge.startPoint(), split_face, snap_tolerance)
            assert intersect_vertex, "No intersection found"
            intersect_vec = cq.Vector(intersect_vertex.toTuple()) - cq.Vector(edge.startPoint().toTuple())
            intersect_vec_norm = intersect_vec/intersect_vec.Length
            return Split.from_edge(workplane, edge, intersect_vec_norm, "towards")
        return split_face

    @staticmethod
    def from_lines(
        workplane: cq.Workplane, 
        lines: Union[list[LineTuple], LineTuple], 
        axis: Union[Literal["X", "Y", "Z"], VectorTuple] = "Z",
        dir: Literal["away", "towards", "both"] = "both",
    ):
        if isinstance(lines, tuple):
            edges_pnts = np.array([norm_line_tuple(lines), norm_line_tuple(lines)])
        elif isinstance(lines, list) and len(lines) == 1:
            edges_pnts = np.array([norm_line_tuple(lines[0]), norm_line_tuple(lines[0])])

        else: 
            edges_pnts = [norm_line_tuple(line) for line in lines]
        maxDim = workplane.findSolid().BoundingBox().DiagonalLength * 10.0
        normal_vector = np.array(get_normal_from_axis(axis).toTuple())

        if dir in ("both", "towards"):
            edges_pnts[0] += maxDim * normal_vector
        if dir in ("both", "away"):        
            edges_pnts[-1] -= maxDim * normal_vector

        side1 = edges_pnts[:, 0].tolist()
        side2 = edges_pnts[:, 1].tolist()
        wire_pnts = [side1[0], *side2, *side1[1:][::-1]] 
        return Split.from_pnts(wire_pnts)


def split_workplane(workplane: cq.Workplane, splits: Sequence[cq.Face], use_cache: bool = True):
    shape_combo = [*workplane.vals(), *splits]
    cache_exists = CQCache.get_cache_exists(shape_combo) if use_cache else False
    cache_file_name = CQCache.get_file_name(shape_combo) if use_cache else ""
    if use_cache and cache_exists:
        shape = CQCache.import_brep(cache_file_name)
    else:
        for split in splits:      
            workplane = workplane.split(split)
        shape = CQExtensions.fuse_shapes(workplane.vals())
        if use_cache:
            CQCache.export_brep(shape, cache_file_name)
    return cq.Workplane(shape)
