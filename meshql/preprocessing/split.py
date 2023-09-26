import numpy as np
import cadquery as cq
from cadquery.cq import VectorLike
from typing import Literal, Optional, Sequence, Union
from meshql.utils.cq import CQCache, CQExtensions, CQLinq
from meshql.utils.types import LineTuple, OrderedSet, VectorTuple
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


def split_workplane(workplane: cq.Workplane, split_face: Union[cq.Face, Sequence[cq.Face]], use_cache: bool = True):
    splits = ([split_face] if isinstance(split_face, cq.Face) else split_face)
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


def split_wire_from_edge(
        workplane: cq.Workplane,
        edge: cq.Edge, 
        axis: Union[Literal["X", "Y", "Z"], VectorTuple, cq.Vector] = "Z",
        dir: Literal["away", "towards", "both"] = "both",
        snap_tolerance: Optional[float] = None,
    ):
        
        scaled_edge = edge
        maxDim = workplane.findSolid().BoundingBox().DiagonalLength * 10.0
        normal_vector = get_normal_from_axis(axis)        

        max_dim_edge = scaled_edge.translate(normal_vector * maxDim) if dir in ("both", "towards") else scaled_edge
        min_dim_edge = scaled_edge.translate(-normal_vector * maxDim) if dir in ("both", "away") else scaled_edge

        min_max_line_points = (min_dim_edge.endPoint(), max_dim_edge.endPoint())
        max_min_line_points = (max_dim_edge.startPoint(), min_dim_edge.startPoint())

        if cq.Vector(min_max_line_points[-1]) != max_dim_edge.endPoint():
            min_max_line_points = min_max_line_points[::-1]
        if cq.Vector(max_min_line_points[-1]) != min_dim_edge.startPoint():
            max_min_line_points = max_min_line_points[::-1]

        split_wire = cq.Face.makeNSidedSurface([
            min_dim_edge,
            cq.Edge.makeLine(*min_max_line_points),
            max_dim_edge,
            cq.Edge.makeLine(*max_min_line_points),
        ], [])

        # TODO: for some reason this snap tolerance is not working
        # if snap_tolerance:
        #     intersect_vertex = CQExtensions.split_intersect(workplane, edge.startPoint(), split_wire, snap_tolerance)
        #     assert intersect_vertex, "No intersection found"
        #     intersect_vec = cq.Vector(intersect_vertex.toTuple()) - cq.Vector(edge.startPoint().toTuple())
        #     intersect_vec_norm = intersect_vec/intersect_vec.Length
        #     return split_wire_from_edge(workplane, edge, intersect_vec_norm, "towards")

        return split_wire


class Split:
    def __init__(self, workplane: cq.Workplane, use_cache: bool = False) -> None:
        self.curr_workplane = workplane
        self.use_cache = use_cache
        self.split_faces = OrderedSet[cq.Face]()

    def split(self, split_face: Union[cq.Face, Sequence[cq.Face]]):
        self.split_faces.update([face for face in ([split_face] if isinstance(split_face, cq.Face) else split_face)])
        self.curr_workplane = split_workplane(self.curr_workplane, split_face, self.use_cache)

    def show(self, split_face: cq.Face):
        show(self.curr_workplane.newObject([split_face]))
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
        self.split(split_face)
        return self
    
    def from_faces(
        self, 
        face_type: Literal['interior', 'exterior'],
        snap_tolerance: Optional[float] = None,
        angle_offset: VectorTuple = (0,0,0),
    ):
        offset = cq.Vector(tuple(np.radians(angle_offset)))
        type_groups = CQLinq.groupByTypes(self.curr_workplane, only_faces=True)
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
        split_faces = []
        for edge, faces in face_edge_groups.items():
            average_normal = np.average([face.normalAt().toTuple() for face in faces], axis=0)
            average_normal_vec = cq.Vector(tuple(average_normal)) + offset
            edge_vec = (cq.Vector(edge.endPoint().toTuple()) - cq.Vector(edge.startPoint().toTuple()))/edge.Length()
            # dot_prod = edge_vec.dot(average_normal_vec)
            # is_parallel = np.round(dot_prod, 5) == 0
            # print(is_parallel)
            # print(edge_vec)
            if 1-abs(edge_vec.z) < 0.01:
                split_wire = split_wire_from_edge(self.curr_workplane, edge, average_normal_vec, dir, snap_tolerance)
                # show(self.curr_workplane.newObject([split_wire]))
                split_faces.append(split_wire)
        # show(self.curr_workplane.newObject(split_wires))
        self.split(split_faces)
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
        self.split(split_face)
        return self

    def from_edge(
        self,
        edge: cq.Edge, 
        axis: Union[Literal["X", "Y", "Z"], VectorTuple, cq.Vector] = "Z",
        dir: Literal["away", "towards", "both"] = "both",
        snap_tolerance: Optional[float] = None,
    ):
        split_face = split_wire_from_edge(self.curr_workplane, edge, axis, dir, snap_tolerance)
        self.split(split_face)
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
