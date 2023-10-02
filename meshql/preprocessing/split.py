import numpy as np
import cadquery as cq
from cadquery.cq import CQObject
from typing import Callable, Literal, Optional, Sequence, Union, cast
from meshql.utils.cq import CQCache, CQExtensions, CQGroupTypeString, CQLinq
from meshql.selector import SelectorQuerier
from meshql.utils.types import Axis, LineTuple, OrderedSet, VectorSequence, to_array, to_vec
from jupyter_cadquery import show

SplitAt = Literal["end", "per"]

class Split(SelectorQuerier):
    def __init__(self, workplane: cq.Workplane, split_at: SplitAt, use_cache: bool = False, use_raycast: bool = False) -> None:
        super().__init__(workplane, only_faces=True, use_raycast=use_raycast)
        self.initial_workplane = workplane
        self.split_at = split_at
        self.use_cache = use_cache
        self.split_face_groups = list[list[cq.Face]]()
        self.maxDim = workplane.findSolid().BoundingBox().DiagonalLength

    def apply_split(self, split_shapes: Union[cq.Shape, Sequence[cq.Shape]]):
        split_shapes = [split_shapes] if isinstance(split_shapes, cq.Shape) else list(split_shapes)
        self.split_face_groups += [split_shapes]
        if self.split_at == "per":
            self.workplane = split_workplane(self.workplane, split_shapes, self.use_cache)

    def finalize(self):
        if self.split_at == "end":
            split_faces = [face for split_face_group in self.split_face_groups for face in split_face_group]
            self.workplane = split_workplane(self.workplane, split_faces, self.use_cache)
        return self

    def show(self, theme: Literal["light", "dark"] = "light"):
        assert len(self.split_face_groups) > 0, "No split faces to show"
        show(self.workplane.newObject(self.split_face_groups[-1]), theme=theme)
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
    
    # def from_ratio(
    #     self,
    #     select: Callable[[cq.Workplane], cq.Workplane],
    #     wire_select: Callable[[cq.Workplane], cq.Workplane],
    #     type: Literal['interior', 'exterior'],
    #     snap_tolerance: Optional[float] = None,
    #     dir: Optional[Literal["away", "towards", "both"]] = None,
    #     axis: Union[Axis, Literal["avg", "face1", "face2"]] = "avg",
    #     angle_offset: VectorTuple = (0,0,0),
    #     split_dependant=False
    # ):
    #     pass




    def from_normals(
        self, 
        type: Literal['interior', 'exterior'],
        selector: Optional[cq.Selector] = None,
        snap_tolerance: Optional[float] = None,
        dir: Optional[Literal["away", "towards", "both"]] = None,
        axis: Union[Axis, Literal["avg", "face1", "face2"]] = "avg",
        angle_offset: VectorSequence = (0,0,0),
        split_dependant=False
    ):
        offset = to_vec(np.radians(list(angle_offset)))
        if not dir and axis is not str:
            dir = "towards"
        elif dir is None:
            dir = "away" if type == "interior" else "towards"

        if (self.split_at == "per" and len(self.split_face_groups) > 0) or split_dependant:
            selected_faces = CQLinq.groupByTypes(self.workplane, only_faces=True, use_raycast=split_dependant)[type]
        else:
            selected_faces = self.type_groups[type]
        face_edge_groups = cast(
            dict[cq.Edge, OrderedSet[cq.Face]], 
            CQLinq.groupBy(selected_faces, "face", "edge")
        )
        
        if selector:
            filtered_edges = selector.filter(face_edge_groups.keys())
        else:
            filtered_edges = face_edge_groups.keys()

        split_faces = list[cq.Shape]()
        snap_edges = OrderedSet[cq.Edge]()
        for edge in filtered_edges:
            faces = face_edge_groups[edge]
            edge_vec = (cq.Vector(edge.endPoint().toTuple()) - cq.Vector(edge.startPoint().toTuple()))/edge.Length()
            if 1-abs(edge_vec.z) < 0.1:
                if axis == "avg":
                    average_normal = np.average([face.normalAt().toTuple() for face in faces], axis=0)
                    normal_vec = cq.Vector(tuple(average_normal)) + offset
                elif axis == "face1":
                    normal_vec = list(faces)[0].normalAt()
                elif axis == "face2":
                    normal_vec = list(faces)[1].normalAt()
                else:
                    normal_vec = to_vec(axis, normalize=True)

                split_face = get_edge_split_face(self.workplane, edge, normal_vec, dir, snap_tolerance, snap_edges)
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
        axis: Union[Literal["X", "Y", "Z"], VectorSequence, cq.Vector] = "Z",
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
            edges_pnts = np.array([to_array(lines), to_array(lines)])
        elif isinstance(lines, list) and len(lines) == 1:
            edges_pnts = np.array([to_array(lines[0]), to_array(lines[0])])
        else: 
            edges_pnts = np.array([to_array(line) for line in lines])
        normal_vector = to_vec(axis, normalize=True)

        if dir in ("both", "towards"):
            edges_pnts[0] += self.maxDim * normal_vector
        if dir in ("both", "away"):        
            edges_pnts[-1] -= self.maxDim * normal_vector

        side1 = edges_pnts[:, 0].tolist()
        side2 = edges_pnts[:, 1].tolist()
        wire_pnts = [side1[0], *side2, *side1[1:][::-1]] 
        self.from_pnts(wire_pnts)
        return self


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
    if ((edge2.endPoint().Center() - edge1.endPoint().Center()).Length < (edge2.startPoint().Center() - edge1.endPoint().Center()).Length):
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
    maxDim = workplane.findSolid().BoundingBox().DiagonalLength
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
        maxDim = workplane.findSolid().BoundingBox().DiagonalLength
        normal_vector = to_vec(axis, normalize=True)
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
                closest_intersection_edge = CQExtensions.find_nearest(intersected_edges, edge)
                snap_edge = cast(cq.Edge, CQExtensions.find_nearest(workplane, closest_intersection_edge, snap_tolerance, excluded=[edge]))
                if snap_edge and snap_edge not in snap_edges:
                    snap_edges.add(snap_edge)
                    split_face = get_split_face_from_edges(edge, snap_edge, is_line_end=False)
                    return split_face

        return split_face

