from dataclasses import dataclass, field
import os
import tempfile
import hashlib
import base64
from plotly import graph_objects as go
import cadquery as cq
from cadquery.cq import CQObject
from typing import Callable, Iterable, Literal, Optional, Sequence, TypeVar, Union, cast
import numpy as np
from meshql.utils.plot import add_plot
from meshql.utils.shapes import get_sampling
from meshql.utils.types import OrderedSet, NumpyFloat
from OCP.BRepTools import BRepTools
from OCP.BRep import BRep_Builder
from OCP.TopoDS import TopoDS_Shape
from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf

ShapeType = Literal["compound", "solid", "shell", "face", "wire", "edge", "vertex"]
CQGroupTypeString = Literal["split", "interior", "exterior"]
CQEdgeOrFace = Union[cq.Edge, cq.Face]
TShape = TypeVar('TShape', bound=cq.Shape)


TEMPDIR_PATH = tempfile.gettempdir()
CACHE_DIR_NAME = "meshql_geom_cache"
CACHE_DIR_PATH = os.path.join(TEMPDIR_PATH, CACHE_DIR_NAME)

@dataclass
class DirectedPath:
    edge: cq.Edge
    "edge or face of path"

    direction: Literal[1, -1] = 1
    "direction of path"

    def __post_init__(self):
        assert isinstance(self.edge, cq.Edge), "edge must be an edge"
        assert self.direction in [-1, 1], "direction must be -1 or 1"
        self.vertices = self.edge.Vertices()[::self.direction]
        self.start = self.vertices[0]
        self.end = self.vertices[-1]

    def __eq__(self, __value: object) -> bool:
        return self.edge == __value

    def __hash__(self) -> int:
        return self.edge.__hash__()

@dataclass
class Group:
    paths: list[DirectedPath] = field(default_factory=list)
    "elements in group"

    prev_group: Optional["Group"] = None
    "previous group"

    next_group: Optional["Group"] = None
    "next group"

    @property
    def start(self):
        return self.paths[0].start

    @property
    def end(self):
        return self.paths[-1].end

SHAPE_TYPE_STR_MAPPING: dict[type[CQObject], ShapeType] = {
    cq.Compound: "compound",
    cq.Solid: "solid",
    cq.Shell: "shell",
    cq.Face: "face",
    cq.Wire: "wire",
    cq.Edge: "edge",
    cq.Vertex: "vertex",
}
SHAPE_TYPE_CLASS_MAPPING = dict(zip(SHAPE_TYPE_STR_MAPPING.values(), SHAPE_TYPE_STR_MAPPING.keys()))

SHAPE_TYPE_RANKING = dict(zip(SHAPE_TYPE_STR_MAPPING.keys(), range(len(SHAPE_TYPE_STR_MAPPING)+1)[::-1]))

class CQLinq:    
    @staticmethod
    def select_tagged(workplane: cq.Workplane, tags: Union[str, Iterable[str]], shape_type: Optional[ShapeType] = None):
        for tag in ([tags] if isinstance(tags, str) else tags):
            if shape_type is None:
                yield from workplane._getTagged(tag).vals()
            else:
               yield from CQLinq.select(workplane._getTagged(tag).vals(), shape_type)    

    @staticmethod
    def select(target: Union[cq.Workplane, Iterable[CQObject], CQObject], shape_type: Optional[ShapeType] = None):
        cq_objs = target.vals() if isinstance(target, cq.Workplane) else ([target] if isinstance(target, CQObject) else list(target))
        
        if type is None:
            yield from cq_objs
        
        for cq_obj in cq_objs:
            assert isinstance(cq_obj, cq.Shape), "target must be a shape"
            if shape_type is None:
                yield cq_obj
            elif shape_type == "compound":
                yield from cq_obj.Compounds()
            elif shape_type == "solid":
                yield from cq_obj.Solids()
            elif shape_type == "shell":
                yield from cq_obj.Shells()
            elif shape_type == "face":
                yield from cq_obj.Faces()
            elif shape_type == "wire":
                yield from cq_obj.Wires()
            elif shape_type == "edge":
                yield from cq_obj.Edges()
            elif shape_type == "vertex":
                yield from cq_obj.Vertices()

    @staticmethod
    def select_batch(target: Union[cq.Workplane, Iterable[CQObject]], parent_type: ShapeType, child_type: ShapeType):
        cq_objs = list(target.vals() if isinstance(target, cq.Workplane) else target)
        if SHAPE_TYPE_STR_MAPPING[type(cq_objs[0])] == child_type:
            yield cq_objs
        else:
            parent_cq_objs = CQLinq.select(cq_objs, parent_type)
            for parent_occ_obj in parent_cq_objs:
                yield CQLinq.select([parent_occ_obj], child_type)

    # TODO: take a look later if this function is even needed
    @staticmethod
    def filter(objs: Iterable[CQObject], filter_objs: Iterable[CQObject], invert: bool):
        filtered = []
        filter_objs = OrderedSet(filter_objs)
        for cq_obj in objs:
            if (invert and cq_obj in filter_objs) or (not invert and cq_obj not in filter_objs):
                filtered.append(cq_obj)

        # sort filtered in the same order as filtered_objs
        return [cq_obj for cq_obj in filter_objs if cq_obj in filtered]
        
    @staticmethod
    def find_nearest(
        target: Union[cq.Workplane, CQObject, Sequence[CQObject]], 
        near_shape: Union[cq.Shape, cq.Vector], 
        tolerance: Optional[float] = None, 
        excluded: Optional[Sequence[CQObject]] = None,
        shape_type: Optional[ShapeType] = None
    ):
        min_dist_shape, min_dist = None, float("inf")
        for shape in CQLinq.select(target, shape_type or SHAPE_TYPE_STR_MAPPING[type(near_shape)]):
            if excluded and shape in excluded:
                continue
            near_center = near_shape.Center() if isinstance(near_shape, cq.Shape) else near_shape
            dist =  (shape.Center() - near_center).Length
            if dist != 0 and dist < min_dist:
                if (tolerance and dist <= tolerance) or not tolerance:
                    min_dist_shape, min_dist = shape, dist
        return cast(Optional[cq.Shape], min_dist_shape)



    @staticmethod
    def sortByConnect(target: Union[cq.Edge, Sequence[cq.Edge]]):
        unsorted_cq_edges = [target] if isinstance(target, cq.Edge) else target
        cq_edges = list(unsorted_cq_edges[1:])
        sorted_paths = [DirectedPath(unsorted_cq_edges[0])]
        while cq_edges:
            for i, cq_edge in enumerate(cq_edges):
                vertices = cq_edge.Vertices()
                if vertices[0].toTuple() == sorted_paths[-1].end.toTuple():
                    sorted_paths.append(DirectedPath(cq_edge))
                    cq_edges.pop(i)
                    break
                elif vertices[-1].toTuple() == sorted_paths[-1].end.toTuple():
                    sorted_paths.append(DirectedPath(cq_edge, direction=-1))
                    cq_edges.pop(i)
                    break
                elif vertices[0].toTuple() == sorted_paths[0].start.toTuple():
                    sorted_paths.insert(0, DirectedPath(cq_edge, direction=-1))
                    cq_edges.pop(i)
                    break
    
            else:
                raise ValueError("Edges do not form a closed loop")
        
        assert sorted_paths[-1].end == sorted_paths[0].start, "Edges do not form a closed loop"
        return sorted_paths
    



    @staticmethod
    def groupByTypes(
        target: Union[cq.Workplane, Sequence[CQObject]], 
        max_dim: float,
        tol: Optional[float] = None,
        prev_groups: Optional[dict[CQGroupTypeString, OrderedSet[CQObject]]] = None,
        only_faces=False, 
    ): 
        workplane = target if isinstance(target, cq.Workplane) else cq.Workplane().add(target)
        add_wire_to_group = lambda wires, group: group.update([
            *wires,
            *CQLinq.select(wires, "edge"),
            *CQLinq.select(wires, "vertex"),
        ])
        
        groups: dict[CQGroupTypeString, OrderedSet[CQObject]] = {
            "split": OrderedSet[CQObject](),
            "interior": OrderedSet[CQObject](),
            "exterior": OrderedSet[CQObject](),
        }
        is_2d = CQExtensions.get_dimension(workplane) == 2
        workplane = workplane.extrude(-1) if is_2d else workplane
        for solid in workplane.solids().vals():
            for face in solid.Faces():
                face_group = None
                if prev_groups:
                    for prev_groups_type, prev_group in prev_groups.items():
                        if face in prev_group:
                            face_group = groups[prev_groups_type]
                            break
                
                if face_group is None:
                    face_group_type = CQExtensions.get_face_group_type(workplane, face, max_dim, tol)
                    face_group = groups[face_group_type]

                face_group.add(face)
                if not only_faces:
                    add_wire_to_group(face.Wires(), face_group)

        if is_2d:
            exterior_group_tmp = groups["exterior"].difference(groups["interior"])
            groups["interior"] = groups["interior"].difference(groups["exterior"])
            groups["exterior"] = exterior_group_tmp

        return groups


    @staticmethod
    def groupBy(target: Union[cq.Workplane, Iterable[CQObject]], parent_type: ShapeType, child_type: ShapeType): 
        groups: dict[CQObject, OrderedSet[CQObject]] = {}
        
        cq_objs = target.vals() if isinstance(target, cq.Workplane) else (target if isinstance(target, Iterable) else [target])
        for cq_obj in cq_objs:
            parents = CQLinq.select(cq_obj, parent_type)
            for parent in parents:
                children = CQLinq.select(parent, child_type)
                for child in children:
                    if child not in groups:
                        groups[child] = OrderedSet[CQObject]()
                    groups[child].add(parent)
        return groups




    @staticmethod
    def find(target: Union[cq.Workplane, Iterable[CQObject]], where: Callable[[CQObject], bool]):
        cq_objs = target.vals() if isinstance(target, cq.Workplane) else (target if isinstance(target, Iterable) else [target])
        for cq_obj in cq_objs:
            if where(cq_obj):
                yield cq_obj

class CQExtensions:
    @staticmethod
    def is_interior_face(face: CQObject):
        assert isinstance(face, cq.Face), "object must be a face"
        face_normal = face.normalAt()
        face_centroid = face.Center()
        interior_dot_product = face_normal.dot(face_centroid)
        return interior_dot_product < 0


    @staticmethod
    def get_face_group_type(
        workplane: cq.Workplane, 
        face: cq.Face, 
        maxDim: float, 
        tol: Optional[float] = None
    ) -> CQGroupTypeString:
        total_solid = workplane.findSolid()
        is_planar = face.geomType() == "PLANE"
        if is_planar:
            face_center, face_normal = face.Center(), face.normalAt()
            normalized_face_normal = face_normal/face_normal.Length
            is_interior = is_split = CQExtensions.is_interior_face(face)
            if is_interior:
                intersect_line = cq.Edge.makeLine(face_center, face_center + (normalized_face_normal*0.1))
                intersect_vertices = total_solid.intersect(intersect_line, tol=tol).Vertices()
                is_split = len(intersect_vertices) > 0 and intersect_vertices[0].distance(face) < 1E-8
        else:
            try:
                nearest_center_gp_point = GeomAPI_ProjectPointOnSurf(face.Center().toPnt(), face._geomAdaptor()).NearestPoint()                
                face_center = cq.Vector(nearest_center_gp_point.X(), nearest_center_gp_point.Y(), nearest_center_gp_point.Z())
                face_normal = face.normalAt(face_center)
            except:
                face_center, face_normal = face.Center(), face.normalAt()
            normalized_face_normal = face_normal/face_normal.Length
            intersect_line = cq.Edge.makeLine(face_center, face_center + (normalized_face_normal*maxDim))
            intersect_vertices = total_solid.intersect(intersect_line, tol=tol).Vertices()
            is_interior = len(intersect_vertices) != 0
            is_split = is_interior and intersect_vertices[0].distance(face) < 1E-8

        if is_split:
            return "split"
        return "interior" if is_interior else "exterior"


    @staticmethod
    def get_angle_between(prev: CQEdgeOrFace, curr: CQEdgeOrFace):
        if isinstance(prev, cq.Edge) and isinstance(curr, cq.Edge):
            prev_tangent_vec = prev.tangentAt(0.5) # type: ignore
            tangent_vec = curr.tangentAt(0.5)      # type: ignore
        else:
            prev_tangent_vec = prev.normalAt() # type: ignore
            tangent_vec = curr.normalAt()      # type: ignore
        angle = prev_tangent_vec.getAngle(tangent_vec)
        assert not np.isnan(angle), "angle should not be NaN"
        return angle

    @staticmethod
    def fuse_shapes(shapes: Sequence[CQObject]) -> cq.Shape:
        fused_shape: Optional[cq.Shape] = None
        for shape in shapes:
            assert isinstance(shape, cq.Shape), "shape must be a shape"
            if fused_shape:
                fused_shape = fused_shape.fuse(shape)
            else:
                fused_shape = shape
        assert fused_shape is not None, "No shapes to fuse"
        return fused_shape

    @staticmethod
    def plot_cq(
        target: Union[cq.Workplane, CQObject, Sequence[CQObject], Sequence[Group], Sequence[Sequence[CQObject]]], 
        title: str = "Plot", 
        samples_per_spline: int = 50,
        ctx = None
    ):
        from meshql.entity import CQEntityContext
        ctx = cast(CQEntityContext, ctx)

        fig = go.Figure(
            layout=go.Layout(title=go.layout.Title(text=title))
        )
        if isinstance(target, cq.Workplane):
            edge_groups = [[edge] for edge in CQLinq.select(target, "edge")]
        elif isinstance(target, CQObject):
            edge_groups = [[edge] for edge in CQLinq.select(target, "edge")]
        elif isinstance(target, Sequence) and isinstance(target[0], CQObject):
            edge_groups = [cast(Sequence[CQObject], target)]
        elif isinstance(target, Sequence) and isinstance(target[0], Group):
            edge_groups = [[path.edge for path in cast(Group, group).paths]for group in target]
        else:
            target = cast(Sequence[Sequence], target)
            edge_groups = cast(Sequence[Sequence[CQObject]], target)

        for i, edges in enumerate(edge_groups):

            edge_name = f"Edge{ctx.select(edges[0]).tag}" if ctx else f"Edge{i}"
            sampling = get_sampling(0, 1, samples_per_spline, False)
            coords = np.concatenate([np.array([vec.toTuple() for vec in edge.positions(sampling)], dtype=NumpyFloat) for edge in edges]) # type: ignore
            add_plot(coords, fig, edge_name)

        fig.layout.yaxis.scaleanchor = "x"  # type: ignore
        fig.show()

    @staticmethod
    def get_dimension(workplane: cq.Workplane):
        return 2 if len(workplane.solids().vals()) == 0 else 3

    @staticmethod
    def import_workplane(target: Union[cq.Workplane, str, Iterable[CQObject]]):
        if isinstance(target, str):
            if target.lower().endswith(".step"):
                workplane = cq.importers.importStep(target)
            elif target.lower().endswith(".dxf"):
                workplane = cq.importers.importDXF(target)
            else:
                raise ValueError(f"Unsupported file type: {target}")
        elif isinstance(target, Iterable):
            workplane = cq.Workplane().newObject(target)
        else:
            workplane = target

        return workplane


    @staticmethod
    def scale(shape: TShape, x: float = 1, y: float = 1, z: float = 1) -> TShape:
        t = cq.Matrix([
            [x, 0, 0, 0],
            [0, y, 0, 0],
            [0, 0, z, 0],
            [0, 0, 0, 1]
        ])
        return shape.transformGeometry(t)

class CQCache:
    @staticmethod
    def import_brep(file_path: str):
        """
        Import a boundary representation model
        Returns a TopoDS_Shape object
        """
        builder = BRep_Builder()
        shape = TopoDS_Shape()
        return_code = BRepTools.Read_s(shape, file_path, builder)
        if return_code is False:
            raise ValueError("Import failed, check file name")
        return cq.Compound(shape)

    @staticmethod
    def get_cache_exists(obj: Union[Sequence[CQObject], CQObject]):
        cache_file_name = CQCache.get_file_name(obj)
        return os.path.isfile(cache_file_name)

    @staticmethod
    def get_file_name(shape: Union[Sequence[CQObject], CQObject]):
        prev_vector = cq.Vector(0,0,0)
        for vertex in cast(Sequence[cq.Vertex], CQLinq.select(shape, "vertex")):
            prev_vector += vertex.Center()
        
        hasher = hashlib.md5()
        hasher.update(bytes(str(tuple(np.round(prev_vector.toTuple(), 4))), "utf-8"))
        # encode the hash as a filesystem safe string
        shape_id = base64.urlsafe_b64encode(hasher.digest()).decode("utf-8")
        return f"{CACHE_DIR_PATH}/{shape_id}.brep"

    @staticmethod
    def export_brep(shape: cq.Shape, file_path: str):
        if CACHE_DIR_NAME not in os.listdir(TEMPDIR_PATH):
            os.mkdir(CACHE_DIR_PATH)
        shape.exportBrep(file_path)

    @staticmethod
    def clear_cache():
        if CACHE_DIR_NAME in os.listdir(TEMPDIR_PATH):
            for file in os.listdir(CACHE_DIR_PATH):
                os.remove(os.path.join(CACHE_DIR_PATH, file))

