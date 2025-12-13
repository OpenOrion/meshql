from pathlib import Path
from typing import Annotated, Iterable, Iterator, Literal, MutableSet, Sequence, TypeVar, Union
import numpy as np
import cadquery as cq
from cadquery.cq import CQObject
from cadquery.occ_impl.shape_protocols import Shapes as CQShapes
import meshly
from pydantic import BaseModel, BeforeValidator, PlainSerializer
from OCP.TopoDS import TopoDS_Compound, TopoDS_Solid, TopoDS_Shell, TopoDS_Face, TopoDS_Wire, TopoDS_Edge, TopoDS_Vertex

# CadQuery and geometry types
CQType = CQShapes
RegionGroupType = Literal["split", "interior", "exterior"]
CQEdgeOrFace = Union[cq.Edge, cq.Face]

# Type mappings - moved from cq.py to break circular imports
CQ_TYPE_STR_MAPPING: dict[type[CQObject], CQType] = {
    cq.Compound: "Compound",
    cq.Solid: "Solid",
    cq.Shell: "Shell",
    cq.Face: "Face",
    cq.Wire: "Wire",
    cq.Edge: "Edge",
    cq.Vertex: "Vertex",
}

OCC_TYPE_STR_MAPPING: dict[type[CQObject], CQType] = {
    TopoDS_Compound: "Compound",
    TopoDS_Solid: "Solid",
    TopoDS_Shell: "Shell",
    TopoDS_Face: "Face",
    TopoDS_Wire: "Wire",
    TopoDS_Edge: "Edge",
    TopoDS_Vertex: "Vertex",
}

CQ_TYPE_CLASS_MAPPING = dict(
    zip(CQ_TYPE_STR_MAPPING.values(), CQ_TYPE_STR_MAPPING.keys())
)

CQ_TYPE_RANKING = dict(
    zip(CQ_TYPE_STR_MAPPING.keys(), range(len(CQ_TYPE_STR_MAPPING) + 1)[::-1])
)
NumpyFloat = np.float32
Number = Union[int, float, NumpyFloat]
VectorSequence = Union[tuple[Number, Number, Number],
                       tuple[Number, Number], np.ndarray]
VectorLike = Union[VectorSequence, cq.Vector]
LineTuple = tuple[VectorSequence, VectorSequence]
Axis = Union[Literal["X", "Y", "Z"], VectorSequence, cq.Vector]
PathLike = Union[str, Path]
LoadTarget = Union[cq.Workplane, PathLike, Sequence[CQObject], meshly.Mesh]


def to_array(vec: VectorLike):
    if isinstance(vec, np.ndarray):
        return vec
    elif isinstance(vec, tuple):
        return np.array(vec if len(vec) == 3 else (*vec, 0))
    elif isinstance(vec, cq.Vector):
        return np.array(vec.toTuple())


def to_2d_array(vecs: Iterable[VectorLike]):
    return [to_array(vec) for vec in vecs]


def to_vec(axis: Axis):
    if isinstance(axis, str):
        return cq.Vector([1 if axis == "X" else 0, 1 if axis == "Y" else 0, 1 if axis == "Z" else 0])
    elif isinstance(axis, tuple):
        return cq.Vector(axis)
    elif isinstance(axis, np.ndarray):
        return cq.Vector(tuple(axis))
    return axis


T = TypeVar("T")


class OrderedSet(MutableSet[T]):
    """A set that preserves insertion order by internally using a dict."""

    def __init__(self, iterable: Iterable[T] = []):
        self._d = dict.fromkeys(iterable)

    def add(self, x: T) -> None:
        self._d[x] = None

    def discard(self, x: T) -> None:
        self._d.pop(x, None)

    def update(self, iterable: Iterable[T]) -> None:
        self._d.update(dict.fromkeys(iterable))

    def difference(self, d: "OrderedSet") -> "OrderedSet[T]":
        copy = OrderedSet(self)
        copy -= d
        return copy

    def union(self, d: "OrderedSet") -> "OrderedSet[T]":
        copy = OrderedSet(self)
        copy |= d
        return copy

    def intersection(self, d: "OrderedSet") -> "OrderedSet[T]":
        copy = OrderedSet(self)
        copy &= d
        return copy

    @property
    def first(self) -> T:
        return next(iter(self))

    @property
    def last(self) -> T:
        *_, last = iter(self)
        return last

    def __contains__(self, x: object) -> bool:
        return self._d.__contains__(x)

    def __len__(self) -> int:
        return self._d.__len__()

    def __iter__(self) -> Iterator[T]:
        return self._d.__iter__()

    def __str__(self):
        return f"{{{', '.join(str(i) for i in self)}}}"

    def __repr__(self):
        return f"<OrderedSet {self}>"


class PlaneData(BaseModel):
    """Pydantic model representing serializable plane information"""
    origin: tuple[float, float, float]
    normal: tuple[float, float, float]
    u_dir: tuple[float, float, float]  # U direction vector
    v_dir: tuple[float, float, float]  # V direction vector
    u_size: float  # Size in U direction
    v_size: float  # Size in V direction


def _face_to_plane(face: cq.Face) -> PlaneData:
    """Convert cq.Face to serializable PlaneData"""
    # Get bounding box to determine size
    bbox = face.BoundingBox()
    u_size = max(bbox.xlen, 1.0)  # Ensure minimum size
    v_size = max(bbox.ylen, 1.0)

    # Get center point as origin
    center = face.Center()
    origin = (center.x, center.y, center.z)

    # Get normal vector at center
    normal_vec = face.normalAt(center)
    normal = (normal_vec.x, normal_vec.y, normal_vec.z)

    # Calculate U and V direction vectors
    u_vec = cq.Vector(1, 0, 0)  # Default U direction
    v_vec = cq.Vector(0, 1, 0)  # Default V direction

    # If normal is not parallel to Z, adjust U and V
    if abs(normal_vec.z) < 0.99:
        # Use cross product to get perpendicular vectors
        if abs(normal_vec.x) < 0.99:
            u_vec = normal_vec.cross(cq.Vector(1, 0, 0)).normalized()
        else:
            u_vec = normal_vec.cross(cq.Vector(0, 1, 0)).normalized()
        v_vec = normal_vec.cross(u_vec).normalized()

    u_dir = (u_vec.x, u_vec.y, u_vec.z)
    v_dir = (v_vec.x, v_vec.y, v_vec.z)

    return PlaneData(
        origin=origin,
        normal=normal,
        u_dir=u_dir,
        v_dir=v_dir,
        u_size=u_size,
        v_size=v_size
    )


def _plane_to_face(plane_data: PlaneData) -> cq.Face:
    """Convert PlaneData back to cq.Face"""
    try:
        # Create vectors from stored tuples
        origin_vec = cq.Vector(*plane_data.origin)
        normal_vec = cq.Vector(*plane_data.normal)
        u_vec = cq.Vector(*plane_data.u_dir)
        v_vec = cq.Vector(*plane_data.v_dir)

        # Create corner points of the rectangular face
        half_u = plane_data.u_size / 2
        half_v = plane_data.v_size / 2

        corners = [
            origin_vec + u_vec * (-half_u) + v_vec * (-half_v),
            origin_vec + u_vec * (half_u) + v_vec * (-half_v),
            origin_vec + u_vec * (half_u) + v_vec * (half_v),
            origin_vec + u_vec * (-half_u) + v_vec * (half_v)
        ]

        # Convert to tuples for polygon creation
        corner_tuples = [(c.x, c.y, c.z) for c in corners]

        # Create face from polygon
        wire = cq.Wire.makePolygon(corner_tuples)
        return cq.Face.makeFromWires(wire)
    except Exception:
        # Fallback: create a simple square if conversion fails
        wire = cq.Wire.makePolygon(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)])
        return cq.Face.makeFromWires(wire)


def _validate_plane_input(value) -> cq.Face:
    """Validate and convert input to cq.Face"""
    if isinstance(value, cq.Face):
        return value
    elif isinstance(value, PlaneData):
        return _plane_to_face(value)
    elif isinstance(value, dict):
        plane_data = PlaneData(**value)
        return _plane_to_face(plane_data)
    else:
        raise ValueError(f"Cannot convert {type(value)} to cq.Face")


# Serializable cq.Face annotation using PlaneData
CQPlane = Annotated[
    cq.Face,
    BeforeValidator(_validate_plane_input),
    PlainSerializer(_face_to_plane, return_type=PlaneData),
]
