from dataclasses import dataclass
import cadquery as cq
from cadquery.cq import CQObject
from typing import Callable, Optional, Sequence, Union
from meshql.utils.cq import (
    GroupType,
    CQType,
)
from meshql.utils.cq_linq import SetOperation
from meshql.utils.types import OrderedSet
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SHAPE


class IndexSelector(cq.Selector):
    def __init__(self, indices: Sequence[int]):
        self.indices = indices

    def filter(self, objectList):
        return [objectList[i] for i in self.indices]


FilterExpression = Optional[Callable[[CQObject], bool]]


class FilterSelector(cq.Selector):
    def __init__(self, objFilter: FilterExpression):
        self.objFilter = objFilter

    def filter(self, objectList):
        return list(filter(self.objFilter, objectList))


class GroupSelector(cq.Selector):
    def __init__(self, allow: OrderedSet[CQObject]):
        self.allow = allow

    def filter(self, objectList):
        filtered_objs = []
        for obj in objectList:
            if obj in self.allow:
                filtered_objs.append(obj)
        return filtered_objs


@dataclass
class Selection:
    selector: Union[cq.Selector, str, None] = None
    tag: Union[str, Sequence[str], None] = None
    type: Optional[GroupType] = None
    indices: Optional[Sequence[int]] = None
    filter: Optional[Callable[[CQObject], bool]] = None
    region_set_operation: Optional[SetOperation] = None


from OCP.TopAbs import (
    TopAbs_COMPOUND,
    TopAbs_SOLID,
    TopAbs_SHELL,
    TopAbs_FACE,
    TopAbs_WIRE,
    TopAbs_EDGE,
    TopAbs_VERTEX,
    TopAbs_SHAPE,
)

ENUM_MAPPING = {
    "Compound": TopAbs_COMPOUND,
    "Solid": TopAbs_SOLID,
    "Shell": TopAbs_SHELL,
    "Face": TopAbs_FACE,
    "Wire": TopAbs_WIRE,
    "Edge": TopAbs_EDGE,
    "Vertex": TopAbs_VERTEX,
}


class ShapeExplorer:
    def __init__(self, shape: cq.Shape) -> None:
        self.shape = shape.wrapped
        self.explorer = TopExp_Explorer()

    def search(
        self,
        shape_type: CQType,
        not_from: Optional[CQType] = None,
    ):
        """
        Searchs all the shapes of type `shape_type` in the shape. If `not_from` is specified, will avoid all the shapes
        that are attached to the type `not_from`
        """
        to_avoid = TopAbs_SHAPE if not_from is None else ENUM_MAPPING[not_from]
        self.explorer.Init(self.shape, ENUM_MAPPING[shape_type], to_avoid)

        collection = []
        while self.explorer.More():
            shape = cq.Shape.cast(self.explorer.Current())
            collection.append(shape)
            self.explorer.Next()

        return list(set(collection))  # the 'set' is used to remove duplicates


class ConnectedShapesExplorer:
    def __init__(self, base_shape, child_shape) -> None:
        self.base_shape = base_shape
        self.child_shape = child_shape
        self.explorer = ShapeExplorer(base_shape)

    def _connected_by_vertices(self, shape, by_all=False):
        child_vertices = self.child_shape.Vertices()
        shape_vertices = shape.Vertices()

        if by_all:
            return all(v in child_vertices for v in shape_vertices)
        else:
            return any(v in child_vertices for v in shape_vertices)

    def search(self, shape_type: CQType, include_child_shape=False):
        candidate_shapes = self.explorer.search(shape_type)
        if not include_child_shape:
            child_shapes = ShapeExplorer(self.child_shape).search(shape_type)
            candidate_shapes = [
                shape for shape in candidate_shapes if shape not in child_shapes
            ]

        connected_shapes = []
        for shape in candidate_shapes:
            if self._connected_by_vertices(shape):
                connected_shapes.append(shape)
        return connected_shapes
