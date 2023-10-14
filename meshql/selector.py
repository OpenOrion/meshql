from dataclasses import dataclass
import cadquery as cq
from cadquery.cq import CQObject
from typing import Callable, Optional, Protocol, Sequence, Union
from meshql.utils.cq import SHAPE_TYPE_CLASS_MAPPING, CQGroupTypeString, CQLinq, ShapeType
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

class Selectable(Protocol):
    workplane: cq.Workplane
    initial_workplane: cq.Workplane
    type_groups: dict[CQGroupTypeString, OrderedSet[CQObject]]

@dataclass
class Selection:
    selector: Union[cq.Selector, str, None] = None
    tag: Union[str, None] = None
    type: Optional[CQGroupTypeString] = None
    indices: Optional[Sequence[int]] = None 
    filter: Optional[Callable[[CQObject], bool]] = None

    def select(self, selectable: Selectable, shape_type: ShapeType, is_initial: bool = False, is_exclusive: bool = False, is_intersection: bool = False):
        workplane = selectable.initial_workplane if is_initial else selectable.workplane
        cq_obj = workplane._getTagged(self.tag) if self.tag else workplane
        filtered_entities = list(CQLinq.select(cq_obj, shape_type))

        if isinstance(self.selector, str):
            filtered_entities = cq.StringSyntaxSelector(self.selector).filter(filtered_entities)
        elif isinstance(self.selector, cq.Selector):
            filtered_entities = self.selector.filter(filtered_entities)

        if self.type:
            inv_type = "exterior" if self.type == "interior" else "interior"
            if is_exclusive:
                type_group = self.type and selectable.type_groups[self.type].difference(selectable.type_groups[inv_type])
            elif is_intersection:
                type_group = self.type and selectable.type_groups[self.type].intersection(selectable.type_groups[inv_type])
            else:
                type_group = selectable.type_groups[self.type]
            filtered_entities = GroupSelector(type_group).filter(filtered_entities)

        if self.indices is not None:
            filtered_entities = IndexSelector(self.indices).filter(filtered_entities)
        if self.filter is not None:
            filtered_entities = FilterSelector(self.filter).filter(filtered_entities)

        return filtered_entities

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
    "compound": TopAbs_COMPOUND,
    "solid": TopAbs_SOLID,
    "shell": TopAbs_SHELL,
    "face": TopAbs_FACE,
    "wire": TopAbs_WIRE,
    "edge": TopAbs_EDGE,
    "vertex": TopAbs_VERTEX,
}

class ShapeExplorer:
    def __init__(self, shape: cq.Shape) -> None:
        self.shape = shape.wrapped
        self.explorer = TopExp_Explorer()

    def search(
        self, shape_type: ShapeType, not_from: Optional[ShapeType] = None,
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

    def search(self, shape_type: ShapeType, include_child_shape=False):
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


# if __name__ == "__main__":
#     import cadquery as cq
#     from jupyter_cadquery.viewer.client import show

#     box = cq.Workplane().box(10, 10, 10).faces(">Z").connected("Edge", True)

#     show(
#         box,
#         height=800,
#         cad_width=1500,
#         reset_camera=False,
#         default_edgecolor=(255, 255, 255),
#         zoom=1,
#         axes=True,
#         axes0=True,
#         render_edges=True,
#     )