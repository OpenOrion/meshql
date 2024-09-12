from dataclasses import dataclass
import cadquery as cq
from cadquery.cq import CQObject
from typing import Callable, Optional, Sequence, Union
from meshql.utils.cq import (
    CQ_TYPE_STR_MAPPING,
    CQUtils,
    GroupTypeString,
    CQType,
)
from meshql.utils.cq_linq import CQLinq
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


class Selectable:
    workplane: cq.Workplane
    initial_workplane: cq.Workplane
    initial_3d_workplane: cq.Workplane
    type_groups: Optional[dict[GroupTypeString, OrderedSet[CQObject]]]

    def __init__(self, tol: Optional[float] = None) -> None:
        self.initial_workplane = self.workplane = cq.Workplane("XY")
        self.tol = tol
        self.type_groups = None
        self.tagged_workplane = False

    def load_workplane(
        self,
        target: Union[cq.Workplane, str, Sequence[CQObject]],
        on_split: Optional[Callable[[str], cq.Workplane]] = None,
    ):
        from meshql.preprocessing.split import Split

        workplane = CQUtils.import_workplane(target)

        # extrudes 2D shapes to 3D
        is_2d = CQUtils.get_dimension(workplane) == 2
        if is_2d:
            workplane = workplane.extrude(-1)

        if on_split:
            split = Split(workplane, self.tol)
            workplane = on_split(split).apply().workplane

        self.initial_3d_workplane = workplane

        if is_2d:
            # fuses top faces to appear as one Compound in GMSH
            faces = workplane.faces(">Z").vals()
            fused_face = CQUtils.fuse_shapes(faces)
            workplane = cq.Workplane(fused_face)

        self.workplane = self.initial_workplane = workplane

    def _get_group(self, group_type: Optional[GroupTypeString] = None):
        if group_type:
            if self.type_groups is None:
                self.type_groups = CQLinq.groupByTypes(
                    self.initial_3d_workplane, self.tol, check_splits=False
                )
            return self.type_groups[group_type]
        return OrderedSet[CQObject]()

    def fromTagged(
        self,
        tags: Union[str, Sequence[str]],
        resolve_type: Optional[CQType] = None,
        invert: bool = True,
    ):
        if isinstance(tags, str) and resolve_type is None:
            self.workplane = self.workplane._getTagged(tags)
        else:
            tagged_objs = list(CQLinq.select_tagged(self.workplane, tags, resolve_type))
            tagged_cq_type = CQ_TYPE_STR_MAPPING[type(tagged_objs[0])]
            workplane_objs = CQLinq.select(self.workplane, tagged_cq_type)
            filtered_objs = CQLinq.filter(workplane_objs, tagged_objs, invert)
            self.workplane = self.workplane.newObject(filtered_objs)
        return self

    def tag(self, names: Union[str, Sequence[str]]):
        if isinstance(names, str):
            self.workplane.tag(names)
        else:
            for i, cq_obj in enumerate(self.workplane.vals()):
                self.workplane.newObject([cq_obj]).tag(names[i])
        return self


@dataclass
class Selection:
    selector: Union[cq.Selector, str, None] = None
    tag: Union[str, Sequence[str], None] = None
    group_type: Optional[GroupTypeString] = None
    indices: Optional[Sequence[int]] = None
    filter: Optional[Callable[[CQObject], bool]] = None

    def select(
        self,
        selectable: Selectable,
        cq_type: CQType,
        is_exclusive: bool = False,
        is_intersection: bool = False,
    ):
        if self.tag:
            cq_obj = selectable.fromTagged(self.tag)
            filtered_entities = CQLinq.select(cq_obj, cq_type)
        else:
            filtered_entities = CQLinq.select(selectable.workplane, cq_type)

        if isinstance(self.selector, str):
            filtered_entities = cq.StringSyntaxSelector(self.selector).filter(
                filtered_entities
            )
        elif isinstance(self.selector, cq.Selector):
            filtered_entities = self.selector.filter(filtered_entities)

        if self.group_type:
            inv_type = "exterior" if self.group_type == "interior" else "interior"
            if is_exclusive:
                group = self.group_type and selectable._get_group(
                    self.group_type
                ).difference(selectable.type_groups[inv_type])
            elif is_intersection:
                group = self.group_type and selectable._get_group(
                    self.group_type
                ).intersection(selectable.type_groups[inv_type])
            else:
                group = selectable._get_group(self.group_type)
            filtered_entities = GroupSelector(group).filter(filtered_entities)

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
