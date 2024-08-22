import cadquery as cq
from cadquery.cq import CQObject
from typing import Callable, Iterable, Literal, Optional, Sequence, Union, cast
from cadquery.selectors import Selector
from meshql.boundary_condition import BoundaryCondition
from meshql.entity import CQEntityContext, Entity
from meshql.preprocessing.split import split_workplane

from meshql.utils.cq import (
    CQ_TYPE_STR_MAPPING,
    CQExtensions,
    CQGroupTypeString,
    CQLinq,
    CQType,
)
from meshql.utils.types import OrderedSet
from meshql.visualizer import visualize_mesh

ShowType = Literal["mesh", "cq", "plot"]


class GeometryQL:
    _workplane: cq.Workplane
    _initial_workplane: cq.Workplane
    _entity_ctx: CQEntityContext
    _type_groups: Optional[dict[CQGroupTypeString, OrderedSet[CQObject]]] = None

    @staticmethod
    def gmsh():
        from meshql.gmsh.ql import GmshGeometryQL

        return GmshGeometryQL()

    def __init__(self) -> None:
        self._initial_workplane = self._workplane = None  # type: ignore
        self.boundary_conditions = dict[str, BoundaryCondition]()
        self._type_groups = None
        self.is_2d = False

    def get_obj_type(self, type: Optional[CQGroupTypeString] = None):
        if type and self._type_groups is None:
            self._type_groups = CQLinq.groupByTypes(
                self._initial_workplane, exclude_split=self.is_2d
            )
            return self._type_groups[type]


    def solids(
        self,
        selector: Union[Selector, str, None] = None,
        tag: Union[str, None] = None,
        type: Optional[CQGroupTypeString] = None,
        indices: Optional[Sequence[int]] = None,
    ):
        obj_type = self.get_obj_type(type)
        selector = CQExtensions.get_selector(selector, obj_type, indices)
        self._workplane = self._workplane.solids(selector, tag)
        return self

    def shells(
        self,
        selector: Union[Selector, str, None] = None,
        tag: Union[str, None] = None,
        type: Optional[CQGroupTypeString] = None,
        indices: Optional[Sequence[int]] = None,
    ):
        obj_type = self.get_obj_type(type)
        selector = CQExtensions.get_selector(selector, obj_type, indices)
        self._workplane = self._workplane.shells(selector, tag)
        return self

    def faces(
        self,
        selector: Union[Selector, str, None] = None,
        tag: Union[str, None] = None,
        type: Optional[CQGroupTypeString] = None,
        indices: Optional[Sequence[int]] = None,
    ):
        obj_type = self.get_obj_type(type)
        selector = CQExtensions.get_selector(selector, obj_type, indices)
        self._workplane = self._workplane.faces(selector, tag)
        return self

    def edges(
        self,
        selector: Union[Selector, str, None] = None,
        tag: Union[str, None] = None,
        type: Optional[CQGroupTypeString] = None,
        indices: Optional[Sequence[int]] = None,
    ):
        obj_type = self.get_obj_type(type)
        selector = CQExtensions.get_selector(selector, obj_type, indices)
        self._workplane = self._workplane.edges(selector, tag)
        return self

    def wires(
        self,
        selector: Union[Selector, str, None] = None,
        tag: Union[str, None] = None,
        type: Optional[CQGroupTypeString] = None,
        indices: Optional[Sequence[int]] = None,
    ):
        obj_type = self.get_obj_type(type)
        selector = CQExtensions.get_selector(selector, obj_type, indices)
        self._workplane = self._workplane.wires(selector, tag)
        return self

    def vertices(
        self,
        selector: Union[Selector, str, None] = None,
        tag: Union[str, None] = None,
        type: Optional[CQGroupTypeString] = None,
        indices: Optional[Sequence[int]] = None,
    ):
        obj_type = self.get_obj_type(type)
        selector = CQExtensions.get_selector(selector, obj_type, indices)
        self._workplane = self._workplane.vertices(selector, tag)
        return self

    def tag(self, names: Union[str, Sequence[str]]):
        if isinstance(names, str):
            self._workplane.tag(names)
        else:
            for i, cq_obj in enumerate(self._workplane.vals()):
                self._workplane.newObject([cq_obj]).tag(names[i])
        return self

    def fromTagged(
        self,
        tags: Union[str, Iterable[str]],
        resolve_type: Optional[CQType] = None,
        invert: bool = True,
    ):
        if isinstance(tags, str) and resolve_type is None:
            self._workplane = self._workplane._getTagged(tags)
        else:
            tagged_objs = list(
                CQLinq.select_tagged(self._workplane, tags, resolve_type)
            )
            tagged_cq_type = CQ_TYPE_STR_MAPPING[type(tagged_objs[0])]
            workplane_objs = CQLinq.select(self._workplane, tagged_cq_type)
            filtered_objs = CQLinq.filter(workplane_objs, tagged_objs, invert)
            self._workplane = self._workplane.newObject(filtered_objs)
        return self


    def _after_load(self): ...

    def vals(self):
        return self._entity_ctx.select_many(self._workplane)

    def val(self):
        return self._entity_ctx.select(self._workplane.val())

    def _tag_workplane(self):
        "Tag all gmsh entity tags to workplane"
        for cq_type, registry in self._entity_ctx.entity_registries.items():
            for occ_obj in registry.keys():
                tag = f"{cq_type}/{registry[occ_obj].tag}"
                self._workplane.newObject([occ_obj]).tag(tag)

    @property
    def mesh(self): ...

    def show(
        self,
        type: ShowType = "cq",
        only_markers: bool = False,
    ):
        if type == "mesh":
            assert self.mesh is not None, "Mesh is not generated yet."
            visualize_mesh(self.mesh, only_markers=only_markers)
        elif type == "plot":
            CQExtensions.plot_cq(self._workplane, ctx=self._entity_ctx)
        elif type == "cq":
            from jupyter_cadquery import show

            show(self._workplane, theme="dark")
        else:
            raise NotImplementedError(f"Unknown show type {type}")
        return self

    def _addEntityGroup(
        self,
        group_name: str,
        entities: OrderedSet[Entity],
        boundary_condition: Optional[BoundaryCondition] = None,
    ): ...

    def addBoundaryCondition(
        self,
        group: Union[
            str,
            Sequence[str],
            BoundaryCondition,
            Callable[[int, cq.Face], BoundaryCondition],
        ],
    ):
        if isinstance(group, Sequence):
            for i, group_name in enumerate(group):
                new_group_entity = list(self.vals())[i]
                self._addEntityGroup(group_name, OrderedSet([new_group_entity]))
        else:
            if isinstance(group, str):
                group_label = group
            elif isinstance(group, Callable):
                for i, face in enumerate(self._workplane.vals()):
                    assert isinstance(
                        face, cq.Face
                    ), "Boundary condition can only be applied to faces"
                    group_val = group(i, face)
                    group_label = group_val.label
                    assert (
                        group_label not in self.boundary_conditions
                    ), f"Boundary condition {group_label} added already"
                    self.boundary_conditions[group_label] = group_val
            else:
                group_label = group.label
                assert (
                    group_label not in self.boundary_conditions
                ), f"Boundary condition {group.label} added already"
                self.boundary_conditions[group.label] = group

            self._addEntityGroup(group_label, self.vals())

        return self
