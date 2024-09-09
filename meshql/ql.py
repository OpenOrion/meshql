import cadquery as cq
from cadquery.cq import CQObject
from typing import Callable, Iterable, Literal, Optional, Sequence, Union
from meshql.boundary_condition import BoundaryCondition
from meshql.entity import CQEntityContext, Entity

from meshql.selector import Selectable, Selection
from meshql.utils.cq import (
    CQ_TYPE_STR_MAPPING,
    CQExtensions,
    CQGroupTypeString,
    CQLinq,
    CQType,
)
from meshql.utils.types import OrderedSet
from meshql.utils.logging import logger
from meshql.visualizer import visualize_mesh

ShowType = Literal["mesh", "cq", "plot"]


class GeometryQL(Selectable):
    workplane: cq.Workplane
    initial_workplane: cq.Workplane
    entity_ctx: CQEntityContext
    type_groups: Optional[dict[CQGroupTypeString, OrderedSet[CQObject]]] = None
    refresh_type_groups: Callable[[], dict[CQGroupTypeString, OrderedSet[CQObject]]]

    @staticmethod
    def gmsh():
        from meshql.gmsh.ql import GmshGeometryQL

        return GmshGeometryQL()

    def __init__(self) -> None:
        self.initial_workplane = self.workplane = None  # type: ignore
        self.boundary_conditions = dict[str, BoundaryCondition]()
        self.type_groups = None

    def solids(
        self,
        selector: Union[cq.Selector, str, None] = None,
        tag: Union[str, None] = None,
        type: Optional[CQGroupTypeString] = None,
        indices: Optional[Sequence[int]] = None,
        filter: Optional[Callable[[CQObject], bool]] = None,
    ):
        selection = Selection(selector, tag, type, indices, filter)
        self.workplane = self.workplane.newObject(selection.select(self, "solid"))
        return self

    def faces(
        self,
        selector: Union[cq.Selector, str, None] = None,
        tag: Union[str, None] = None,
        type: Optional[CQGroupTypeString] = None,
        indices: Optional[Sequence[int]] = None,
        filter: Optional[Callable[[CQObject], bool]] = None,
    ):
        selection = Selection(selector, tag, type, indices, filter)
        self.workplane = self.workplane.newObject(selection.select(self, "face"))
        return self

    def edges(
        self,
        selector: Union[cq.Selector, str, None] = None,
        tag: Union[str, None] = None,
        type: Optional[CQGroupTypeString] = None,
        indices: Optional[Sequence[int]] = None,
        filter: Optional[Callable[[CQObject], bool]] = None,
    ):
        selection = Selection(selector, tag, type, indices, filter)
        self.workplane = self.workplane.newObject(selection.select(self, "edge"))
        return self

    def wires(
        self,
        selector: Union[cq.Selector, str, None] = None,
        tag: Union[str, None] = None,
        type: Optional[CQGroupTypeString] = None,
        indices: Optional[Sequence[int]] = None,
        filter: Optional[Callable[[CQObject], bool]] = None,
    ):
        selection = Selection(selector, tag, type, indices, filter)
        self.workplane = self.workplane.newObject(selection.select(self, "wire"))
        return self

    def vertices(
        self,
        selector: Union[cq.Selector, str, None] = None,
        tag: Union[str, None] = None,
        type: Optional[CQGroupTypeString] = None,
        indices: Optional[Sequence[int]] = None,
        filter: Optional[Callable[[CQObject], bool]] = None,
    ):
        selection = Selection(selector, tag, type, indices, filter)
        self.workplane = self.workplane.newObject(selection.select(self, "vertex"))
        return self

    def fromTagged(
        self,
        tags: Union[str, Iterable[str]],
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

    def _after_load(self): ...

    def vals(self):
        return self.entity_ctx.select_many(self.workplane)

    def val(self):
        return self.entity_ctx.select(self.workplane.val())

    def _tag_workplane(self):
        "Tag all gmsh entity tags to workplane"
        for cq_type, registry in self.entity_ctx.entity_registries.items():
            for occ_obj in registry.keys():
                tag = f"{cq_type}/{registry[occ_obj].tag}"
                self.workplane.newObject([occ_obj]).tag(tag)

    @property
    def mesh(self): ...

    def show(
        self,
        type: ShowType = "cq",
        theme: Literal["light", "dark"] = "light",
        only_faces: bool = False,
        only_markers: bool = False,
    ):
        if type == "mesh":
            assert self.mesh is not None, "Mesh is not generated yet."
            visualize_mesh(self.mesh, only_markers=only_markers)
        elif type == "plot":
            CQExtensions.plot_cq(self.workplane, ctx=self.entity_ctx)
        elif type == "cq":
            from jupyter_cadquery import show

            try:
                if only_faces:
                    root_assembly = cq.Assembly()
                    for i, face in enumerate(self.workplane.faces().vals()):
                        root_assembly.add(cq.Workplane(face), name=f"face/{i+1}")
                    show(root_assembly, theme=theme)
                else:
                    show(self.workplane, theme=theme)
            except:
                logger.warn("inadequate CQ geometry, trying to display in GMSH ...")

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
            BoundaryCondition,
            Callable[[int, cq.Face], BoundaryCondition],
        ],
    ):
        if isinstance(group, Callable):
                for i, face in enumerate(self.workplane.vals()):
                    assert isinstance(
                        face, cq.Face
                    ), "Boundary condition can only be applied to faces"
                    group_val = group(i, face)
                    group_label = group_val.label
                    self.boundary_conditions[group_label] = group_val
                    self._addEntityGroup(group_label, self.vals())
        else:
            group_label = group.label
            assert (
                group_label not in self.boundary_conditions
            ), f"Boundary condition {group.label} added already"
            self.boundary_conditions[group.label] = group
            self._addEntityGroup(group_label, self.vals())

        return self
