import cadquery as cq
from cadquery.cq import CQObject
from typing import Callable, Literal, Optional, Sequence, Union
from meshql.boundary_condition import BoundaryCondition
from meshql.gmsh.entity import CQEntityContext, Entity

from meshql.selector import Selectable, Selection
from meshql.utils.cq import (
    GroupTypeString,
    CQType,
)

from meshql.utils.cq_cache import CQCache
from meshql.utils.plot import plot_cq
from meshql.utils.types import OrderedSet
from meshql.utils.logging import logger
from meshql.utils.mesh_visualizer import visualize_mesh

ShowType = Literal["mesh", "cq", "plot"]


class GeometryQL(Selectable):
    entity_ctx: CQEntityContext

    def __init__(self, tol: Optional[float] = None) -> None:
        super().__init__(tol)
        self.boundary_conditions = dict[str, BoundaryCondition]()
        self.tagged_workplane = False

    @staticmethod
    def gmsh():
        from meshql.gmsh.ql import GmshGeometryQL

        return GmshGeometryQL()

    def __enter__(self):
        CQCache.load_cache()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        CQCache.save_cache()
        return

    def end(self, num: int = 1):
        self.workplane = self.workplane.end(num)
        return self

    def solids(
        self,
        selector: Union[cq.Selector, str, None] = None,
        tag: Union[str, None] = None,
        type: Optional[GroupTypeString] = None,
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
        type: Optional[GroupTypeString] = None,
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
        type: Optional[GroupTypeString] = None,
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
        type: Optional[GroupTypeString] = None,
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
        type: Optional[GroupTypeString] = None,
        indices: Optional[Sequence[int]] = None,
        filter: Optional[Callable[[CQObject], bool]] = None,
    ):
        selection = Selection(selector, tag, type, indices, filter)
        self.workplane = self.workplane.newObject(selection.select(self, "vertex"))
        return self

    def fromTagged(
        self,
        tags: Union[str, Sequence[str]],
        resolve_type: Optional[CQType] = None,
        invert: bool = True,
    ):

        # check if the tags are entity tags, then lazy load tagged workplane
        is_tag_entity = False
        if isinstance(tags, str):
            is_tag_entity = "/" in tags
        else:
            for tag in tags:
                if "/" in tag:
                    is_tag_entity = True
                    break

        if is_tag_entity and not self.tagged_workplane:
            self._tag_workplane()
            self.tagged_workplane = True

        return super().fromTagged(tags, resolve_type, invert)

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
            plot_cq(self.workplane, ctx=self.entity_ctx)
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
