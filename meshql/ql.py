import cadquery as cq
from cadquery.cq import CQObject
from typing import Callable, Literal, Optional, Protocol, Sequence, Type, Union
from meshql.boundary_condition import BoundaryCondition
from meshql.gmsh.entity import Entity

from meshql.preprocessing.preprocess import Preprocess
from meshql.selector import FilterSelector, GroupSelector, IndexSelector, Selection
from meshql.utils.cq import (
    CQ_TYPE_STR_MAPPING,
    CQUtils,
    GroupType,
    CQType,
)

from meshql.utils.cq_cache import CQCache
from meshql.utils.cq_linq import CQLinq, SetOperation
from meshql.utils.plot import plot_cq
from meshql.utils.types import OrderedSet
from meshql.utils.logging import logger
from meshql.utils.mesh_visualizer import visualize_mesh


ShowType = Literal["mesh", "cq", "plot"]


class GeometryQLContext:
    def __init__(self, tol: Optional[float] = None):
        self.tol = tol
        self.initial_workplane = cq.Workplane()
        self.selection: Optional[Selection] = None
        self.boundary_conditions = dict[str, BoundaryCondition]()
        self.is_2d = False
        self.is_split = False
        self.region_groups: Optional[dict[GroupType, OrderedSet[CQObject]]] = None


class GeometryQL:
    def __init__(
        self,
        ctx: Optional[GeometryQLContext] = None,
        workplane: Optional[cq.Workplane] = None,
        selection: Optional[Selection] = None,
        prev_ql: Optional["GeometryQL"] = None,
    ) -> None:
        self._ctx = ctx or GeometryQLContext()
        self._workplane = workplane or cq.Workplane()
        self._selection = selection
        self._prev_ql = prev_ql

    @staticmethod
    def gmsh():
        from meshql.gmsh.ql import GmshGeometryQL

        return GmshGeometryQL()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def load(
        self,
        target: Union[cq.Workplane, str, Sequence[CQObject]],
        on_preprocess: Optional[Callable[["GeometryQL"], Preprocess]] = None,
    ):
        imported_workplane = CQUtils.import_workplane(target)

        # extrudes 2D shapes to 3D
        self._ctx.is_2d = CQUtils.get_dimension(imported_workplane) == 2
        if self._ctx.is_2d:
            workplane_3d = imported_workplane.extrude(-1)
        else:
            workplane_3d = imported_workplane

        self._workplane = workplane_3d
        self._ctx.initial_workplane = workplane_3d

        if on_preprocess:
            self.is_split = True
            self._workplane = on_preprocess(self).apply()

        if self._ctx.is_2d:
            # fuses top faces to appear as one Compound in GMSH
            faces = self._workplane.faces(">Z").vals()
            fused_face = CQUtils.fuse_shapes(faces)
            self._workplane = cq.Workplane(fused_face)

    def end(self, num: int = 1):
        ql = self
        for _ in range(num):
            ql = ql._prev_ql
        return ql

    def select(
        self,
        selection: Selection,
        cq_type: CQType,
    ):

        if selection.tag:
            cq_obj = self.fromTagged(selection.tag)
            filtered_objs = CQLinq.select(cq_obj, cq_type)
        else:
            filtered_objs = CQLinq.select(self._workplane, cq_type)

        if isinstance(selection.selector, str):
            filtered_objs = cq.StringSyntaxSelector(selection.selector).filter(
                filtered_objs
            )
        elif isinstance(selection.selector, cq.Selector):
            filtered_objs = selection.selector.filter(filtered_objs)

        if selection.type:
            region_groups = self._get_region_groups()
            filtered_objs = GroupSelector(region_groups[selection.type]).filter(
                filtered_objs
            )

        if selection.region_set_operation:
            region_groups = self._get_region_groups()

            if selection.type:
                region_type = selection.type
            else:
                assert (
                    self._selection and self._selection.type
                ), "Group selection not defined for set operation"
                region_type = self._selection.type
            filtered_objs = CQLinq.groupBySet(
                filtered_objs,
                region_type,
                region_groups,
                selection.region_set_operation,
            )

        if selection.indices is not None:
            filtered_objs = IndexSelector(selection.indices).filter(filtered_objs)
        if selection.filter is not None:
            filtered_objs = FilterSelector(selection.filter).filter(filtered_objs)

        workplane = self._workplane.newObject(filtered_objs)
        return self.__class__(self._ctx, workplane, selection, self)

    def solids(
        self,
        selector: Union[cq.Selector, str, None] = None,
        tag: Union[str, None] = None,
        type: Optional[GroupType] = None,
        indices: Optional[Sequence[int]] = None,
        filter: Optional[Callable[[CQObject], bool]] = None,
    ):
        selection = Selection(selector, tag, type, indices, filter)
        self.select(selection, "solid")

    def faces(
        self,
        selector: Union[cq.Selector, str, None] = None,
        tag: Union[str, None] = None,
        type: Optional[GroupType] = None,
        indices: Optional[Sequence[int]] = None,
        filter: Optional[Callable[[CQObject], bool]] = None,
    ):
        selection = Selection(selector, tag, type, indices, filter)
        return self.select(selection, "face")

    def edges(
        self,
        selector: Union[cq.Selector, str, None] = None,
        tag: Union[str, None] = None,
        type: Optional[GroupType] = None,
        indices: Optional[Sequence[int]] = None,
        filter: Optional[Callable[[CQObject], bool]] = None,
    ):
        selection = Selection(selector, tag, type, indices, filter)
        return self.select(selection, "edge")

    def wires(
        self,
        selector: Union[cq.Selector, str, None] = None,
        tag: Union[str, None] = None,
        type: Optional[GroupType] = None,
        indices: Optional[Sequence[int]] = None,
        filter: Optional[Callable[[CQObject], bool]] = None,
    ):
        selection = Selection(selector, tag, type, indices, filter)
        return self.select(selection, "wire")

    def vertices(
        self,
        selector: Union[cq.Selector, str, None] = None,
        tag: Union[str, None] = None,
        type: Optional[GroupType] = None,
        indices: Optional[Sequence[int]] = None,
        filter: Optional[Callable[[CQObject], bool]] = None,
    ):
        selection = Selection(selector, tag, type, indices, filter)
        return self.select(selection, "vertex")

    def fromTagged(
        self,
        tags: Union[str, Sequence[str]],
        resolve_type: Optional[CQType] = None,
        invert: bool = True,
    ):
        if isinstance(tags, str) and resolve_type is None:
            workplane = self._workplane._getTagged(tags)
        else:
            tagged_objs = list(
                CQLinq.select_tagged(self._workplane, tags, resolve_type)
            )
            tagged_cq_type = CQ_TYPE_STR_MAPPING[type(tagged_objs[0])]
            workplane_objs = CQLinq.select(self._workplane, tagged_cq_type)
            filtered_objs = CQLinq.filter(workplane_objs, tagged_objs, invert)
            workplane = self._workplane.newObject(filtered_objs)
        return self.__class__(self._ctx, workplane, Selection(), self)

    def _get_region_groups(self):
        if self._ctx.region_groups is None:
            self._ctx.region_groups = CQLinq.groupByRegionTypes(
                self._ctx.initial_workplane,
                self._ctx.tol,
                check_splits=self._ctx.is_split,
            )
        return self._ctx.region_groups

    def tag(self, names: Union[str, Sequence[str]]):
        if isinstance(names, str):
            self._workplane.tag(names)
        else:
            for i, cq_obj in enumerate(self._workplane.vals()):
                self._workplane.newObject([cq_obj]).tag(names[i])
        return self

    def vals(self):
        return self._workplane.vals()

    def val(self):
        return self._workplane.val()

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
            plot_cq(self._workplane)
        elif type == "cq":
            from jupyter_cadquery import show

            try:
                if only_faces:
                    root_assembly = cq.Assembly()
                    for i, face in enumerate(self._workplane.faces().vals()):
                        root_assembly.add(cq.Workplane(face), name=f"face/{i+1}")
                    show(root_assembly, theme=theme)
                else:
                    show(self._workplane, theme=theme)
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
            for i, face in enumerate(self._workplane.vals()):
                assert isinstance(
                    face, cq.Face
                ), "Boundary condition can only be applied to faces"
                group_val = group(i, face)
                group_label = group_val.label
                self._ctx.boundary_conditions[group_label] = group_val
                self._addEntityGroup(group_label, self.vals())
        else:
            group_label = group.label
            assert (
                group_label not in self._ctx.boundary_conditions
            ), f"Boundary condition {group.label} added already"
            self._ctx.boundary_conditions[group.label] = group
            self._addEntityGroup(group_label, self.vals())

        return self
