from functools import cached_property
import cadquery as cq
from cadquery.cq import CQObject
from typing import Callable, Literal, Optional, Sequence, Union, TypeVar, Type

from pydantic import Field
from meshql.gmsh.entity import CQEntityMapper, Entity

from meshql.core.transaction import TransactionContext, Transaction
from meshql.preprocessing.preprocess import Preprocessor
from meshql.core.selector import FilterSelector, GroupSelector, IndexSelector, Selection
from meshql.utils.cache import CacheService
from meshql.utils.cq import CQUtils
from meshql.utils.types import (
    CQ_TYPE_STR_MAPPING,
    RegionGroupType,
    CQType,
    PathLike,
)

from meshql.utils.cq_linq import CQLinq
from meshql.utils.plot import plot_cq
from meshql.utils.types import OrderedSet, LoadTarget
from meshql.utils.logging import logger
from meshql.utils.mesh_visualizer import visualize_mesh
from meshql.mesh.loaders import load_to_gmsh
import meshly

ShowType = Literal["mesh", "cq", "plot"]
TPreprocessor = TypeVar("TPreprocessor", bound=Preprocessor)


class GeometryQLContext(TransactionContext):

    is_2d: bool = False
    is_structured: bool = False
    transfinite_edge_groups: list[set[cq.Edge]] = []
    region_group_types: Optional[dict[str, RegionGroupType]] = None
    preprocessor: Optional[Preprocessor] = None
    tol: Optional[float] = None
    initial_workplane: Optional[cq.Workplane] = Field(
        exclude=True, default=None)

    @cached_property
    def entities(self) -> CQEntityMapper:
        if self.initial_workplane is None:
            raise ValueError("Initial workplane is not set in the context.")
        return CQEntityMapper(self.initial_workplane)

    @property
    def is_split(self) -> bool:
        from meshql.preprocessing.split import Split
        return isinstance(self.preprocessor, Split)


class GeometryQL:
    def __init__(
        self,
        ctx: Optional[GeometryQLContext] = None,
        workplane: Optional[cq.Workplane] = None,
        selection: Optional[Selection] = None,
        prev_ql: Optional["GeometryQL"] = None,
    ) -> None:
        self._ctx = ctx or GeometryQLContext()
        self._mesh: Optional[meshly.Mesh] = None
        self._workplane = workplane
        self._selection = selection
        self._prev_ql = prev_ql

    @staticmethod
    def gmsh():
        from meshql.gmsh.ql import GmshGeometryQL

        return GmshGeometryQL()

    @staticmethod
    def compute_mesh(
        target: LoadTarget,
        transactions: Sequence[Transaction],
        preprocess: Optional[Union[Preprocessor, tuple[type[TPreprocessor],
                                                       Callable[[TPreprocessor], TPreprocessor]]]] = None,
        dim: int = 3,
    ) -> meshly.Mesh:
        """
        Compute a mesh from a target geometry with preprocessing and transactions.

        Args:
            target: The geometry to load (LoadTarget - can be a file path, Workplane, or Mesh)
            transactions: List of Transaction objects to apply (e.g., mesh refinement, physical groups)
            preprocess: Optional tuple of (Preprocessor type, preprocessor configuration function)
            dim: Mesh dimension (2 or 3, default: 3)

        Returns:
            meshly.Mesh: The generated mesh

        Example:
            from meshql import GeometryQL
            from meshql.gmsh.refinement import SetMeshSize
            from meshql.gmsh.physical_group import SetPhysicalGroup

            mesh = GeometryQL.compute_mesh(
                target=cq.Workplane("XY").box(10, 10, 10),
                transactions=[
                    SetMeshSize(entities=..., size=0.5),
                    SetPhysicalGroup(entities=..., name="boundary")
                ],
                dim=3
            )
        """
        from meshql.gmsh.ql import GmshGeometryQL
        import gmsh

        gmsh.initialize()
        try:
            geo = GmshGeometryQL()
            geo.load(target, preprocess)

            # Add all transactions to the context
            geo._ctx.add_transactions(transactions)

            # Generate the mesh
            geo.generate(dim)

            return geo._mesh
        finally:
            gmsh.finalize()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def load(
        self,
        target: LoadTarget,
        preprocess:  Optional[Union[Preprocessor, tuple[type[TPreprocessor],
                                    Callable[[TPreprocessor], TPreprocessor]]]] = None,
    ):
        is_meshly_zip_path = (isinstance(target, PathLike)
                              and str(target).lower().endswith(".zip"))
        is_meshly_su2_path = (isinstance(target, PathLike)
                              and str(target).lower().endswith(".su2"))

        if isinstance(target, meshly.Mesh) or is_meshly_zip_path or is_meshly_su2_path:
            if is_meshly_zip_path:
                target = meshly.MeshUtils.load_from_zip(meshly.Mesh, target)
                print(f"Loaded meshly mesh from {target}")
            elif is_meshly_su2_path:
                from su2fmt import parse_mesh
                mesh_result = parse_mesh(target)
                if isinstance(mesh_result, list):
                    target = meshly.Mesh.combine(mesh_result)
                else:
                    target = mesh_result
            # Import meshly mesh directly into gmsh
            assert preprocess is None, "Preprocessing not supported for mesh targets"
            load_to_gmsh(target)
            self._workplane = None
            self._mesh = target
        else:
            imported_workplane = CQUtils.import_workplane(target)

            # extrudes 2D shapes to 3D
            self._ctx.is_2d = CQUtils.get_dimension(imported_workplane) == 2
            if self._ctx.is_2d:
                workplane_3d = imported_workplane.extrude(-1)
            else:
                workplane_3d = imported_workplane

            self._workplane = self._ctx.initial_workplane = workplane_3d
            if preprocess:
                if isinstance(preprocess, Preprocessor):
                    self._ctx.preprocessor = preprocess
                else:
                    preprocess_type, preprocess_callback = preprocess
                    initial_preprocessor = preprocess_type(
                        workplane=self._workplane)
                    self._ctx.preprocessor = preprocess_callback(
                        initial_preprocessor)

                applied_preprocessor = self._ctx.preprocessor.apply()
                self._workplane = self._ctx.initial_workplane = applied_preprocessor.workplane
                # Set region_groups from preprocessor (already cached within apply())
                self._ctx.region_group_types = self._ctx.preprocessor.region_group_types
            if self._ctx.is_2d:
                # fuses top faces to appear as one Compound in GMSH
                faces = self._workplane.faces(">Z").vals()
                fused_face = CQUtils.fuse_shapes(faces)
                self._workplane = cq.Workplane(fused_face)

    def print_val(self):
        print(f"Workplane Val: {self.ql._workplane.val()}")
        return self

    def end(self, num: int = 1):
        ql = self
        for _ in range(num):
            ql = ql._prev_ql
        assert ql is not None, "previous ql state not defined"
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
            filtered_objs = IndexSelector(
                selection.indices).filter(filtered_objs)
        if selection.filter is not None:
            filtered_objs = FilterSelector(
                selection.filter).filter(filtered_objs)

        workplane = self._workplane.newObject(filtered_objs)
        return self.__class__(self._ctx, workplane, selection, self)

    def solids(
        self,
        selector: Union[cq.Selector, str, None] = None,
        tag: Union[str, None] = None,
        type: Optional[RegionGroupType] = None,
        indices: Optional[Sequence[int]] = None,
        filter: Optional[Callable[[CQObject], bool]] = None,
    ):
        selection = Selection(selector, tag, type, indices, filter)
        self.select(selection, "Solid")

    def faces(
        self,
        selector: Union[cq.Selector, str, None] = None,
        tag: Union[str, None] = None,
        type: Optional[RegionGroupType] = None,
        indices: Optional[Sequence[int]] = None,
        filter: Optional[Callable[[CQObject], bool]] = None,
    ):
        selection = Selection(selector, tag, type, indices, filter)
        return self.select(selection, "Face")

    def edges(
        self,
        selector: Union[cq.Selector, str, None] = None,
        tag: Union[str, None] = None,
        type: Optional[RegionGroupType] = None,
        indices: Optional[Sequence[int]] = None,
        filter: Optional[Callable[[CQObject], bool]] = None,
    ):
        selection = Selection(selector, tag, type, indices, filter)
        return self.select(selection, "Edge")

    def wires(
        self,
        selector: Union[cq.Selector, str, None] = None,
        tag: Union[str, None] = None,
        type: Optional[RegionGroupType] = None,
        indices: Optional[Sequence[int]] = None,
        filter: Optional[Callable[[CQObject], bool]] = None,
    ):
        selection = Selection(selector, tag, type, indices, filter)
        return self.select(selection, "Wire")

    def vertices(
        self,
        selector: Union[cq.Selector, str, None] = None,
        tag: Union[str, None] = None,
        type: Optional[RegionGroupType] = None,
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
        is_initialization = self._ctx.region_group_types is None

        if is_initialization:
            # Calculate workplane hash first
            workplane_hash = CQUtils.get_shape_checksum(
                self._ctx.initial_workplane.val()
            )

            # Try to load region groups from cache
            cached_region_groups = CacheService.load_regions_group_types(
                workplane_hash
            )

            if cached_region_groups is not None:
                # Use cached region groups as starting point
                self._ctx.region_group_types = cached_region_groups
            else:
                # No cache, start with empty dict
                self._ctx.region_group_types = {}

        # Compute region groups (will update self._ctx.region_groups with any new faces)
        region_groups = CQLinq.groupByRegionTypes(
            self._ctx.initial_workplane,
            self._ctx.is_2d,
            self._ctx.tol,
            check_splits=self._ctx.is_split,
            current_region_groups=self._ctx.region_group_types,
        )

        # Save to cache if this was the first computation
        if is_initialization:
            workplane_hash = CQUtils.get_shape_checksum(
                self._ctx.initial_workplane.val())
            CacheService.cache_regions_group_types(
                workplane_hash, self._ctx.region_group_types)

        return region_groups

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

    def show(
        self,
        type: ShowType = "cq",
        theme: Literal["light", "dark"] = "light",
        only_faces: bool = False,
        only_markers: bool = False,
        output_path: Optional[str] = None,
        open_in_browser: bool = False,
        only_surface: bool = False,
        only_volume: bool = False,
    ):
        if type == "mesh":
            assert self._mesh is not None, "Mesh is not generated yet."
            visualize_mesh(
                self._mesh,
                only_markers=only_markers,
                output_path=output_path,
                open_in_browser=open_in_browser,
                only_surface=only_surface,
                only_volume=only_volume
            )
        elif type == "plot":
            plot_cq(self._workplane)
        elif type == "cq":
            from jupyter_cadquery import show

            try:
                if only_faces:
                    root_assembly = cq.Assembly()
                    for i, face in enumerate(self._workplane.faces().vals()):
                        root_assembly.add(cq.Workplane(face),
                                          name=f"face/{i+1}")
                    show(root_assembly, theme=theme)
                else:
                    show(self._workplane, theme=theme)
            except:
                logger.warn(
                    "inadequate CQ geometry, trying to display in GMSH ...")

        else:
            raise NotImplementedError(f"Unknown show type {type}")
        return self

    def _addEntityGroup(
        self,
        group_name: str,
        entities: OrderedSet[Entity],
    ): ...
