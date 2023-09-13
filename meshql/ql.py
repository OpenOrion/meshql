import gmsh
import numpy as np
import cadquery as cq
from cadquery.cq import CQObject
from typing import Callable, Iterable, Literal, Optional, Sequence, Union, cast
from cadquery.selectors import Selector
from meshql.entity import CQEntityContext, Entity
from meshql.preprocessing.split import split_workplane
from meshql.transaction import Transaction, TransactionContext
from meshql.transactions.algorithm import MeshAlgorithm2DType, MeshAlgorithm3DType, SetMeshAlgorithm2D, SetMeshAlgorithm3D
from meshql.transactions.boundary_layer import UnstructuredBoundaryLayer, UnstructuredBoundaryLayer2D, get_boundary_ratio
from meshql.transactions.physical_group import SetPhysicalGroup
from meshql.transactions.refinement import Recombine, Refine, SetMeshSize, SetSmoothing
from meshql.transactions.transfinite import SetTransfiniteEdge, SetTransfiniteFace, SetTransfiniteSolid, TransfiniteArrangementType, TransfiniteMeshType
from meshql.mesh.exporters import export_to_su2
from meshql.utils.cq import CQ_TYPE_RANKING, CQ_TYPE_STR_MAPPING, CQExtensions, CQGroupTypeString, CQLinq, CQType
from meshql.utils.types import OrderedSet
from meshql.visualizer import visualize_mesh
from jupyter_cadquery import show

class GeometryQL:
    _workplane: cq.Workplane
    _initial_workplane: cq.Workplane
    def __init__(self) -> None:
        self._initial_workplane = self._workplane = None # type: ignore
        self._ctx = TransactionContext()
        self.is_structured = False
        self._transfinite_edge_groups = list[set[cq.Edge]]()
    def __enter__(self):
        gmsh.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        gmsh.finalize()

    def end(self, num: Optional[int] = None):
        if num is None:
            self._workplane = self._initial_workplane
        else:
            self._workplane = self._workplane.end(num)
        return self

    def load(self, target: Union[cq.Workplane, str, Iterable[CQObject]], splits: Optional[Callable[[cq.Workplane], Sequence[cq.Face]]] = None, use_cache: bool = False):
        assert self._workplane is None, "Workplane is already loaded."

        workplane = self._initial_workplane = CQExtensions.import_workplane(target)
        is_2d = CQExtensions.get_dimension(workplane) == 2

        # extrudes 2D shapes to 3D
        if is_2d:
            workplane = workplane.extrude(-1)

        if splits:
            workplane = split_workplane(workplane, splits(workplane), use_cache)


        if is_2d:
            # fuses top faces to appear as one Compound in GMSH
            faces = cast(Sequence[cq.Face], workplane.faces(">Z").vals())
            fused_face = CQExtensions.fuse_shapes(faces)
            self._initial_workplane = cq.Workplane(fused_face)
        else:
            self._initial_workplane = workplane

        self._type_groups = CQLinq.groupByTypes(self._initial_workplane, exclude_split=is_2d)


        self._workplane = self._initial_workplane
        topods = self._workplane.toOCC()
        gmsh.model.occ.importShapesNativePointer(topods._address())
        gmsh.model.occ.synchronize()

        self._entity_ctx = CQEntityContext(self._workplane)

        self._tag_workplane()

        return self    
    

    def _tag_workplane(self):
        "Tag all gmsh entity tags to workplane"
        for cq_type, registry in self._entity_ctx.entity_registries.items():
            for occ_obj in registry.keys():
                tag = f"{cq_type}/{registry[occ_obj].tag}"
                self._workplane.newObject([occ_obj]).tag(tag)

    def solids(self, selector: Union[Selector, str, None] = None, tag: Union[str, None] = None, type: Optional[CQGroupTypeString] = None, indices: Optional[Sequence[int]] = None):
        obj_type = type and self._type_groups[type]
        selector = CQExtensions.get_selector(selector, obj_type, indices)
        self._workplane = self._workplane.solids(selector, tag)
        return self

    def faces(self, selector: Union[Selector, str, None] = None, tag: Union[str, None] = None, type: Optional[CQGroupTypeString] = None, indices: Optional[Sequence[int]] = None):
        obj_type = type and self._type_groups[type]
        selector = CQExtensions.get_selector(selector, obj_type, indices)
        self._workplane = self._workplane.faces(selector, tag)
        return self
    
    def edges(self, selector: Union[Selector, str, None] = None, tag: Union[str, None] = None, type: Optional[CQGroupTypeString] = None, indices: Optional[Sequence[int]] = None):
        obj_type = type and self._type_groups[type]
        selector = CQExtensions.get_selector(selector, obj_type, indices)
        self._workplane = self._workplane.edges(selector, tag)
        return self

    def wires(self, selector: Union[Selector, str, None] = None, tag: Union[str, None] = None, type: Optional[CQGroupTypeString] = None, indices: Optional[Sequence[int]] = None):
        obj_type = type and self._type_groups[type]
        selector = CQExtensions.get_selector(selector, obj_type, indices)
        self._workplane = self._workplane.wires(selector, tag)
        return self

    def vertices(self, selector: Selector | str | None = None, tag: str | None = None, type: Optional[CQGroupTypeString] = None, indices: Optional[Sequence[int]] = None):
        obj_type = type and self._type_groups[type]
        selector = CQExtensions.get_selector(selector, obj_type, indices)
        self._workplane = self._workplane.vertices(selector, tag)
        return self

    def vals(self):
        return self._entity_ctx.select_many(self._workplane)

    def val(self):
        return self._entity_ctx.select(self._workplane.val())


    def tag(self, names: Union[str, Sequence[str]]):
        if isinstance(names, str):
            self._workplane.tag(names)
        else:
            for i, cq_obj in enumerate(self._workplane.vals()):
                self._workplane.newObject([cq_obj]).tag(names[i])
        return self

    def fromTagged(self, tags: Union[str, Iterable[str]], resolve_type: Optional[CQType] = None, invert: bool = True):        
        if isinstance(tags, str) and resolve_type is None:
            self._workplane = self._workplane._getTagged(tags)
        else:
            tagged_objs = list(CQLinq.select_tagged(self._workplane, tags, resolve_type))
            tagged_cq_type = CQ_TYPE_STR_MAPPING[type(tagged_objs[0])]
            workplane_objs = CQLinq.select(self._workplane, tagged_cq_type)
            filtered_objs = CQLinq.filter(workplane_objs, tagged_objs, invert)
            self._workplane = self._workplane.newObject(filtered_objs)
        return self

    def addPhysicalGroup(self, group: Union[str, Sequence[str]]):
        if  isinstance(group, str):
            set_physical_group = SetPhysicalGroup(self.vals(), group)
            self._ctx.add_transaction(set_physical_group)
        else:
            objs = list(self.vals())
            group_entities: dict[str, OrderedSet[Entity]] = {}

            for i, group_name in enumerate(group):
                new_group_entity = objs[i]
                if group_name not in group_entities:
                    group_entities[group_name] = OrderedSet()
                group_entities[group_name].add(new_group_entity)
            
            for group_name, group_objs in group_entities.items():
                set_physical_group = SetPhysicalGroup(group_objs, group_name)
                self._ctx.add_transaction(set_physical_group)

        return self

    def recombine(self, angle: float = 45):
        faces = self._entity_ctx.select_many(self._workplane, "face")
        recombines = [Recombine(face, angle) for face in faces]
        self._ctx.add_transactions(recombines)
        return self

    def setMeshSize(self, size: Union[float, Callable[[float,float,float], float]]):
        points = self._entity_ctx.select_many(self._workplane, "vertex")
        set_size = SetMeshSize(points, size)
        self._ctx.add_transaction(set_size)
        return self

    def setMeshAlgorithm(self, type: MeshAlgorithm2DType, per_face: bool = False):
        faces = self._entity_ctx.select_many(self._workplane, "face")
        set_algorithms = [SetMeshAlgorithm2D(face, type, per_face) for face in faces]
        self._ctx.add_transactions(set_algorithms)
        return self

    def setMeshAlgorithm3D(self, type: MeshAlgorithm3DType):
        set_algorithm3D = SetMeshAlgorithm3D(type)
        self._ctx.add_transaction(set_algorithm3D)
        return self

    def smooth(self, num_smooths = 1):
        faces = self._entity_ctx.select_many(self._workplane)
        set_smoothings = [SetSmoothing(face, num_smooths) for face in faces]
        self._ctx.add_transactions(set_smoothings)
        return self

    def refine(self, num_refines = 1):
        refine = Refine(num_refines)
        self._ctx.add_transaction(refine)
        return self

    def setTransfiniteEdge(self, num_nodes: Optional[Union[Sequence[int], int]] = None, mesh_type: Optional[Union[TransfiniteMeshType, Sequence[TransfiniteMeshType]]] = None, coef: Optional[Union[float, Sequence[float]]] = None):
        edge_batch = self._entity_ctx.select_batch(self._workplane, "face", "edge")
        for edges in edge_batch:
            for i, edge in enumerate(edges):
                transaction = cast(SetTransfiniteEdge, self._ctx.get_transaction(SetTransfiniteEdge, edge))
                if transaction is not None:
                    if num_nodes is not None:
                        transaction.num_elems = num_nodes if isinstance(num_nodes, int) else num_nodes[i]
                    if mesh_type is not None:
                        transaction.mesh_type = mesh_type if isinstance(mesh_type, str) else mesh_type[i]
                    if coef is not None:
                        transaction.coef = coef if isinstance(coef, (int, float)) else coef[i]
                else:
                    assert num_nodes is not None, "num_nodes must be specified"
                    mesh_type = mesh_type or "Progression"
                    coef = coef or 1.0
                    set_transfinite_edge = SetTransfiniteEdge(
                        edge, 
                        num_nodes if isinstance(num_nodes, int) else num_nodes[i], 
                        mesh_type if isinstance(mesh_type, str) else mesh_type[i], 
                        coef if isinstance(coef, (int, float)) else coef[i]
                    )
                    self._ctx.add_transaction(set_transfinite_edge)

        return self

    def setTransfiniteFace(self, arrangement: TransfiniteArrangementType = "Left"):
        self.is_structured = True    
        cq_face_batch = CQLinq.select_batch(self._workplane, "solid", "face")
        for i, cq_faces in enumerate(cq_face_batch):
            faces = self._entity_ctx.select_many(cq_faces)
            set_transfinite_faces = [SetTransfiniteFace(face, arrangement) for face in faces]
            self._ctx.add_transactions(set_transfinite_faces)

        return self

    def setTransfiniteSolid(self):
        self.is_structured = True    
        solids = self._entity_ctx.select_many(self._workplane, "solid")
        set_transfinite_solids = [SetTransfiniteSolid(solid) for solid in solids]
        self._ctx.add_transactions(set_transfinite_solids)
        return self

    def _getTransfiniteEdgeGroups(self, cq_faces: Sequence[cq.Face]):
        transfinite_edge_groups: list[set[cq.Edge]] = []
        for cq_face in cq_faces:
            sorted_edges = CQLinq.sort(cq_face.Edges())
            for i, path in enumerate(sorted_edges):
                cq_edge = path.edge
                parllel_edge_index = (i+2 if i+2 < len(sorted_edges) else (i+2) - len(sorted_edges))
                cq_parllel_edge = sorted_edges[parllel_edge_index].edge
                found_group: Optional[set] = None
                for i, group in enumerate(transfinite_edge_groups):
                    if not found_group:
                        if cq_edge in group:
                            group.add(cq_parllel_edge)
                            found_group = group
                        elif cq_parllel_edge in group:
                            group.add(cq_edge)
                            found_group = group
                    else: 
                        if cq_edge in group or cq_parllel_edge in group:
                            found_group.update(group)
                            transfinite_edge_groups.remove(group)

                if found_group is None:
                    transfinite_edge_groups.append(set([path.edge, cq_parllel_edge]))
        return transfinite_edge_groups

    def _setTransfiniteFaceAuto(
        self, 
        cq_faces: Sequence[cq.Face], 
        max_nodes: int, 
        min_nodes: int = 1,
        arrangement: TransfiniteArrangementType = "Left"
    ):
        self.is_structured = True    
        for cq_face in cq_faces:
            face = self._entity_ctx.select(cq_face)
            set_transfinite_face = SetTransfiniteFace(face, arrangement)
            self._ctx.add_transaction(set_transfinite_face)
        self._transfinite_edge_groups = self._getTransfiniteEdgeGroups(cq_faces)
        
        for transfinite_group in self._transfinite_edge_groups:
            total_length = sum([cq_edge.Length() for cq_edge in transfinite_group])
            group_max_num_nodes = 0
            for cq_edge in transfinite_group:
                edge_num_nodes = int(np.ceil((cq_edge.Length()/total_length)*max_nodes))
                if edge_num_nodes < min_nodes:
                    edge_num_nodes = min_nodes
                if edge_num_nodes > group_max_num_nodes:
                    group_max_num_nodes = edge_num_nodes
            
            assert group_max_num_nodes > 0, "group_max_num_nodes must be greater than 0, make num_nodes higher"
            group_edges = self._entity_ctx.select_many(transfinite_group)
            set_transfinite_edges = [SetTransfiniteEdge(edge, group_max_num_nodes) for edge in group_edges]
            self._ctx.add_transactions(set_transfinite_edges)


    def addTransaction(self, toTransaction: Callable[["GeometryQL"], Transaction]):
        self._ctx.add_transaction(toTransaction(self))
        return self

    def setTransfiniteAuto(
        self,
        max_nodes: int,
        min_nodes: int = 1,
        auto_recombine: bool = True
    ):
        self.is_structured = True
        if CQExtensions.get_dimension(self._workplane) == 2:
            cq_faces = cast(Sequence[cq.Face], list(CQLinq.select(self._workplane, "face")))
            self._setTransfiniteFaceAuto(cq_faces, max_nodes, min_nodes)

        else:
            for cq_solid in cast(Sequence[cq.Solid], CQLinq.select(self._workplane, "solid")):
                solid = self._entity_ctx.select(cq_solid)
                set_transfinite_solid = SetTransfiniteSolid(solid)
                self._ctx.add_transaction(set_transfinite_solid)
            cq_faces = cast(Sequence[cq.Face], list(CQLinq.select(self._workplane, "face")))
            self._setTransfiniteFaceAuto(cq_faces, max_nodes, min_nodes)

        # transfinite_auto = SetTransfiniteAuto()
        # self._ctx.add_transaction(transfinite_auto)


        if auto_recombine:
            self.recombine()

        return self

    def _addStructuredBoundaryLayer(
            self, 
            cq_objs: Sequence[CQObject], 
            size: Optional[float] = None,
            ratio: Optional[float] = None,
        ):
        assert self.is_structured, "Structured boundary layer can only be applied after setTransfiniteAuto"
        assert (size is None) != (ratio is None), "Either size or ratio must be specified, not both"

        boundary_vertices =  list(CQLinq.select(cq_objs, "vertex"))

        for (cq_edge, edge) in self._entity_ctx.entity_registries["edge"].items():
            transaction = cast(SetTransfiniteEdge, self._ctx.get_transaction(SetTransfiniteEdge, edge))
            assert edge.type == "edge", "StructuredBoundaryLayer only accepts edges"
            if size:
                edge_ratio = get_boundary_ratio(cq_edge.Length(), size, transaction.num_elems) # type: ignore
            elif ratio:
                edge_ratio = ratio
            else:
                raise ValueError("Either size or ratio must be specified, not both")
            cq_curr_edge_vertices =  cq_edge.Vertices() # type: ignore
            if cq_curr_edge_vertices[0] in boundary_vertices and cq_curr_edge_vertices[-1] not in boundary_vertices:
                transaction.coef = edge_ratio

            elif cq_curr_edge_vertices[-1] in boundary_vertices and cq_curr_edge_vertices[0] not in boundary_vertices:
                transaction.coef = -edge_ratio

    def addBoundaryLayer(self, size: float, ratio: Optional[float] = None, num_layers: Optional[int] = None, auto_recombine: bool = True):
        if self.is_structured:
            self._addStructuredBoundaryLayer(self._workplane.vals(), size, ratio)
        else:
            ratio = ratio or 1.0
            assert num_layers is not None and size is not None and ratio is not None, "num_layers, hwall_n and ratio must be specified for unstructured boundary layer"
            if CQ_TYPE_RANKING[type(self._workplane.val())] < CQ_TYPE_RANKING[cq.Face]:
                boundary_layer = UnstructuredBoundaryLayer2D(self.vals(), ratio, size, num_layers)
            else:
                boundary_layer = UnstructuredBoundaryLayer(self.vals(), ratio, size, num_layers)
                if auto_recombine:
                    self.recombine()
            self._ctx.add_transaction(boundary_layer)
        return self

    def generate(self, dim: int = 3):
        self._ctx.generate(dim)
        return self

    def write(self, filename: str, dim: int = 3):
        if self._ctx.mesh is None:
            self.generate(dim)
        assert self._ctx.mesh is not None
        if filename.endswith(".su2"):
            export_to_su2(self._ctx.mesh, filename)
        else:
            gmsh.write(filename)
        return self

    def showTransfiniteGroup(self, group_index: int):
        assert self.is_structured, "Structured boundary layer can only be applied after setTransfiniteAuto"
        assert group_index < len(self._transfinite_edge_groups), f"Group index {group_index} is out of range"
        group = self._transfinite_edge_groups[group_index]
        show(self._workplane.newObject(group), theme="dark")
        return self

    def show(self, type: Literal["gmsh", "mesh", "cq", "plot"] = "cq", only_markers: bool = False):
        if type == "gmsh":
            gmsh.fltk.run()
        elif type == "mesh":
            assert self._ctx.mesh is not None, "Mesh is not generated yet."
            visualize_mesh(self._ctx.mesh, only_markers=only_markers)
        elif type == "plot":
            CQExtensions.plot_cq(self._workplane, ctx=self._entity_ctx)
        elif type == "cq":
            show(self._workplane, theme="dark")
        else:
            raise NotImplementedError(f"Unknown show type {type}")
        return self


    def close(self):
        gmsh.finalize()
