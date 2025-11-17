import cadquery as cq
from cadquery.cq import CQObject
from dataclasses import dataclass
from typing import Iterable, Optional, OrderedDict, Sequence, Union, cast
from meshql.utils.cq import OCC_TYPE_STR_MAPPING, CQType, CQ_TYPE_STR_MAPPING
from meshql.utils.cq_linq import CQLinq
from meshql.utils.types import OrderedSet
from OCP.TopoDS import TopoDS_Shape, TopoDS_Face
from cadquery.occ_impl.shapes import downcast, HASH_CODE_MAX

ENTITY_DIM_MAPPING: dict[CQType, int] = {
    "Vertex": 0,
    "Edge": 1,
    "Face": 2,
    "Solid": 3,
}


@dataclass
class Entity:
    type: CQType
    "dimension type of the entity."

    tag: int = -1
    "tag of the entity."

    name: Optional[str] = None
    "name of the entity."

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Entity):
            return self.type == __value.type and self.tag == __value.tag
        return False

    @property
    def dim(self):
        if self.type not in ENTITY_DIM_MAPPING:
            raise ValueError(
                f"Entity type {self.type} not supported, only {ENTITY_DIM_MAPPING.keys()}"
            )

        return ENTITY_DIM_MAPPING[self.type]

    def __hash__(self) -> int:
        return hash((self.type, self.tag))


class CQEntityContext:
    "Maps OCC objects to gmsh entity tags"

    def __init__(self, workplane: cq.Workplane, level: CQType = "Edge") -> None:
        self.dimension = 3 if len(workplane.solids().vals()) else 2

        self.entity_registries: dict[CQType, OrderedDict[int, Entity]] = {
            "Compound": OrderedDict[int, Entity](),
            "Solid": OrderedDict[int, Entity](),
            "Shell": OrderedDict[int, Entity](),
            "Face": OrderedDict[int, Entity](),
            "Wire": OrderedDict[int, Entity](),
            "Edge": OrderedDict[int, Entity](),
            "Vertex": OrderedDict[int, Entity](),
        }
        self.shape_lookup: dict[int, TopoDS_Shape] = {}

        if self.dimension == 3:
            self._init_3d_objs(workplane.solids().vals(), level)
        else:
            self._init_2d_objs(workplane.faces().vals(), level)

    def add(self, shape: TopoDS_Shape):
        casted_shape = downcast(shape)
        entity_type = OCC_TYPE_STR_MAPPING[type(casted_shape)]
        obj_hash = casted_shape.HashCode(HASH_CODE_MAX)
        registry = self.entity_registries[entity_type]
        if obj_hash not in registry:
            tag = len(registry) + 1
            registry[obj_hash] = Entity(entity_type, tag)
            self.shape_lookup[obj_hash] = casted_shape

    def select(self, obj: CQObject):
        entity_type = CQ_TYPE_STR_MAPPING[type(obj)]
        registry = self.entity_registries[entity_type]
        return registry[obj.wrapped.HashCode(HASH_CODE_MAX)]

    def select_many(
        self,
        target: Union[cq.Workplane, Iterable[CQObject]],
        type: Optional[CQType] = None,
    ):
        entities = OrderedSet[Entity]()
        objs = target.vals() if isinstance(target, cq.Workplane) else target

        selected_objs = objs if type is None else CQLinq.select(objs, type)
        for obj in selected_objs:
            try:
                selected_entity = self.select(obj)
                entities.add(selected_entity)
            except:
                ...

        return entities

    def select_batch(
        self,
        target: Union[cq.Workplane, Iterable[CQObject]],
        parent_type: CQType,
        child_type: CQType,
    ):
        objs = target.vals() if isinstance(target, cq.Workplane) else target
        selected_batches = CQLinq.select_batch(objs, parent_type, child_type)
        for selected_batch in selected_batches:
            yield self.select_many(selected_batch)

    def _init_3d_objs(self, solids: Sequence[cq.Solid], level: CQType):
        for solid in solids:
            if level != "Solid":
                for shell in CQLinq.select_occ(solid.wrapped, "Shell"):
                    if level != "Shell":
                        for face in CQLinq.select_occ(shell, "Face"):
                            if level != "Face":
                                self._init_2d_objs(CQLinq.select_occ(face, "Wire"), level)
                            self.add(face)
                    self.add(shell)
            self.add(solid.wrapped)

    def _init_2d_objs(self, faces: Iterable[Union[TopoDS_Shape, CQObject]], level: CQType):
        for face in faces:
            # assert isinstance(face, (cq.Face, TopoDS_Shape)), "Only faces are supported in 2D mode"
            occ_face = face.wrapped if isinstance(face, cq.Shape) else face
            if level != "Face":
                for wire in CQLinq.select_occ(
                    face.wrapped if isinstance(face, cq.Face) else face, "Wire"
                ):
                    if level != "Wire":
                        for edge in CQLinq.select_occ(wire, "Edge"):
                            if level != "Edge":
                                for vertex in CQLinq.select_occ(edge, "Vertex"):
                                    self.add(vertex)
                            self.add(edge)
                    self.add(wire)
            self.add(occ_face)
