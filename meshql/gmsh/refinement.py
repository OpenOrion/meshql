import gmsh
from typing import Callable, Literal, Union
from dataclasses import dataclass
from meshql.gmsh.entity import Entity
from meshql.core.transaction import (
    Transaction,
    SingleEntityTransaction,
    MultiEntityTransaction,
)
from meshql.utils.types import OrderedSet


class Recombine(SingleEntityTransaction):
    class_name: Literal["Recombine"] = "Recombine"
    entity: Entity
    "Entity to recombine for"

    angle: float = 45
    "Angle to recombine with"

    def before_gen(self):
        gmsh.model.mesh.setRecombine(
            self.entity.dim, self.entity.tag, self.angle)


class SetSmoothing(SingleEntityTransaction):
    class_name: Literal["SetSmoothing"] = "SetSmoothing"
    entity: Entity
    "Entity to smooth for"

    num_smooths: int = 1
    "Number of times to smooth the mesh"

    def after_gen(self):
        gmsh.model.mesh.setSmoothing(
            self.entity.dim, self.entity.tag, self.num_smooths)


class Refine(Transaction):
    class_name: Literal["Refine"] = "Refine"
    num_refines: int = 1
    "Number of times to refine the mesh"

    def after_gen(self):
        for _ in range(self.num_refines):
            gmsh.model.mesh.refine()


class SetMeshSize(MultiEntityTransaction):
    class_name: Literal["SetMeshSize"] = "SetMeshSize"
    entities: OrderedSet[Entity]
    "Entities to set mesh sizes for"

    size: Union[float, Callable[[float, float, float], float]]
    "Size to set points"

    def before_gen(self):
        point_tags = [(entity.dim, entity.tag) for entity in self.entities]
        if isinstance(self.size, float):
            gmsh.model.mesh.setSize(point_tags, self.size)
        else:
            gmsh.model.mesh.setSizeCallback(
                # type: ignore
                lambda dim, tag, x, y, z, lc: lc if tag in point_tags else self.size(x, y, z))
