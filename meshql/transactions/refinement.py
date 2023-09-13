import gmsh
from typing import Callable, Union
from dataclasses import dataclass
from meshql.entity import Entity
from meshql.transaction import Transaction, SingleEntityTransaction, MultiEntityTransaction
from meshql.utils.types import OrderedSet

@dataclass(eq=False)
class Recombine(SingleEntityTransaction):
    entity: Entity
    "Entity to recombine for"

    angle: float = 45
    "Angle to recombine with"

    def before_gen(self):
        gmsh.model.mesh.setRecombine(self.entity.dim, self.entity.tag, self.angle)

@dataclass(eq=False)
class SetSmoothing(SingleEntityTransaction):
    entity: Entity
    "Entity to smooth for"
    
    num_smooths: int = 1
    "Number of times to smooth the mesh"

    def after_gen(self):
        gmsh.model.mesh.setSmoothing(self.entity.dim, self.entity.tag, self.num_smooths)


@dataclass(eq=False)
class Refine(Transaction):
    num_refines: int = 1
    "Number of times to refine the mesh"
    
    def after_gen(self):
        for _ in range(self.num_refines):
            gmsh.model.mesh.refine()

@dataclass(eq=False)
class SetMeshSize(MultiEntityTransaction):
    entities: OrderedSet[Entity]
    "Entities to set mesh sizes for"

    size: Union[float, Callable[[float, float, float], float]]
    "Size to set points"

    def before_gen(self):
        point_tags = [(entity.dim, entity.tag) for entity in self.entities]
        if isinstance(self.size, float):
            gmsh.model.mesh.setSize(point_tags, self.size)
        else:
            gmsh.model.mesh.setSizeCallback(lambda dim, tag, x, y, z, lc: lc if tag in point_tags else self.size(x, y, z)) # type: ignore
