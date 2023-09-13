import gmsh
from dataclasses import dataclass
from meshql.entity import Entity
from meshql.transaction import MultiEntityTransaction
from meshql.utils.types import OrderedSet


@dataclass(eq=False)
class SetPhysicalGroup(MultiEntityTransaction):
    entities: OrderedSet[Entity]
    "The entities that will be added to the physical group."

    name: str
    "The name of the physical group."

    def __post_init__(self):
        super().__post_init__()
        self.entity_type = self.entities.first.dim

    def before_gen(self):
        entity_tags = []
        for entity in self.entities:
            assert entity.type == self.entities.first.type, "all entities must be of the same type"
            entity.name = self.name
            entity_tags.append(entity.tag)

        physical_group_tag = gmsh.model.addPhysicalGroup(self.entities.first.dim, entity_tags)
        gmsh.model.set_physical_name(self.entity_type, physical_group_tag, self.name)