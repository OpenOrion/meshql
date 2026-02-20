import gmsh
from dataclasses import dataclass
from typing import Literal
from meshql.gmsh.entity import Entity
from meshql.core.transaction import MultiEntityTransaction
from meshql.utils.types import OrderedSet
from pydantic import computed_field


class SetPhysicalGroup(MultiEntityTransaction):
    class_name: Literal["SetPhysicalGroup"] = "SetPhysicalGroup"
    entities: OrderedSet[Entity]
    "The entities that will be added to the physical group."

    name: str
    "The name of the physical group."

    @computed_field
    def entity_type(self) -> int:
        return self.entities.first.dim

    def __init__(self, **data):
        super().__init__(**data)
        # assert len(self.entities) > 0, "At least one entity must be provided"
        self.ref_id = self.name

    def before_gen(self):
        entity_tags = []
        for entity in self.entities:
            assert (
                entity.type == self.entities.first.type
            ), "all entities must be of the same type"
            entity.name = self.name
            entity_tags.append(entity.tag)
        physical_group_tag = gmsh.model.addPhysicalGroup(
            self.entities.first.dim, entity_tags
        )
        gmsh.model.set_physical_name(
            self.entity_type, physical_group_tag, self.name)
