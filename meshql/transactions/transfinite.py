import gmsh
import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional, Sequence
from meshql.entity import Entity, ENTITY_DIM_MAPPING
from meshql.transaction import SingleEntityTransaction, MultiEntityTransaction, Transaction
from meshql.utils.types import OrderedSet

TransfiniteArrangementType = Literal["Left", "Right", "AlternateLeft", "AlternateRight"]
TransfiniteMeshType = Literal["Progression", "Bump", "Beta"]

def get_num_nodes_for_ratios(total_num_nodes: int, ratios: Sequence[float]):
    assert np.round(np.sum(ratios), 5) == 1, "Ratios must sum to 1"
    assert total_num_nodes > len(ratios), f"Number of nodes must be greater than number of ratios {len(ratios)}"
    allocated_nodes = []
    for ratio in ratios:
        num_nodes = int(np.round(ratio * total_num_nodes)) or 1
        allocated_nodes.append(num_nodes)

    # remove nodes uniformly from highest ratios
    descending_ratio_indexes = sorted(range(len(ratios)), key=lambda i: -ratios[i])
    num_allocated_nodes = sum(allocated_nodes)
    if num_allocated_nodes > total_num_nodes:
        total_node_diff = num_allocated_nodes - total_num_nodes
        for i in descending_ratio_indexes:
            if allocated_nodes[i] > 1:
                allocated_node_diff = int(np.ceil(total_node_diff*ratios[i]))
                allocated_nodes[i] -= allocated_node_diff
                num_allocated_nodes -= allocated_node_diff
            if num_allocated_nodes == total_num_nodes:
                break
    assert sum(allocated_nodes) == total_num_nodes, (
        f"Number of allocated nodes must equal num_nodes, {num_allocated_nodes} != {total_num_nodes}"
    )

    return allocated_nodes


@dataclass(eq=False)
class SetTransfiniteEdge(SingleEntityTransaction):
    entity: Entity
    "edge to be added to the boundary layer"

    num_elems: int
    "number of elems for edge"

    mesh_type: TransfiniteMeshType = "Progression"
    "mesh type for edge"

    coef: float = 1.0
    "coefficients for edge"

    def __post_init__(self):
        super().__post_init__()

    def before_gen(self):
        assert self.entity.type == "edge", "SetTransfiniteEdge only accepts edges"
        num_nodes = self.num_elems + 1
        gmsh.model.mesh.setTransfiniteCurve(self.entity.tag, num_nodes, self.mesh_type, self.coef)



@dataclass(eq=False)
class SetTransfiniteFace(SingleEntityTransaction):
    entity: Entity
    "face to apply field"

    arrangement: TransfiniteArrangementType = "Left"
    "arrangement of transfinite face"

    corners: Optional[OrderedSet[Entity]] = None
    "corner point tags for transfinite face"

    def before_gen(self):
        assert self.entity.type == "face", "SetTransfiniteFace only accepts faces"
        corner_tags = [corner.tag for corner in self.corners] if self.corners else []
        gmsh.model.mesh.setTransfiniteSurface(self.entity.tag, self.arrangement, corner_tags)

@dataclass(eq=False)
class SetTransfiniteSolid(SingleEntityTransaction):
    entity: Entity
    "face to apply field"

    corners: Optional[OrderedSet[Entity]] = None
    "corner point tags for transfinite face"

    def before_gen(self):
        assert self.entity.type == "solid", "SetTransfiniteSolid only accepts solids"
        corner_tags = [corner.tag for corner in self.corners] if self.corners else []
        gmsh.model.mesh.setTransfiniteVolume(self.entity.tag, corner_tags)



@dataclass(eq=False)
class SetCompound(MultiEntityTransaction):
    entities: OrderedSet[Entity]
    "face to apply field"

    def before_gen(self):
        entity_tags = [entity.tag for entity in self.entities]
        gmsh.model.mesh.setCompound(ENTITY_DIM_MAPPING["edge"], entity_tags)


@dataclass(eq=False)
class SetTransfiniteAuto(Transaction):

    def before_gen(self):
        gmsh.option.setNumber('Mesh.MeshSizeMin', 0.5)
        gmsh.option.setNumber('Mesh.MeshSizeMax', 0.5)
        gmsh.model.mesh.setTransfiniteAutomatic()
