import gmsh
from enum import Enum
from dataclasses import dataclass
from typing import Literal
from meshql.entity import Entity
from meshql.transaction import SingleEntityTransaction, Transaction


MESH_ALGORITHM_2D_MAPPING = {
    "MeshAdapt": 1,
    "Automatic": 2,
    "InitialMeshOnly": 3,
    "Delaunay": 5,
    "FrontalDelaunay": 6,
    "BAMG": 7,
    "FrontalDelaunayQuads": 8,
    "PackingOfParallelograms": 9,
    "QuasiStructuredQuad": 11,
}

MESH_ALGORITHM_3D_MAPPING = {
   "Delaunay": 1,
   "InitialMeshOnly": 3,
   "Frontal": 4,
   "MMG3D": 7,
   "RTree": 9,
   "HXT": 10,
}

MeshAlgorithm2DType =  Literal[
    "MeshAdapt",
    "Automatic",
    "InitialMeshOnly",
    "Delaunay",
    "FrontalDelaunay",
    "BAMG",
    "FrontalDelaunayQuads",
    "PackingOfParallelograms",
    "QuasiStructuredQuad"
]

MeshAlgorithm3DType =  Literal[
    "Delaunay",
    "InitialMeshOnly",
    "Frontal",
    "MMG3D", 
    "RTree", 
    "HXT"
]

@dataclass(eq=False)
class SetMeshAlgorithm2D(SingleEntityTransaction):
    entity: Entity
    "Entity to set algorithm for"

    type: MeshAlgorithm2DType
    "algorithm to use"

    per_face: bool = False
    "if True, set algorithm per face, otherwise for whole system"

    def before_gen(self):
        if self.per_face:
            assert self.entity.type == "face", "Can only set per face for edges"
            gmsh.model.mesh.setAlgorithm(self.entity.dim, self.entity.tag, MESH_ALGORITHM_2D_MAPPING[self.type])
        else:
            algo_val = MESH_ALGORITHM_2D_MAPPING[self.type]
            gmsh.option.setNumber("Mesh.Algorithm", algo_val)

@dataclass(eq=False)
class SetMeshAlgorithm3D(Transaction):
    type: MeshAlgorithm3DType
    "algorithm to use"

    def before_gen(self):
        algo_val = MESH_ALGORITHM_3D_MAPPING[self.type]
        gmsh.option.setNumber("Mesh.Algorithm3D", algo_val)

