import gmsh
from enum import Enum
from dataclasses import dataclass
from typing import Literal, Optional
from meshql.gmsh.entity import Entity
from meshql.gmsh.transaction import GmshTransaction


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

MESH_SUBDIVISION_ALGORITHM_MAPPING = {
    "None": 0,
    "AllQuadrangles": 1,
    "AllHexahedra": 2,
    "Barycentric": 3,
}

MeshAlgorithm2DType = Literal[
    "MeshAdapt",
    "Automatic",
    "InitialMeshOnly",
    "Delaunay",
    "FrontalDelaunay",
    "BAMG",
    "FrontalDelaunayQuads",
    "PackingOfParallelograms",
    "QuasiStructuredQuad",
]

MeshAlgorithm3DType = Literal[
    "Delaunay", "InitialMeshOnly", "Frontal", "MMG3D", "RTree", "HXT"
]

MeshSubdivisionType = Literal[
    "None",
    "AllQuadrangles",
    "AllHexahedra",
    "Barycentric",
]


@dataclass(eq=False)
class SetMeshAlgorithm2D(GmshTransaction):
    type: MeshAlgorithm2DType
    "algorithm to use"

    entity: Optional[Entity] = None
    "Entity to set algorithm for"

    def before_gen(self):
        if self.entity:
            assert self.entity.type == "face", "Can only set per face for edges"
            gmsh.model.mesh.setAlgorithm(
                self.entity.dim, self.entity.tag, MESH_ALGORITHM_2D_MAPPING[self.type]
            )
        else:
            algo_val = MESH_ALGORITHM_2D_MAPPING[self.type]
            gmsh.option.setNumber("Mesh.Algorithm", algo_val)


@dataclass(eq=False)
class SetMeshAlgorithm3D(GmshTransaction):
    type: MeshAlgorithm3DType
    "algorithm to use"

    def before_gen(self):
        algo_val = MESH_ALGORITHM_3D_MAPPING[self.type]
        gmsh.option.setNumber("Mesh.Algorithm3D", algo_val)


@dataclass(eq=False)
class SetSubdivisionAlgorithm(GmshTransaction):
    type: MeshSubdivisionType
    "algorithm to use"

    def before_gen(self):
        algo_val = MESH_SUBDIVISION_ALGORITHM_MAPPING[self.type]
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", algo_val)
