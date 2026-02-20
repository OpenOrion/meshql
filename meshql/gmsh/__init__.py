from .algorithm import *
from .boundary_layer import *
from .physical_group import *
from .refinement import *
from .transfinite import *


GmshTransactions = Union[
    SetPhysicalGroup,
    UnstructuredBoundaryLayer,
    UnstructuredBoundaryLayer2D,
    SetTransfiniteEdge,
    SetTransfiniteFace,
    SetTransfiniteSolid,
    SetCompound,
    SetTransfiniteAuto,
    Recombine,
    SetSmoothing,
    Refine,
    SetMeshSize,
    SetMeshAlgorithm2D,
    SetMeshAlgorithm3D,
    SetSubdivisionAlgorithm,
]
