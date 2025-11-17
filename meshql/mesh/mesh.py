from enum import Enum


class GmshElementType(Enum):
    LINE = 1
    TRIANGLE = 2
    QUADRILATERAL = 3
    TETRAHEDRON = 4
    HEXAHEDRON = 5
    PRISM = 6
    PYRAMID = 7
    POINT = 15


# Complete VTK to GMSH element type mapping
VTK_TO_GMSH_ELEMENT_TYPE = {
    1: 15,   # VTK_VERTEX -> POINT
    3: 1,    # VTK_LINE -> LINE
    5: 2,    # VTK_TRIANGLE -> TRIANGLE
    9: 3,    # VTK_QUAD -> QUADRILATERAL
    10: 4,   # VTK_TETRA -> TETRAHEDRON
    12: 5,   # VTK_HEXAHEDRON -> HEXAHEDRON
    13: 6,   # VTK_WEDGE -> PRISM
    14: 7,   # VTK_PYRAMID -> PYRAMID
}
