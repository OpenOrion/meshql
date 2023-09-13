
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence
import numpy.typing as npt
import numpy as np
from meshql.utils.types import NumpyFloat

class ElementType(Enum):
    LINE = 1
    TRIANGLE = 2
    QUADRILATERAL = 3
    TETRAHEDRON = 4
    HEXAHEDRON = 5
    PRISM = 6
    PYRAMID = 7
    POINT = 15


@dataclass
class Mesh:
    dim: int
    elements: Sequence[npt.NDArray[np.uint16]]
    element_types: Sequence[ElementType]
    points: npt.NDArray[NumpyFloat]
    markers: dict[str, Sequence[npt.NDArray[np.uint16]]]
    marker_types: dict[str, Sequence[ElementType]]
    target_points: dict[str, dict[np.uint16, str]] = field(default_factory=dict)
