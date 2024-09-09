from typing import Callable, Iterable, Optional, Union, cast
import gmsh
import cadquery as cq
from meshql.boundary_condition import BoundaryCondition
from meshql.entity import CQEntityContext, Entity
from meshql.mesh.exporters import export_to_su2
from meshql.preprocessing.split import Split
from meshql.ql import GeometryQL, ShowType
from cadquery.cq import CQObject


class SnappyHexMeshGeometryQL(GeometryQL):
    def __init__(self) -> None:
        super().__init__()

    @property
    def mesh(self): ...

    def load(
        self,
        wind_tunnel: Optional[Union[str, CQObject]] = None,
        subject: Optional[Union[str, CQObject]] = None,
    ):
        return self

    def generate(self, dim: int = 3):
        return self

    def write(self, filename: str, dim: int = 3):
        return self
