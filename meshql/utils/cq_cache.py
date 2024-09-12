import tempfile
from cachetools import LRUCache
import os
from pathlib import Path
import hashlib
from cachetools import LRUCache
import cadquery as cq
from cadquery.cq import CQObject
from typing import Sequence, Union
import numpy as np
from OCP.BRepTools import BRepTools
from OCP.BRep import BRep_Builder
from OCP.TopoDS import TopoDS_Shape
import numpy as np
from OCP.TopoDS import TopoDS_Shape, TopoDS_Vertex, TopoDS
from OCP.BRepTools import BRepTools
from OCP.BRep import BRep_Builder, BRep_Tool
import cadquery as cq
import pickle as pkl

TEMPDIR_PATH = tempfile.gettempdir()
CACHE_DIR_NAME = "meshql_geom_cache"
CACHE_DIR_PATH = os.path.join(TEMPDIR_PATH, CACHE_DIR_NAME)
ShapeChecksum = str

class CQCache:
    group_cache: LRUCache = LRUCache(maxsize=1000)
    checksum_cache: dict[ShapeChecksum, str] = {}

    @staticmethod
    def import_brep(file_path: str):
        """
        Import a boundary representation model
        Returns a TopoDS_Shape object
        """
        builder = BRep_Builder()
        shape = TopoDS_Shape()
        return_code = BRepTools.Read_s(shape, file_path, builder)
        if return_code is False:
            raise ValueError("Import failed, check file name")
        return cq.Compound(shape)

    @staticmethod
    def get_cache_exists(obj: Union[Sequence[CQObject], CQObject]):
        cache_file_name = CQCache.get_file_name(obj)
        return os.path.isfile(cache_file_name)

    @staticmethod
    def vertex_to_Tuple(vertex: TopoDS_Vertex):
        geom_point = BRep_Tool.Pnt_s(vertex)
        return (geom_point.X(), geom_point.Y(), geom_point.Z())

    @staticmethod
    def load_cache():
        group_type_cache_path = Path(CACHE_DIR_PATH) / "group_type_cache.pkl"
        if group_type_cache_path.exists():
            with open(group_type_cache_path, "rb") as f:
                CQCache.group_cache = pkl.load(f)

    @staticmethod
    def save_cache():
        group_type_cache_path = Path(CACHE_DIR_PATH) / "group_type_cache.pkl"
        with open(group_type_cache_path, "wb") as f:
            pkl.dump(CQCache.group_cache, f)



    @staticmethod
    def get_part_checksum(shape: Union[cq.Shape, cq.Workplane], precision=3):
        shape = shape if isinstance(shape, cq.Shape) else shape.val()
        if shape in CQCache.checksum_cache:
            return CQCache.checksum_cache[shape]
        vertices = np.array(
            [
                CQCache.vertex_to_Tuple(TopoDS.Vertex_s(v))
                for v in shape._entities("Vertex")
            ]
        )

        rounded_vertices = np.round(vertices, precision)
        rounded_vertices[rounded_vertices == -0] = 0

        sorted_indices = np.lexsort(rounded_vertices.T)
        sorted_vertices = rounded_vertices[sorted_indices]

        vertices_hash = hashlib.md5(sorted_vertices.tobytes()).digest()
        checksum = hashlib.md5(vertices_hash).hexdigest()
        CQCache.checksum_cache[shape] = checksum
        return checksum
    
    @staticmethod
    def get_file_name(shape: Union[Sequence[CQObject], CQObject]):
        # encode the hash as a filesystem safe string
        if isinstance(shape, cq.Shape):
            shape_id = CQCache.get_part_checksum(shape)
        else:
            cat_shape_id = ""
            for obj in shape:
                cat_shape_id += CQCache.get_part_checksum(obj)
            shape_id = hashlib.md5(cat_shape_id.encode()).hexdigest()
        return f"{CACHE_DIR_PATH}/{shape_id}.brep"

    @staticmethod
    def export_brep(shape: cq.Shape, file_path: str):
        cache_dir_path = Path(TEMPDIR_PATH) / CACHE_DIR_NAME
        if not cache_dir_path.exists():
            cache_dir_path.mkdir()
        file_path = cache_dir_path / file_path
        shape.exportBrep(str(file_path))

    @staticmethod
    def clear_cache():
        cache_dir_path = Path(TEMPDIR_PATH) / CACHE_DIR_NAME
        if cache_dir_path.exists():
            for file in cache_dir_path.iterdir():
                file.unlink()
