import tempfile
import os
from pathlib import Path
import hashlib
import cadquery as cq
from cadquery.cq import CQObject
from typing import Sequence, Union
import numpy as np
from OCP.BRepTools import BRepTools
from OCP.BRep import BRep_Builder
from OCP.TopoDS import TopoDS_Shape
import numpy as np
from OCP.TopoDS import TopoDS_Shape
from OCP.BRepTools import BRepTools
from OCP.BRep import BRep_Builder
import cadquery as cq

from meshql.utils.cq import CQUtils

TEMPDIR_PATH = tempfile.gettempdir()
CACHE_DIR_NAME = "meshql_geom_cache"
CACHE_DIR_PATH = os.path.join(TEMPDIR_PATH, CACHE_DIR_NAME)

class CQCache:

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
    def get_file_name(shape: Union[Sequence[CQObject], CQObject]):
        # encode the hash as a filesystem safe string
        if isinstance(shape, cq.Shape):
            shape_id = CQUtils.get_part_checksum(shape)
        else:
            cat_shape_id = ""
            for obj in shape:
                cat_shape_id += CQUtils.get_part_checksum(obj)
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
