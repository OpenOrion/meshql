"""Cache service for storing and retrieving preprocessor states and region groups."""

import json
from pathlib import Path
from typing import Optional
import cadquery as cq

from meshql.utils.cq import ShapeChecksum
from meshql.utils.types import RegionGroupType


class CacheService:
    """Service for caching preprocessor workplanes and region groups."""
    CACHE_DIR = Path("/tmp/meshql")

    @staticmethod
    def cache_brep(checksum: str, workplane: cq.Workplane) -> None:
        """Cache the BREP file of the workplane."""
        cache_path = CacheService.CACHE_DIR / "breps" / f"{checksum}.brep"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        workplane.val().exportBrep(str(cache_path))

    @staticmethod
    def load_brep(checksum: str) -> Optional[cq.Workplane]:
        """Load the BREP file of the workplane from cache."""
        cache_path = CacheService.CACHE_DIR / "breps" / f"{checksum}.brep"
        if cache_path.exists():
            return cq.importers.importBrep(str(cache_path))
        return None

    @staticmethod
    def cache_regions_group_types(
        checksum: str,
        region_group_types: dict[ShapeChecksum, RegionGroupType]
    ) -> None:
        """Cache the preprocessor state as a JSON file."""
        cache_path = CacheService.CACHE_DIR / "regions" / f"{checksum}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            f.write(json.dumps(region_group_types))

    @staticmethod
    def load_regions_group_types(
        checksum: str,
    ) -> Optional[dict[ShapeChecksum, RegionGroupType]]:
        """Load the preprocessor state from a cached JSON file."""
        cache_path = CacheService.CACHE_DIR / "regions" / f"{checksum}.json"

        if cache_path.exists():
            with open(cache_path, "r") as f:
                data = f.read()
            return json.loads(data)
        return None

    @staticmethod
    def clear_cache() -> None:
        """Clear all cached BREP files and region groups."""
        import shutil
        if CacheService.CACHE_DIR.exists():
            shutil.rmtree(CacheService.CACHE_DIR)
            CacheService.CACHE_DIR.mkdir(parents=True, exist_ok=True)
