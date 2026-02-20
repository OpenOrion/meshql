from functools import cached_property
import hashlib
from typing import Optional
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field

import cadquery as cq

from meshql.utils.cache import CacheService
from meshql.utils.cq import CQUtils
from meshql.utils.cq_linq import CQLinq
from meshql.utils.types import RegionGroupType, OrderedSet


class Preprocessor(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    workplane: cq.Workplane = Field(exclude=True)

    def get_checksum(self) -> str:
        json_hash = hashlib.sha256(self.model_dump_json().encode("utf-8"))
        workplane_hash = CQUtils.get_shape_checksum(self.workplane.val())
        return hashlib.sha256(json_hash.digest() + workplane_hash.encode("utf-8")).hexdigest()

    region_group_types: dict[str, RegionGroupType] = Field(
        default_factory=dict, exclude=True)

    @cached_property
    def face_edge_groups(self) -> dict[cq.Edge, OrderedSet[cq.Face]]:
        return CQLinq.groupBy(self.workplane, "Face", "Edge")

    @cached_property
    def groups(self) -> dict[RegionGroupType, OrderedSet[cq.Shape]]:
        """Compute and cache region groups."""

        region_group_types = {**self.region_group_types, **
                              (CacheService.load_regions_group_types(self.get_checksum()) or {})}

        region_groups = CQLinq.groupByRegionTypes(
            self.workplane,
            is_2d=CQUtils.get_dimension(self.workplane) == 2,
            check_splits=True,
            current_region_groups=region_group_types,
        )

        self.region_group_types = region_group_types
        CacheService.cache_regions_group_types(
            self.get_checksum(), region_group_types)
        return region_groups

    def refresh(self):
        if hasattr(self, "groups"):
            del self.groups
        if hasattr(self, "face_edge_groups"):
            del self.face_edge_groups

    def apply(self):
        """Apply preprocessing. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement apply()")

    def cache_region_groups(self):
        """Cache region groups after preprocessing is complete."""
        if self.region_group_types:
            workplane_hash = CQUtils.get_shape_checksum(self.workplane.val())
            CacheService.cache_regions_group_types(
                workplane_hash, self.region_group_types)
