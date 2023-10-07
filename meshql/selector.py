from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Union, cast
import cadquery as cq
from meshql.utils.cq import CQ_TYPE_STR_MAPPING, CQGroupTypeString, CQLinq, CQType, FilterSelector, GroupSelector, IndexSelector
from meshql.utils.types import OrderedSet
from cadquery.cq import CQObject

    
@dataclass
class Selection:
    selector: Union[cq.Selector, str, None] = None
    tag: Union[str, None] = None
    type: Optional[CQGroupTypeString] = None
    indices: Optional[Sequence[int]] = None 
    filter: Optional[Callable[[CQObject], bool]] = None

    def select(self, selectable: "WorkplaneSelectable", cq_type: CQType, is_initial: bool = False, is_exclusive: bool = False, is_intersection: bool = False):
        workplane = selectable.initial_workplane if is_initial else selectable.workplane
        cq_obj = workplane._getTagged(self.tag) if self.tag else workplane
        filtered_entities = list(CQLinq.select(cq_obj, cq_type))

        if isinstance(self.selector, str):
            filtered_entities = cq.StringSyntaxSelector(self.selector).filter(filtered_entities)
        elif isinstance(self.selector, cq.Selector):
            filtered_entities = self.selector.filter(filtered_entities)

        if self.type:
            inv_type = "exterior" if self.type == "interior" else "interior"
            if is_exclusive:
                type_group = self.type and selectable.type_groups[self.type].difference(selectable.type_groups[inv_type])
            elif is_intersection:
                type_group = self.type and selectable.type_groups[self.type].intersection(selectable.type_groups[inv_type])
            else:
                type_group = selectable.type_groups[self.type]
            filtered_entities = GroupSelector(type_group).filter(filtered_entities)

        if self.indices is not None:
            filtered_entities = IndexSelector(self.indices).filter(filtered_entities)
        if self.filter is not None:
            filtered_entities = FilterSelector(self.filter).filter(filtered_entities)

        return filtered_entities

class WorkplaneSelectable:
    workplane: cq.Workplane
    def __init__(self, workplane: cq.Workplane) -> None:
        self.workplane = workplane
        self.initial_workplane = workplane
        
    def refresh(self, use_raycast: bool = False):
        self.type_groups = CQLinq.groupByTypes(self.workplane, use_raycast=use_raycast)
        self.face_edge_groups = cast(
            dict[cq.Edge, OrderedSet[cq.Face]], 
            CQLinq.groupBy(self.workplane, "face", "edge")
        )

    def end(self, num: Optional[int] = None):
        if num is None:
            self.workplane = self.initial_workplane
        else:
            self.workplane = self.workplane.end(num)
        return self

    def solids(self, selector: Union[cq.Selector, str, None] = None, tag: Union[str, None] = None, type: Optional[CQGroupTypeString] = None, indices: Optional[Sequence[int]] = None, filter: Optional[Callable[[CQObject], bool]] = None):
        selection = Selection(selector, tag, type, indices, filter)
        self.workplane = self.workplane.newObject(selection.select(self, "solid"))
        return self

    def faces(self, selector: Union[cq.Selector, str, None] = None, tag: Union[str, None] = None, type: Optional[CQGroupTypeString] = None, indices: Optional[Sequence[int]] = None, filter: Optional[Callable[[CQObject], bool]] = None):
        selection = Selection(selector, tag, type, indices, filter)
        self.workplane = self.workplane.newObject(selection.select(self, "face"))
        return self
    
    def edges(self, selector: Union[cq.Selector, str, None] = None, tag: Union[str, None] = None, type: Optional[CQGroupTypeString] = None, indices: Optional[Sequence[int]] = None, filter: Optional[Callable[[CQObject], bool]] = None):
        selection = Selection(selector, tag, type, indices, filter)
        self.workplane = self.workplane.newObject(selection.select(self, "edge"))
        return self

    def wires(self, selector: Union[cq.Selector, str, None] = None, tag: Union[str, None] = None, type: Optional[CQGroupTypeString] = None, indices: Optional[Sequence[int]] = None, filter: Optional[Callable[[CQObject], bool]] = None):
        selection = Selection(selector, tag, type, indices, filter)
        self.workplane = self.workplane.newObject(selection.select(self, "wire"))
        return self

    def vertices(self, selector: Union[cq.Selector, str, None] = None, tag: Union[str, None] = None, type: Optional[CQGroupTypeString] = None, indices: Optional[Sequence[int]] = None, filter: Optional[Callable[[CQObject], bool]] = None):
        selection = Selection(selector, tag, type, indices, filter)
        self.workplane = self.workplane.newObject(selection.select(self, "vertex"))
        return self

    def fromTagged(self, tags: Union[str, Iterable[str]], resolve_type: Optional[CQType] = None, invert: bool = True):        
        if isinstance(tags, str) and resolve_type is None:
            self.workplane = self.workplane._getTagged(tags)
        else:
            tagged_objs = list(CQLinq.select_tagged(self.workplane, tags, resolve_type))
            tagged_cq_type = CQ_TYPE_STR_MAPPING[type(tagged_objs[0])]
            workplane_objs = CQLinq.select(self.workplane, tagged_cq_type)
            filtered_objs = CQLinq.filter(workplane_objs, tagged_objs, invert)
            self.workplane = self.workplane.newObject(filtered_objs)
        return self
