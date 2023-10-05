from typing import Callable, Iterable, Optional, Sequence, Union
import cadquery as cq
from meshql.utils.cq import CQ_TYPE_STR_MAPPING, CQGroupTypeString, CQLinq, CQType, FilterSelector, GroupSelector, IndexSelector
from meshql.utils.types import OrderedSet
from cadquery.cq import CQObject

    
def to_selector(
    selector: Optional[Union[cq.Selector, str, None]] = None,
    group: Optional[OrderedSet[CQObject]] = None, 
    indices: Optional[Sequence[int]] = None,
    filter: Optional[Callable[[CQObject], bool]] = None
):
    selectors = []
    if isinstance(selector, str):
        selector = selectors.append(cq.StringSyntaxSelector(selector))
    elif isinstance(selector, cq.Selector):
        selectors.append(selector)
    
    if group is not None:
        selectors.append(GroupSelector(group))
    if indices is not None:
        selectors.append(IndexSelector(indices))
    if filter is not None:
        selectors.append(FilterSelector(filter))

    if len(selectors) > 0:
        prev_selector = selectors[0]
        for selector in selectors[1:]:
            prev_selector = cq.selectors.AndSelector(prev_selector, selector)
        return prev_selector
    return None



class WorkplaneSelectable:
    workplane: cq.Workplane
    def __init__(self, workplane: cq.Workplane, only_faces: bool = False, is_exclusive: bool = False, use_raycast: bool = False) -> None:
        self.workplane = workplane
        self.initial_workplane = workplane
        self.type_groups = CQLinq.groupByTypes(self.workplane, only_faces=only_faces, use_raycast=use_raycast, is_exclusive=is_exclusive)

    def end(self, num: Optional[int] = None):
        if num is None:
            self.workplane = self.initial_workplane
        else:
            self.workplane = self.workplane.end(num)
        return self

    def solids(self, selector: Union[cq.Selector, str, None] = None, tag: Union[str, None] = None, type: Optional[CQGroupTypeString] = None, indices: Optional[Sequence[int]] = None, filter: Optional[Callable[[CQObject], bool]] = None):
        type_group = type and self.type_groups[type]
        selector = to_selector(selector, type_group, indices, filter)
        self.workplane = self.workplane.solids(selector, tag)
        return self

    def faces(self, selector: Union[cq.Selector, str, None] = None, tag: Union[str, None] = None, type: Optional[CQGroupTypeString] = None, indices: Optional[Sequence[int]] = None, filter: Optional[Callable[[CQObject], bool]] = None):
        type_group = type and self.type_groups[type]
        selector = to_selector(selector, type_group, indices, filter)
        self.workplane = self.workplane.faces(selector, tag)
        return self
    
    def edges(self, selector: Union[cq.Selector, str, None] = None, tag: Union[str, None] = None, type: Optional[CQGroupTypeString] = None, indices: Optional[Sequence[int]] = None, filter: Optional[Callable[[CQObject], bool]] = None):
        type_group = type and self.type_groups[type]
        selector = to_selector(selector, type_group, indices, filter)
        self.workplane = self.workplane.edges(selector, tag)
        return self

    def wires(self, selector: Union[cq.Selector, str, None] = None, tag: Union[str, None] = None, type: Optional[CQGroupTypeString] = None, indices: Optional[Sequence[int]] = None, filter: Optional[Callable[[CQObject], bool]] = None):
        type_group = type and self.type_groups[type]
        selector = to_selector(selector, type_group, indices, filter)
        self.workplane = self.workplane.wires(selector, tag)
        return self

    def vertices(self, selector: Union[cq.Selector, str, None] = None, tag: Union[str, None] = None, type: Optional[CQGroupTypeString] = None, indices: Optional[Sequence[int]] = None, filter: Optional[Callable[[CQObject], bool]] = None):
        type_group = type and self.type_groups[type]
        selector = to_selector(selector, type_group, indices, filter)
        self.workplane = self.workplane.vertices(selector, tag)
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
