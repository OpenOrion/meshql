from typing import Callable, Iterable, Optional, Sequence, Union
import cadquery as cq
from meshql.utils.cq import CQ_TYPE_STR_MAPPING, CQGroupTypeString, CQLinq, CQType, FilterSelector, GroupSelector, IndexSelector
from meshql.utils.types import OrderedSet
from cadquery.cq import CQObject

class Selector:
    @staticmethod
    def create(
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
        raise ValueError("No selector provided")



class SelectorQuerier:
    workplane: cq.Workplane
    def __init__(self, workplane: cq.Workplane, only_faces: bool = False, use_raycast: bool = False) -> None:
        self.workplane = workplane
        self.type_groups = CQLinq.groupByTypes(self.workplane, only_faces=only_faces, use_raycast=use_raycast)

    def solids(self, selector: Union[cq.Selector, str, None] = None, tag: Union[str, None] = None, type: Optional[CQGroupTypeString] = None, indices: Optional[Sequence[int]] = None, filter: Optional[Callable[[CQObject], bool]] = None):
        obj_type = type and self.type_groups[type]
        selector = Selector.create(selector, obj_type, indices, filter)
        self.workplane = self.workplane.solids(selector, tag)
        return self

    def faces(self, selector: Union[cq.Selector, str, None] = None, tag: Union[str, None] = None, type: Optional[CQGroupTypeString] = None, indices: Optional[Sequence[int]] = None, filter: Optional[Callable[[CQObject], bool]] = None):
        obj_type = type and self.type_groups[type]
        selector = Selector.create(selector, obj_type, indices, filter)
        self.workplane = self.workplane.faces(selector, tag)
        return self
    
    def edges(self, selector: Union[cq.Selector, str, None] = None, tag: Union[str, None] = None, type: Optional[CQGroupTypeString] = None, indices: Optional[Sequence[int]] = None, filter: Optional[Callable[[CQObject], bool]] = None):
        obj_type = type and self.type_groups[type]
        selector = Selector.create(selector, obj_type, indices, filter)
        self.workplane = self.workplane.edges(selector, tag)
        return self

    def wires(self, selector: Union[cq.Selector, str, None] = None, tag: Union[str, None] = None, type: Optional[CQGroupTypeString] = None, indices: Optional[Sequence[int]] = None, filter: Optional[Callable[[CQObject], bool]] = None):
        obj_type = type and self.type_groups[type]
        selector = Selector.create(selector, obj_type, indices, filter)
        self.workplane = self.workplane.wires(selector, tag)
        return self

    def vertices(self, selector: Union[cq.Selector, str, None] = None, tag: Union[str, None] = None, type: Optional[CQGroupTypeString] = None, indices: Optional[Sequence[int]] = None, filter: Optional[Callable[[CQObject], bool]] = None):
        obj_type = type and self.type_groups[type]
        selector = Selector.create(selector, obj_type, indices, filter)
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
