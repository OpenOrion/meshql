from dataclasses import dataclass
import hashlib
import cadquery as cq
from cadquery.cq import CQObject
from typing import Callable, Iterable, Literal, Optional, Sequence, Union, cast
from meshql.utils.cq import CQ_TYPE_STR_MAPPING, CQType, CQUtils, GroupType
from meshql.utils.cq_cache import CQCache
from meshql.utils.types import OrderedSet
import cadquery as cq

SetOperation = Literal["union", "intersection", "difference"]


@dataclass
class DirectedPath:
    edge: cq.Edge
    "edge or face of path"

    direction: Literal[1, -1] = 1
    "direction of path"

    def __post_init__(self):
        assert isinstance(self.edge, cq.Edge), "edge must be an edge"
        assert self.direction in [-1, 1], "direction must be -1 or 1"
        self.vertices = self.edge.Vertices()[:: self.direction]
        self.start = self.vertices[0]
        self.end = self.vertices[-1]

    def __eq__(self, __value: object) -> bool:
        return self.edge == __value

    def __hash__(self) -> int:
        return self.edge.__hash__()


class CQLinq:
    @staticmethod
    def select_tagged(
        workplane: cq.Workplane,
        tags: Union[str, Iterable[str]],
        shape_type: Optional[CQType] = None,
    ):
        for tag in [tags] if isinstance(tags, str) else tags:
            if shape_type is None:
                yield from workplane._getTagged(tag).vals()
            else:
                yield from CQLinq.select(workplane._getTagged(tag).vals(), shape_type)

    @staticmethod
    def select(
        target: Union[cq.Workplane, Iterable[CQObject], CQObject],
        shape_type: Optional[CQType] = None,
    ):
        cq_objs = (
            target.vals()
            if isinstance(target, cq.Workplane)
            else ([target] if isinstance(target, CQObject) else list(target))
        )

        if shape_type is None:
            return list(cq_objs)

        result: list[CQObject] = []
        for cq_obj in cq_objs:
            assert isinstance(cq_obj, cq.Shape), "target must be a shape"
            if shape_type is None:
                result.append(cq_obj)
            elif shape_type == "compound":
                result.extend(cq_obj.Compounds())
            elif shape_type == "solid":
                result.extend(cq_obj.Solids())
            elif shape_type == "shell":
                result.extend(cq_obj.Shells())
            elif shape_type == "face":
                result.extend(cq_obj.Faces())
            elif shape_type == "wire":
                result.extend(cq_obj.Wires())
            elif shape_type == "edge":
                result.extend(cq_obj.Edges())
            elif shape_type == "vertex":
                result.extend(cq_obj.Vertices())
        return result

    @staticmethod
    def select_batch(
        target: Union[cq.Workplane, Iterable[CQObject]],
        parent_type: CQType,
        child_type: CQType,
    ):
        cq_objs = list(target.vals() if isinstance(target, cq.Workplane) else target)
        if CQ_TYPE_STR_MAPPING[type(cq_objs[0])] == child_type:
            yield cq_objs
        else:
            parent_cq_objs = CQLinq.select(cq_objs, parent_type)
            for parent_occ_obj in parent_cq_objs:
                yield CQLinq.select([parent_occ_obj], child_type)

    # TODO: take a look later if this function is even needed
    @staticmethod
    def filter(objs: Iterable[CQObject], filter_objs: Iterable[CQObject], invert: bool):
        filtered = []
        filter_objs = OrderedSet(filter_objs)
        for cq_obj in objs:
            if (invert and cq_obj in filter_objs) or (
                not invert and cq_obj not in filter_objs
            ):
                filtered.append(cq_obj)

        # sort filtered in the same order as filtered_objs
        return [cq_obj for cq_obj in filter_objs if cq_obj in filtered]

    @staticmethod
    def find_nearest(
        target: Union[cq.Workplane, CQObject, Sequence[CQObject]],
        near_shape: Union[cq.Shape, cq.Vector],
        tolerance: Optional[float] = None,
        excluded: Optional[Sequence[CQObject]] = None,
        shape_type: Optional[CQType] = None,
    ):
        min_dist_shape, min_dist = None, float("inf")
        for shape in CQLinq.select(
            target, shape_type or CQ_TYPE_STR_MAPPING[type(near_shape)]
        ):
            if excluded and shape in excluded:
                continue
            near_center = (
                near_shape.Center() if isinstance(near_shape, cq.Shape) else near_shape
            )
            dist = (shape.Center() - near_center).Length
            if dist != 0 and dist < min_dist:
                if (tolerance and dist <= tolerance) or not tolerance:
                    min_dist_shape, min_dist = shape, dist
        return cast(Optional[cq.Shape], min_dist_shape)

    @staticmethod
    def sortByConnect(target: Union[cq.Edge, Sequence[cq.Edge]]):
        unsorted_cq_edges = [target] if isinstance(target, cq.Edge) else target
        cq_edges = list(unsorted_cq_edges[1:])
        sorted_paths = [DirectedPath(unsorted_cq_edges[0])]
        while cq_edges:
            for i, cq_edge in enumerate(cq_edges):
                vertices = cq_edge.Vertices()
                if vertices[0].toTuple() == sorted_paths[-1].end.toTuple():
                    sorted_paths.append(DirectedPath(cq_edge))
                    cq_edges.pop(i)
                    break
                elif vertices[-1].toTuple() == sorted_paths[-1].end.toTuple():
                    sorted_paths.append(DirectedPath(cq_edge, direction=-1))
                    cq_edges.pop(i)
                    break
                elif vertices[0].toTuple() == sorted_paths[0].start.toTuple():
                    sorted_paths.insert(0, DirectedPath(cq_edge, direction=-1))
                    cq_edges.pop(i)
                    break

            else:
                raise ValueError("Edges do not form a closed loop")

        assert (
            sorted_paths[-1].end == sorted_paths[0].start
        ), "Edges do not form a closed loop"
        return sorted_paths

    # TODO: clean up this function
    @staticmethod
    def groupByRegionTypes(
        target: Union[cq.Workplane, Sequence[CQObject]],
        tol: Optional[float] = None,
        check_splits: bool = True,
        only_faces=False,
    ):
        workplane = (
            target if isinstance(target, cq.Workplane) else cq.Workplane().add(target)
        )
        max_dim = workplane.val().BoundingBox().DiagonalLength * 10
        workplane_checksum = CQCache.get_part_checksum(workplane)
        add_wire_to_group = lambda wires, group: group.update(
            [
                *wires,
                *CQLinq.select(wires, "edge"),
                *CQLinq.select(wires, "vertex"),
            ]
        )

        groups: dict[GroupType, OrderedSet[CQObject]] = {
            "split": OrderedSet[CQObject](),
            "interior": OrderedSet[CQObject](),
            "exterior": OrderedSet[CQObject](),
        }

        is_2d = CQUtils.get_dimension(workplane) == 2
        workplane = workplane.extrude(-1) if is_2d else workplane

        inner_edges = OrderedSet[tuple]()
        if not check_splits:
            for face in workplane.faces().vals():
                assert isinstance(face, cq.Face), "object must be a face"
                for innerWire in face.innerWires():
                    inner_edges.update(
                        [edge.Center().toTuple() for edge in innerWire.Edges()]
                    )

        for solid in workplane.solids().vals():
            for face in solid.Faces():
                face_group = None
                if face_group is None:
                    # check if type group has already been cached
                    # face_checksum = CQCache.get_part_checksum(face)
                    # cache_checksum = hashlib.md5(
                    #     f"{workplane_checksum}{face_checksum}".encode()
                    # ).hexdigest()

                    # if cache_checksum in CQCache.group_cache:
                    #     group_type = CQCache.group_cache[cache_checksum]

                    if check_splits:
                        group_type = CQUtils.get_group_type(
                            workplane, face, max_dim, tol
                        )
                    else:
                        group_type = "exterior"
                        for edge in face.outerWire().Edges():
                            if edge.Center().toTuple() in inner_edges:
                                group_type = "interior"
                                break
                    face_group = groups[group_type]

                    # store group type in cache
                    # CQCache.group_cache[cache_checksum] = group_type

                face_group.add(face)
                if not only_faces:
                    add_wire_to_group(face.Wires(), face_group)

        if is_2d:
            exterior_group_tmp = groups["exterior"].difference(groups["interior"])
            groups["interior"] = groups["interior"].difference(groups["exterior"])
            groups["exterior"] = exterior_group_tmp

        return groups

    @staticmethod
    def groupBySet(
        target: Union[cq.Workplane, Iterable[CQObject], CQObject],
        group_type: GroupType,
        group_types: dict[str, cq.Workplane],
        set_operation: SetOperation,
    ):
        cq_objs = OrderedSet(
            target.vals()
            if isinstance(target, cq.Workplane)
            else ([target] if isinstance(target, CQObject) else list(target))
        )

        inv_type = "interior" if group_type == "exterior" else "exterior"

        if set_operation == "difference":
            return cq_objs.difference(group_types[inv_type])
        elif set_operation == "intersection":
            return cq_objs.intersection(group_types[inv_type])

        elif set_operation == "union":
            return cq_objs.union(group_types[inv_type])

    @staticmethod
    def groupBy(
        target: Union[cq.Workplane, Iterable[CQObject]],
        parent_type: CQType,
        child_type: CQType,
    ):
        groups: dict[CQObject, OrderedSet[CQObject]] = {}

        cq_objs = (
            target.vals()
            if isinstance(target, cq.Workplane)
            else (target if isinstance(target, Iterable) else [target])
        )
        for cq_obj in cq_objs:
            parents = CQLinq.select(cq_obj, parent_type)
            for parent in parents:
                children = CQLinq.select(parent, child_type)
                for child in children:
                    if child not in groups:
                        groups[child] = OrderedSet[CQObject]()
                    groups[child].add(parent)
        return groups

    @staticmethod
    def find(
        target: Union[cq.Workplane, Iterable[CQObject]],
        where: Callable[[CQObject], bool],
    ):
        cq_objs = (
            target.vals()
            if isinstance(target, cq.Workplane)
            else (target if isinstance(target, Iterable) else [target])
        )
        for cq_obj in cq_objs:
            if where(cq_obj):
                yield cq_obj
