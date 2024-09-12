from dataclasses import dataclass, field
import cadquery as cq
from cadquery.cq import CQObject
from typing import Iterable, Literal, Optional, Sequence, Union
import numpy as np
from meshql.utils.types import Axis, OrderedSet, to_vec
from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
import numpy as np
import cadquery as cq

CQType2D = Literal["face", "wire", "edge", "vertex"]
CQType = Union[Literal["compound", "solid", "shell"], CQType2D]
GroupType = Literal["split", "interior", "exterior"]
CQEdgeOrFace = Union[cq.Edge, cq.Face]

CQ_TYPE_STR_MAPPING: dict[type[CQObject], CQType] = {
    cq.Compound: "compound",
    cq.Solid: "solid",
    cq.Shell: "shell",
    cq.Face: "face",
    cq.Wire: "wire",
    cq.Edge: "edge",
    cq.Vertex: "vertex",
}
CQ_TYPE_CLASS_MAPPING = dict(
    zip(CQ_TYPE_STR_MAPPING.values(), CQ_TYPE_STR_MAPPING.keys())
)

CQ_TYPE_RANKING = dict(
    zip(CQ_TYPE_STR_MAPPING.keys(), range(len(CQ_TYPE_STR_MAPPING) + 1)[::-1])
)


class CQUtils:
    @staticmethod
    def is_interior_face(face: CQObject):
        assert isinstance(face, cq.Face), "object must be a face"
        face_normal = face.normalAt()
        face_centroid = face.Center()
        interior_dot_product = face_normal.dot(face_centroid)
        return interior_dot_product < 0

    @staticmethod
    def get_normal_vec(
        faces: OrderedSet[cq.Face],
        axis: Optional[Union[Axis, Literal["avg", "face1", "face2"]]],
        offset: cq.Vector = cq.Vector(0, 0, 0),
    ):
        if axis is None:
            axis = "face1" if len(faces) == 1 else "avg"

        if axis == "avg":
            average_normal = np.average(
                [face.normalAt().toTuple() for face in faces], axis=0
            )
            norm_vec = cq.Vector(tuple(average_normal)) + offset
        elif axis == "face1":
            norm_vec = list(faces)[0].normalAt()
        elif axis == "face2":
            norm_vec = list(faces)[1].normalAt()
        else:
            norm_vec = to_vec(axis)
        return CQUtils.normalize(norm_vec + offset)

    @staticmethod
    def get_group_type(
        workplane: cq.Workplane,
        face: cq.Face,
        maxDim: float,
        tol: Optional[float] = None,
    ) -> GroupType:
        is_split = False
        total_solid = workplane.val()
        is_planar = face.geomType() == "PLANE"
        if is_planar:
            face_center, face_normal = face.Center(), face.normalAt()
            normalized_face_normal = face_normal / face_normal.Length
            is_interior = CQUtils.is_interior_face(face)
            if is_interior:
                intersect_line = cq.Edge.makeLine(
                    face_center, face_center + (normalized_face_normal * 0.1)
                )
                intersect_vertices = total_solid.intersect(
                    intersect_line, tol=tol
                ).Vertices()
                is_split = (
                    len(intersect_vertices) > 0
                    and intersect_vertices[0].distance(face) < 1e-8
                )
        else:
            try:
                nearest_center_gp_point = GeomAPI_ProjectPointOnSurf(
                    face.Center().toPnt(), face._geomAdaptor()
                ).NearestPoint()
                face_center = cq.Vector(
                    nearest_center_gp_point.X(),
                    nearest_center_gp_point.Y(),
                    nearest_center_gp_point.Z(),
                )
                face_normal = face.normalAt(face_center)
            except:
                face_center, face_normal = face.Center(), face.normalAt()
            normalized_face_normal = face_normal / face_normal.Length
            intersect_line = cq.Edge.makeLine(
                face_center, face_center + (normalized_face_normal * maxDim)
            )
            intersect_vertices = total_solid.intersect(
                intersect_line, tol=tol
            ).Vertices()
            is_interior = len(intersect_vertices) != 0
            is_split = is_interior and intersect_vertices[0].distance(face) < 1e-8

        if is_split:
            group_type = "split"
        else:
            group_type = "interior" if is_interior else "exterior"

        return group_type

    @staticmethod
    def get_angle_between(prev: CQEdgeOrFace, curr: CQEdgeOrFace):
        if isinstance(prev, cq.Edge) and isinstance(curr, cq.Edge):
            prev_tangent_vec = prev.tangentAt(0.5)  # type: ignore
            tangent_vec = curr.tangentAt(0.5)  # type: ignore
        else:
            prev_tangent_vec = prev.normalAt()  # type: ignore
            tangent_vec = curr.normalAt()  # type: ignore
        angle = prev_tangent_vec.getAngle(tangent_vec)
        assert not np.isnan(angle), "angle should not be NaN"
        return angle

    @staticmethod
    def normalize(vec: cq.Vector) -> cq.Vector:
        return vec / vec.Length

    @staticmethod
    def fuse_shapes(shapes: Sequence[CQObject]) -> cq.Shape:
        fused_shape: Optional[cq.Shape] = None
        for shape in shapes:
            assert isinstance(shape, cq.Shape), "shape must be a shape"
            if fused_shape:
                fused_shape = fused_shape.fuse(shape)
            else:
                fused_shape = shape
        assert fused_shape is not None, "No shapes to fuse"
        return fused_shape

    @staticmethod
    def get_dimension(workplane: cq.Workplane):
        return 2 if len(workplane.solids().vals()) == 0 else 3

    @staticmethod
    def import_workplane(target: Union[cq.Workplane, str, Iterable[CQObject]]):
        if isinstance(target, str):
            if target.lower().endswith(".step"):
                workplane = cq.importers.importStep(target)
            elif target.lower().endswith(".dxf"):
                workplane = cq.importers.importDXF(target)
            else:
                raise ValueError(f"Unsupported file type: {target}")
        elif isinstance(target, cq.Workplane):
            workplane = target
        elif isinstance(target, Sequence):
            workplane = cq.Workplane().newObject(target)
        else:
            raise ValueError("Invalid target type")

        return workplane

    @staticmethod
    def scale(shape: cq.Shape, x: float = 1, y: float = 1, z: float = 1) -> cq.Shape:
        t = cq.Matrix([[x, 0, 0, 0], [0, y, 0, 0], [0, 0, z, 0], [0, 0, 0, 1]])
        return shape.transformGeometry(t)

    @staticmethod
    def is_clockwise(edge1: cq.Edge, edge2: cq.Edge):
        xy_plane = cq.Plane.XY()

        start_vec: cq.Vector = edge1.endPoint().projectToPlane(
            xy_plane
        ) - edge1.startPoint().projectToPlane(xy_plane)
        end_vec: cq.Vector = edge2.endPoint().projectToPlane(
            xy_plane
        ) - edge2.startPoint().projectToPlane(xy_plane)
        normal = (end_vec.cross(start_vec)).normalized()
        return (normal.x + normal.y + normal.z) < 0

