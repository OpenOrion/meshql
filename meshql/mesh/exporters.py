from typing import Sequence, Union
from .mesh import Mesh


def export_to_su2(meshes: Union[Mesh, Sequence[Mesh]], file_path: str):
    """Export a mesh from SU2 format"""
    from su2fmt import Mesh as Su2Mesh, export_mesh
    from su2fmt.mesh import ElementType as Su2ElementType, Zone
    if not isinstance(meshes, Sequence):
        meshes = [meshes]
    zones = []
    for izone, mesh in enumerate(meshes):
        element_types: Sequence[Su2ElementType] = []
        for element_type in mesh.element_types:
            try:
                su2_element_type = Su2ElementType[element_type.name]
                element_types.append(su2_element_type)
            except:
                raise ValueError("Warning: Element type not supported: ", element_type.name)
        
        marker_types = {}
        for marker_label, marker_element_types in mesh.marker_types.items():
            marker_types[marker_label] = [Su2ElementType[marker_element_type.name] for marker_element_type in marker_element_types]
        
        zone = Zone(
            izone=izone+1,
            ndime=mesh.dim,
            elements=mesh.elements,
            element_types=element_types,
            points=mesh.points,
            markers=mesh.markers,
            marker_types=marker_types
        )
        zones.append(zone)
    su2_mesh = Su2Mesh(len(zones), zones)
    return export_mesh(su2_mesh, file_path)