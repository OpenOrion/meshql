from meshql.utils.types import NumpyFloat
from .mesh import VTK_TO_GMSH_ELEMENT_TYPE, GmshElementType
import numpy as np
import gmsh
import meshly


# Mapping from GMSH ElementType to VTK cell type (inverse of VTK_TO_GMSH_ELEMENT_TYPE)
GMSH_TO_VTK_ELEMENT_TYPE = {v: k for k, v in VTK_TO_GMSH_ELEMENT_TYPE.items()}


def load_from_gmsh() -> meshly.Mesh:
    from meshql.gmsh.entity import ENTITY_DIM_MAPPING

    dim = gmsh.model.getDimension()
    all_indices: list[np.uint32] = []
    all_cell_types: list[np.uint8] = []
    all_index_sizes: list[np.uint32] = []

    node_tags, points_concatted, _ = gmsh.model.mesh.getNodes()
    node_indices = np.argsort(node_tags - 1)  # type: ignore
    vertices = np.array(points_concatted, dtype=NumpyFloat).reshape(
        (-1, 3))[node_indices]

    grouped_concatted_elements = gmsh.model.mesh.getElements()
    for element_type_value, grouped_element_tags, grouped_node_tags_concatted in zip(
        *grouped_concatted_elements
    ):
        if (
            element_type_value == GmshElementType.POINT.value
            or element_type_value == GmshElementType.LINE.value
        ):
            continue
        num_nodes = gmsh.model.mesh.getElementProperties(element_type_value)[3]
        group_elements = (
            np.array(grouped_node_tags_concatted, dtype=np.uint32).reshape(
                (-1, num_nodes)
            )
            - 1
        )

        # Flatten elements and add to indices array
        all_indices.extend(group_elements.flatten())

        # Convert GMSH element type to VTK cell type
        vtk_cell_type = GMSH_TO_VTK_ELEMENT_TYPE[element_type_value]
        all_cell_types.extend([vtk_cell_type] * len(group_elements))

        # Add index sizes for each element
        all_index_sizes.extend([num_nodes] * len(group_elements))

    # Convert to numpy arrays
    indices = np.array(all_indices, dtype=np.uint32)
    cell_types = np.array(all_cell_types, dtype=np.uint8)
    index_sizes = np.array(all_index_sizes, dtype=np.uint32)

    # get physical groups
    markers: dict[str, np.ndarray] = {}
    marker_cell_types: dict[str, np.ndarray] = {}
    physical_groups = gmsh.model.getPhysicalGroups()
    for group_dim, group_tag in physical_groups:
        marker_name = gmsh.model.getPhysicalName(group_dim, group_tag)
        if len(marker_name) == 0 or group_dim == ENTITY_DIM_MAPPING["Solid"]:
            continue
        entities = gmsh.model.getEntitiesForPhysicalGroup(group_dim, group_tag)
        for entity in entities:
            marker_grouped_concatted_elements = gmsh.model.mesh.getElements(
                group_dim, tag=entity
            )
            assert (
                len(marker_grouped_concatted_elements[0]) == 1
            ), "There should only be one group"
            marker_element_type, marker_node_tags_concatted = (
                marker_grouped_concatted_elements[0][0],
                marker_grouped_concatted_elements[2][0],
            )

            # Store marker elements as list of lists
            if marker_name not in markers:
                markers[marker_name] = np.empty(0, dtype=np.uint32)
            if marker_name not in marker_cell_types:
                marker_cell_types[marker_name] = np.empty(0, dtype=np.uint8)

            # Get number of nodes per element for this element type
            num_nodes_per_element = gmsh.model.mesh.getElementProperties(marker_element_type)[
                3]

            # Calculate number of elements in this entity
            num_elements = len(
                marker_node_tags_concatted) // num_nodes_per_element

            # Convert to list of lists for automatic validation
            markers[marker_name] = np.concatenate((
                markers[marker_name],
                np.array(marker_node_tags_concatted, dtype=np.uint32) - 1
            ))
            marker_cell_types[marker_name] = np.concatenate((
                marker_cell_types[marker_name],
                np.array([GMSH_TO_VTK_ELEMENT_TYPE[marker_element_type]]
                         * num_elements, dtype=np.uint8)
            ))

    return meshly.Mesh(
        dim=dim,
        vertices=vertices,  # corresponds to points
        indices=indices,  # flattened array of all element node indices
        # size of each element (number of nodes per element)
        index_sizes=index_sizes,
        cell_types=cell_types,  # VTK cell types
        markers=markers,  # Pass as list of lists - will be auto-converted by validator
        marker_cell_types=marker_cell_types,
    )


def load_to_gmsh(mesh: meshly.Mesh, surface_tag: int = 1) -> None:
    """Import a meshly Mesh into the current gmsh model.

    Args:
        mesh: A meshly Mesh object to import
        surface_tag: The tag to use for the discrete surface entity (default: 1)
    """

    if not isinstance(mesh, meshly.Mesh):
        raise TypeError("mesh must be a meshly.Mesh object")

    # Create a discrete surface entity to hold the mesh
    gmsh.model.addDiscreteEntity(2, surface_tag)  # 2 = dimension for surface

    # Add nodes directly to the mesh
    node_tags = list(range(1, len(mesh.vertices) + 1))  # 1-based indexing
    gmsh.model.mesh.addNodes(
        dim=2,  # dimension
        tag=surface_tag,  # entity tag (must match the discrete entity)
        nodeTags=node_tags,
        # flattened array [x1,y1,z1,x2,y2,z2,...]
        coord=mesh.vertices.flatten()
    )

    # Mixed polygon mesh - handle each polygon type separately
    polygon_indices = mesh.get_polygon_indices()

    # Process each unique cell type and add elements directly to gmsh (merged for efficiency)
    unique_cell_types = np.unique(mesh.cell_types)

    for cell_type in unique_cell_types:
        gmsh_type = VTK_TO_GMSH_ELEMENT_TYPE.get(cell_type)
        if gmsh_type:
            # Create mask for this cell type
            mask = mesh.cell_types == cell_type
            # Use direct numpy boolean indexing - much cleaner and faster
            cell_type_polygons = polygon_indices[mask]
            # Convert to 1-based indexing using vectorized addition
            elements = cell_type_polygons + 1

            # Add elements directly to gmsh using vectorized operations
            element_tags = np.arange(1, len(elements) + 1)
            element_node_tags = elements.flatten()

            gmsh.model.mesh.addElements(
                dim=2,
                tag=surface_tag,  # must match the discrete entity
                elementTypes=[gmsh_type],
                elementTags=[element_tags],
                nodeTags=[element_node_tags]
            )
        else:
            # Count unsupported elements for warning
            unsupported_count = np.sum(mesh.cell_types == cell_type)
            print(
                f"Warning: Unsupported VTK cell type {cell_type}, skipping {unsupported_count} elements")
