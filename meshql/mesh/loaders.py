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

    # Get all mesh elements (including those in markers)
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

    # get physical groups (markers)
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
        surface_tag: The tag to use for the discrete entity (default: 1)
    """

    if not isinstance(mesh, meshly.Mesh):
        raise TypeError("mesh must be a meshly.Mesh object")


    # Create a discrete entity to hold the mesh (surface for 2D, volume for 3D)
    gmsh.model.addDiscreteEntity(mesh.dim, surface_tag)

    # Add nodes directly to the mesh
    node_tags = list(range(1, len(mesh.vertices) + 1))  # 1-based indexing
    gmsh.model.mesh.addNodes(
        dim=mesh.dim,
        tag=surface_tag,  # entity tag (must match the discrete entity)
        nodeTags=node_tags,
        # flattened array [x1,y1,z1,x2,y2,z2,...]
        coord=mesh.vertices.flatten()
    )

    # Get polygon indices once
    polygon_indices = mesh.get_polygon_indices()

    # Build a mask to identify which elements are markers (to exclude from main surface)
    marker_element_mask = _build_marker_mask(mesh, polygon_indices)

    # Add main elements (excluding marker elements)
    next_element_tag = _add_main_surface_elements(
        mesh, polygon_indices, marker_element_mask, surface_tag,
    )

    # Add marker elements as separate physical groups
    if mesh.markers:
        _add_marker_elements(mesh, next_element_tag)


def _build_marker_mask(mesh: meshly.Mesh, polygon_indices: np.ndarray) -> np.ndarray:
    """Build a boolean mask identifying which elements belong to markers.

    Args:
        mesh: The mesh containing markers
        polygon_indices: Array of polygon indices from the mesh

    Returns:
        Boolean array where True indicates the element is a marker
    """
    marker_element_mask = np.zeros(len(mesh.cell_types), dtype=bool)

    if not mesh.markers:
        return marker_element_mask

    # Build a set of marker polygon hashes for fast lookup
    marker_hashes = set()

    for marker_name in mesh.markers.keys():
        if marker_name not in mesh.marker_cell_types:
            continue

        marker_types = mesh.marker_cell_types[marker_name]
        marker_nodes = mesh.markers[marker_name]

        # Reconstruct marker polygons and add their hashes
        node_offset = 0
        for marker_type in marker_types:
            gmsh_type = VTK_TO_GMSH_ELEMENT_TYPE.get(marker_type)
            if gmsh_type:
                props = gmsh.model.mesh.getElementProperties(gmsh_type)
                num_nodes = props[3]
                marker_poly = marker_nodes[node_offset:node_offset + num_nodes]
                # Create a hashable tuple from sorted polygon indices
                marker_hashes.add(tuple(sorted(marker_poly)))
                node_offset += num_nodes

    # Mark elements using hash lookup (O(n) instead of O(n*m))
    for i, poly in enumerate(polygon_indices):
        if tuple(sorted(poly)) in marker_hashes:
            marker_element_mask[i] = True

    return marker_element_mask


def _add_main_surface_elements(
    mesh: meshly.Mesh,
    polygon_indices: np.ndarray,
    marker_element_mask: np.ndarray,
    surface_tag: int,
) -> int:
    """Add main elements to GMSH, excluding marker elements.

    Args:
        mesh: The mesh to import
        polygon_indices: Array of polygon indices
        marker_element_mask: Boolean mask indicating marker elements
        surface_tag: The entity tag

    Returns:
        Next available element tag
    """
    unique_cell_types = np.unique(mesh.cell_types)
    next_element_tag = 1

    for cell_type in unique_cell_types:
        gmsh_type = VTK_TO_GMSH_ELEMENT_TYPE.get(cell_type)
        if not gmsh_type:
            unsupported_count = np.sum(mesh.cell_types == cell_type)
            raise ValueError(
                f"Unsupported VTK cell type {cell_type} found in {unsupported_count} elements. "
                f"Cannot import mesh with unsupported cell types.")

        # Create mask for this cell type, excluding marker elements
        mask = (mesh.cell_types == cell_type) & (~marker_element_mask)
        cell_type_polygons = polygon_indices[mask]

        if len(cell_type_polygons) == 0:
            continue

        # Convert to 1-based indexing
        elements = cell_type_polygons + 1
        num_elements = len(elements)

        element_tags = np.arange(
            next_element_tag, next_element_tag + num_elements)
        next_element_tag += num_elements

        gmsh.model.mesh.addElements(
            dim=mesh.dim,
            tag=surface_tag,
            elementTypes=[gmsh_type],
            elementTags=[element_tags],
            nodeTags=[elements.flatten()]
        )

    return next_element_tag


def _add_marker_elements(mesh: meshly.Mesh, next_element_tag: int) -> None:
    """Add marker elements as separate physical groups in GMSH.

    Args:
        mesh: The mesh containing markers
        next_element_tag: Starting element tag for marker elements
    """
    for marker_name, marker_nodes in mesh.markers.items():
        if marker_name not in mesh.marker_cell_types:
            continue

        marker_types = mesh.marker_cell_types[marker_name]
        if len(marker_types) == 0:
            continue

        # Determine dimension from the first type
        first_vtk_type = marker_types[0]
        first_gmsh_type = VTK_TO_GMSH_ELEMENT_TYPE.get(first_vtk_type)
        if not first_gmsh_type:
            raise ValueError(
                f"Unsupported VTK cell type {first_vtk_type} in marker '{marker_name}'. "
                f"Cannot import marker with unsupported cell type.")

        try:
            props = gmsh.model.mesh.getElementProperties(first_gmsh_type)
            marker_dim = props[1]
        except Exception as e:
            raise ValueError(
                f"Could not determine dimension for marker '{marker_name}': {e}") from e

        # Create a discrete entity for this marker
        entity_tag = gmsh.model.addDiscreteEntity(marker_dim)

        # Find changes in cell types to process chunks
        change_indices = np.concatenate((
            np.where(marker_types[:-1] != marker_types[1:])[0] + 1,
            [len(marker_types)]
        ))

        node_offset = 0
        for start_idx, end_idx in zip([0] + change_indices[:-1].tolist(), change_indices):
            vtk_type = marker_types[start_idx]
            count = end_idx - start_idx

            gmsh_type = VTK_TO_GMSH_ELEMENT_TYPE.get(vtk_type)
            if not gmsh_type:
                raise ValueError(
                    f"Unsupported VTK cell type {vtk_type} in marker '{marker_name}'. "
                    f"Cannot import marker chunk with unsupported cell type.")

            # Get number of nodes for this type
            props = gmsh.model.mesh.getElementProperties(gmsh_type)
            num_nodes = props[3]

            # Extract nodes for this chunk
            chunk_nodes_flat = marker_nodes[node_offset:node_offset +
                                            count * num_nodes]
            chunk_node_tags = chunk_nodes_flat + 1  # Convert to 1-based tags

            # Generate element tags
            element_tags = np.arange(
                next_element_tag, next_element_tag + count)
            next_element_tag += count

            gmsh.model.mesh.addElements(
                dim=marker_dim,
                tag=entity_tag,
                elementTypes=[gmsh_type],
                elementTags=[element_tags],
                nodeTags=[chunk_node_tags]
            )

            node_offset += count * num_nodes

        # Create physical group
        p_tag = gmsh.model.addPhysicalGroup(marker_dim, [entity_tag])
        gmsh.model.setPhysicalName(marker_dim, p_tag, marker_name)
