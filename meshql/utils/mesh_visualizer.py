from typing import Any, Sequence, Union, cast
from meshly import Mesh, MeshUtils
import pythreejs
from IPython.display import display
from IPython.core.display import HTML
import ipywidgets as widgets
from ipywidgets.embed import embed_minimal_html
import numpy as np
import tempfile
import webbrowser
import os
import colorsys


def generate_color_legend_html(title: str, color_labels: dict[str, list[int]]):
    title = f"<h2>{title}</h2>"
    legend = '<table>'
    for label, color in color_labels.items():
        assert len(color) == 3, "Color must be a list of 3 integers"
        legend += f'<tr><td style="background-color: {to_rgb_str(color)}" width="20"></td><td>{label}</td></tr>'
    legend += '</table>'
    return f'<div style="float: left; padding-right: 50px">{title+legend}</div>'


def generate_rgb_values(n_colors, is_grayscale=False):
    if n_colors == 0:
        return []
    colors = []
    for i in np.arange(0., 360., 360. / n_colors):
        hue = i/360.
        if is_grayscale:
            min_rgb = 0.5
            rgb = (1 - min_rgb)*hue + min_rgb
            rgb_values = [rgb, rgb, rgb]
        else:
            lightness = (50 + np.random.rand() * 10)/100.
            saturation = (90 + np.random.rand() * 10)/100.
            rgb_values = list(colorsys.hls_to_rgb(hue, lightness, saturation))

        colors.append(rgb_values)

    return colors


def to_rgb_str(color: Sequence[int]):
    return f"rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})"


def visualize_mesh(
    meshes: Union[Mesh, list[Mesh]],
    view_width=800,
    view_height=600,
    only_markers=False,
    output_path: str = None,
    open_in_browser: bool = False,
    only_surface: bool = False,
    only_volume: bool = False
):

    if not isinstance(meshes, list):
        meshes = [meshes]

    # Legend Colors
    mesh_colors = generate_rgb_values(len(meshes), is_grayscale=True)
    marker_colors = generate_rgb_values(
        sum([len(mesh.get_reconstructed_markers()) for mesh in meshes]))

    # Legend Color Labels
    marker_color_labels = {}
    element_color_labels = {}

    marker_line_segments = []
    wireframe_meshes = []

    SURFACE_COLOR = [0.6, 0.6, 0.6]
    VOLUME_COLOR = [0.5, 0.5, 0.5]

    for i, mesh in enumerate(meshes):
        # Ensure vertices are numpy array
        vertices = np.array(mesh.vertices, dtype=np.float32)

        has_types = hasattr(mesh, 'cell_types') and hasattr(
            mesh, 'index_sizes') and mesh.index_sizes is not None

        if has_types:
            # VTK types:
            # Surface: 5 (Triangle), 9 (Quad)
            # Volume: 10 (Tetra), 12 (Hex), 13 (Wedge), 14 (Pyramid)
            surface_mask = np.isin(mesh.cell_types, [5, 9])
            volume_mask = np.isin(mesh.cell_types, [10, 12, 13, 14])

            def extract_mesh_by_type(mask):
                """Extract a submesh containing only cells matching the mask."""
                if not np.any(mask):
                    return None
                
                # Get indices of matching cells
                matching_cells = np.where(mask)[0]
                
                # Calculate offsets for each cell in the flattened indices array
                cell_offsets = np.concatenate([[0], np.cumsum(mesh.index_sizes[:-1])])
                
                # Extract indices and sizes for matching cells
                new_indices = []
                new_sizes = []
                new_types = []
                
                for cell_idx in matching_cells:
                    offset = int(cell_offsets[cell_idx])
                    size = int(mesh.index_sizes[cell_idx])
                    new_indices.extend(mesh.indices[offset:offset+size].tolist())
                    new_sizes.append(size)
                    new_types.append(int(mesh.cell_types[cell_idx]))
                
                if not new_indices:
                    return None
                    
                return Mesh(
                    vertices=mesh.vertices,
                    indices=np.array(new_indices, dtype=np.uint32),
                    index_sizes=np.array(new_sizes, dtype=np.uint32),
                    cell_types=np.array(new_types, dtype=np.uint8)
                )

            # Store original mesh for wireframe
            surface_mesh_original = None
            volume_mesh_original = None
            
            if only_surface:
                # Extract surface elements (keep original structure for wireframe)
                surface_mesh_original = extract_mesh_by_type(surface_mask)
            elif only_volume:
                volume_mesh_original = extract_mesh_by_type(volume_mask)
            else:
                volume_mesh_original = extract_mesh_by_type(volume_mask)
                # Only show surface elements if there are no volume elements
                # (to avoid duplicate faces on volume boundaries)
                if volume_mesh_original is None:
                    surface_mesh_original = extract_mesh_by_type(surface_mask)
        else:
            # Fallback: create a mesh from all indices if no type information
            if not only_volume and len(mesh.indices) > 0:
                surface_mesh_original = mesh

        # --- 1. Process Markers ---
        marker_edges_map = {}  # (u, v) -> color

        markers = mesh.get_reconstructed_markers()
        for marker_name, marker_elements in markers.items():
            if not marker_elements:
                continue

            # Assign color
            if marker_name not in marker_color_labels:
                marker_color_labels[marker_name] = marker_colors[len(
                    marker_color_labels)]
            color = marker_color_labels[marker_name]

            # Check type
            is_line = len(marker_elements[0]) == 2

            if is_line:
                # Elements are [u, v]
                # Use loop for safety with mixed types, but could be vectorized if consistent
                for edge in marker_elements:
                    u, v = edge[0], edge[1]
                    if u > v:
                        u, v = v, u
                    marker_edges_map[(u, v)] = color
            else:
                # Polygons
                for poly in marker_elements:
                    for j in range(len(poly)):
                        u, v = poly[j], poly[(j+1) % len(poly)]
                        if u > v:
                            u, v = v, u
                        marker_edges_map[(u, v)] = color

        # Create marker geometry
        if marker_edges_map:
            # Convert to lists for pythreejs (it handles lists of segments well)
            # Using numpy for positions/colors is better for performance
            marker_edges_keys = list(marker_edges_map.keys())
            marker_edges_colors = list(marker_edges_map.values())

            marker_edges_arr = np.array(marker_edges_keys)

            # Extract positions: (M, 2, 3)
            marker_pos = vertices[marker_edges_arr]

            # Extract colors: (M, 2, 3)
            # Each edge has 2 vertices with same color
            marker_col = np.array(
                [c for c in marker_edges_colors], dtype=np.float32)  # (M, 3)
            marker_col = np.repeat(
                marker_col[:, np.newaxis, :], 2, axis=1)  # (M, 2, 3)

            marker_lines = pythreejs.LineSegments2(
                cast(Any, pythreejs.LineSegmentsGeometry(
                    positions=marker_pos,
                    colors=marker_col
                )),
                cast(Any, pythreejs.LineMaterial(
                    linewidth=2,
                    vertexColors='VertexColors'
                ))
            )
            marker_line_segments.append(marker_lines)

        # Helper to create wireframe from mesh with proper element edge extraction
        def create_wireframe_from_mesh(original_mesh, color, label):
            """Extract edges from original mesh structure (respects quads, triangles, etc.)"""
            if original_mesh is None:
                return
            
            element_color_labels[label] = color
            
            # Extract all edges from the mesh polygons
            edge_set = set()
            polygons = original_mesh.get_polygon_indices()
            
            for poly in polygons:
                n = len(poly)
                for i in range(n):
                    u, v = poly[i], poly[(i + 1) % n]
                    if u > v:
                        u, v = v, u
                    edge_set.add((u, v))
            
            if not edge_set:
                return
            
            # Convert edges to line segments for pythreejs
            edges = list(edge_set)
            edge_arr = np.array(edges, dtype=np.uint32)
            
            # Extract positions: (M, 2, 3)
            edge_pos = vertices[edge_arr]
            
            # Extract colors: (M, 2, 3) - same color for both vertices of each edge
            edge_col = np.array([color for _ in edges], dtype=np.float32)  # (M, 3)
            edge_col = np.repeat(edge_col[:, np.newaxis, :], 2, axis=1)  # (M, 2, 3)
            
            wireframe_lines = pythreejs.LineSegments2(
                cast(Any, pythreejs.LineSegmentsGeometry(
                    positions=edge_pos,
                    colors=edge_col
                )),
                cast(Any, pythreejs.LineMaterial(
                    linewidth=1,
                    vertexColors='VertexColors'
                ))
            )
            wireframe_meshes.append(wireframe_lines)

        # Create wireframes from original mesh structures
        if not only_markers:
            create_wireframe_from_mesh(surface_mesh_original, SURFACE_COLOR, "Surface Elements")
            create_wireframe_from_mesh(volume_mesh_original, VOLUME_COLOR, "Volume Elements")

    camera = pythreejs.PerspectiveCamera(
        position=[0, 0, 1], far=100000, near=0.001, aspect=cast(Any, view_width/view_height))
    scene = pythreejs.Scene(children=[*marker_line_segments, *wireframe_meshes,
                            pythreejs.AmbientLight(intensity=cast(int, 0.8))], background="black")
    orbit_controls = pythreejs.OrbitControls(controlling=camera)

    renderer = pythreejs.Renderer(
        camera=camera,
        scene=scene,
        controls=[orbit_controls],
        width=view_width,
        height=view_height
    )

    # Plot legend
    marker_legend_html = generate_color_legend_html(
        "Markers", marker_color_labels)
    element_legend_html = generate_color_legend_html(
        "Elements", element_color_labels)

    legend_widget = widgets.HTML(value=marker_legend_html+element_legend_html)
    content = widgets.VBox([renderer, legend_widget])

    if open_in_browser:
        # Create a temporary file
        fd, path = tempfile.mkstemp(suffix='.html')
        try:
            with os.fdopen(fd, 'w') as tmp:
                pass  # Just creating the file

            embed_minimal_html(
                path, views=[content], title="Mesh Visualization")
            print(f"Opening visualization in browser: {path}")
            webbrowser.open('file://' + path)
        except Exception as e:
            print(f"Error opening browser: {e}")
            # Fallback to display if browser fails
            display(content)

    elif output_path:
        embed_minimal_html(output_path, views=[
                           content], title="Mesh Visualization")
        print(f"Visualization exported to {output_path}")
    else:
        display(content)
