from typing import Any, Sequence, Union, cast
from meshly import Mesh
import pythreejs
from IPython.display import display
from IPython.core.display import HTML
import ipywidgets as widgets
import numpy as np
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
    max_edges=50000  # Limit edge rendering for performance
):
    coord_html = widgets.HTML("Coords: ()")

    def on_surf_mousemove(change):
        # write coordinates to html container
        if change.new is None:
            coord_html.value = "Coords: ()"
        else:
            coord_html.value = "Coords: (%f, %f, %f)" % change.owner.point

    if not isinstance(meshes, list):
        meshes = [meshes]

    # Legend Colors
    mesh_colors = generate_rgb_values(len(meshes), is_grayscale=True)
    marker_colors = generate_rgb_values(
        sum([len(mesh.get_reconstructed_markers()) for mesh in meshes]))

    # Legend Color Labels
    marker_color_labels = {}
    mesh_color_labels = {}

    marker_line_segments = []
    buffer_meshes = []
    for i, mesh in enumerate(meshes):
        mesh_color = mesh_colors[i]
        mesh_color_labels[f"Zone {i}"] = mesh_color
        # Marker line segment points and colors
        marker_line_points = []
        marker_segment_colors = []
        marker_elements_to_name = {}
        # Process markers in order (first-registered takes priority)
        for marker_name, marker_elements in mesh.get_reconstructed_markers().items():
            # Check if this marker type consists of line elements (2 vertices each)
            is_line = len(marker_elements) > 0 and len(marker_elements[0]) == 2

            for elements in marker_elements:
                if is_line:
                    # For line elements, elements is already [vertex1, vertex2]
                    line_from_to = (elements[0], elements[1])
                    # Allow same marker name, prevent different marker names from overwriting
                    if line_from_to not in marker_elements_to_name or marker_elements_to_name[line_from_to] == marker_name:
                        marker_elements_to_name[line_from_to] = marker_name
                else:
                    # For polygon elements, iterate all edge combinations
                    for i in range(len(elements)):
                        line_from_to = (elements[i], elements[i+1]) if i + \
                            1 < len(elements) else (elements[-1], elements[0])

                        # Allow same marker name, prevent different marker names from overwriting
                        if line_from_to not in marker_elements_to_name or marker_elements_to_name[line_from_to] == marker_name:
                            marker_elements_to_name[line_from_to] = marker_name

        # Non-marker line segment points - work directly with flattened indices for performance
        non_marker_line_points = []

        # Estimate total number of edges for performance limiting
        estimated_edges = np.sum(
            mesh.index_sizes) if mesh.index_sizes is not None else 0

        # Apply edge limiting for large meshes to prevent crashes
        edge_skip_factor = max(1, estimated_edges //
                               max_edges) if estimated_edges > max_edges else 1
        if edge_skip_factor > 1:
            print(
                f"⚠️  Large mesh detected ({estimated_edges} edges). Showing every {edge_skip_factor}th edge for performance.")

        # Work directly with flattened indices and polygon sizes for efficiency
        edge_count = 0
        if mesh.index_sizes is not None:
            offset = 0
            for poly_idx, polygon_size in enumerate(mesh.index_sizes):
                # Extract indices for this polygon
                point_tags = mesh.indices[offset:offset + polygon_size]

                # Generate edges for this polygon
                for i in range(polygon_size):
                    next_i = (i + 1) % polygon_size
                    line_point_tags = (point_tags[i], point_tags[next_i])
                    line_points = [mesh.vertices[line_point_tags[0]].tolist(),
                                   mesh.vertices[line_point_tags[1]].tolist()]

                    # Check if this edge is a marker - ALWAYS include marker edges
                    marker_point_tags = line_point_tags if line_point_tags in marker_elements_to_name else line_point_tags[
                        ::-1]
                    if marker_point_tags in marker_elements_to_name:
                        marker_name = marker_elements_to_name[marker_point_tags]
                        if marker_name not in marker_color_labels:
                            marker_color_labels[marker_name] = marker_colors[len(
                                marker_color_labels)]
                        marker_color = marker_color_labels[marker_name]
                        marker_segment_colors.append(
                            [marker_color, marker_color])
                        marker_line_points.append(line_points)
                    else:
                        # Apply performance skipping only to non-marker edges
                        if edge_skip_factor == 1 or poly_idx % edge_skip_factor == 0:
                            non_marker_line_points.append(line_points)
                            edge_count += 1

                offset += polygon_size
        else:
            # Fallback: if no index_sizes, assume triangular elements
            print("⚠️  No index_sizes found, assuming triangular mesh")

        if not only_markers and len(non_marker_line_points) > 0:
            non_marker_lines = pythreejs.LineSegments2(
                cast(Any, pythreejs.LineSegmentsGeometry(
                    positions=non_marker_line_points)),
                cast(Any, pythreejs.LineMaterial(
                    linewidth=1, color=to_rgb_str(mesh_color)))
            )
            marker_line_segments.append(non_marker_lines)

        if len(marker_line_points) > 0:
            marker_lines = pythreejs.LineSegments2(
                cast(Any, pythreejs.LineSegmentsGeometry(
                    positions=marker_line_points, colors=marker_segment_colors)),
                cast(Any, pythreejs.LineMaterial(
                    linewidth=2, vertexColors='VertexColors'))
            )
            marker_line_segments.append(marker_lines)

        # Use flattened indices directly for buffer geometry - much faster!
        flattened_indices = mesh.indices.astype(np.uint32)

        buffer_geom = pythreejs.BufferGeometry(attributes=dict(
            position=pythreejs.BufferAttribute(
                mesh.vertices, normalized=False),
            index=pythreejs.BufferAttribute(
                flattened_indices, normalized=False),
        ))

        buffer_mesh = pythreejs.Mesh(
            geometry=buffer_geom,
            material=pythreejs.MeshLambertMaterial(
                color='white', side='DoubleSide'),
        )
        buffer_meshes.append(buffer_mesh)

    camera = pythreejs.PerspectiveCamera(
        position=[0, 0, 1], far=100000, near=0.001, aspect=cast(Any, view_width/view_height))
    scene = pythreejs.Scene(children=[*marker_line_segments, *buffer_meshes,
                            pythreejs.AmbientLight(intensity=cast(int, 0.8))], background="black")
    orbit_controls = pythreejs.OrbitControls(controlling=camera)

    pickable_objects = pythreejs.Group()
    for buffer_mesh in buffer_meshes:
        pickable_objects.add(buffer_mesh)

    mousemove_picker = pythreejs.Picker(
        controlling=pickable_objects,
        event='mousemove'
    )
    mousemove_picker.observe(on_surf_mousemove, names=cast(Any, ['faceIndex']))

    renderer = pythreejs.Renderer(
        camera=camera,
        scene=scene,
        controls=[orbit_controls, mousemove_picker],
        width=view_width,
        height=view_height
    )

    # Plot renderer
    display(coord_html, renderer)

    # Plot legend
    marker_legend_html = generate_color_legend_html(
        "Markers", marker_color_labels)
    mesh_legend_html = generate_color_legend_html("Zones", mesh_color_labels)
    display(HTML(marker_legend_html+mesh_legend_html))
