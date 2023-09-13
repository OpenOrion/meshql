from typing import Any, Sequence, Union, cast
from .mesh.mesh import Mesh
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
    colors=[]
    for i in np.arange(0., 360., 360. / n_colors):
        hue = i/360.
        if is_grayscale:
            min_rgb = 0.5
            rgb = (1 - min_rgb)*hue + min_rgb
            rgb_values = [rgb,rgb,rgb]
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
    only_markers=False
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
    marker_colors = generate_rgb_values(sum([len(mesh.markers) for mesh in meshes]))

    # Legend Color Labels
    marker_color_labels = {}
    mesh_color_labels = {}

    marker_line_segments = []
    buffer_meshes = []
    target_point_spheres = []
    for i, mesh in enumerate(meshes):
        mesh_color = mesh_colors[i]
        mesh_color_labels[f"Zone {i}"] = mesh_color
        max_point = mesh.points.min(axis=0)
        min_point = mesh.points.max(axis=0)
        point_size = max(max_point[0] - min_point[0], max_point[1] - min_point[1])*0.03
        # Marker line segment points and colors
        marker_line_points = []
        marker_segment_colors = []
        marker_elements_to_name = {}
        for marker_name, marker_elements in mesh.markers.items():
            is_line = len(marker_elements) == 2
            for elements in marker_elements:
                # iterate all marker element line combinations
                for i in range(len(elements)):
                    line_from_to = (elements[i], elements[i+1]) if i+1 < len(elements) else (elements[-1], elements[0])
                    
                    # only allow mesh line label override if line is appart of edge if already labeled for edge mesh line
                    if (line_from_to in marker_elements_to_name and is_line) or line_from_to not in marker_elements_to_name:
                        marker_elements_to_name[line_from_to] = marker_name


                if marker_name in mesh.target_points and elements[1] in mesh.target_points[marker_name]:
                    target_point_sphere = pythreejs.Mesh(
                        geometry=pythreejs.SphereGeometry(radius=point_size),
                        material=pythreejs.MeshLambertMaterial(color='red', side='DoubleSide'),
                    )
                    target_point_sphere.position = mesh.points[elements[1]].tolist()
                    target_point_spheres.append(target_point_sphere)

        # Non-marker line segment points
        non_marker_line_points = []
        for point_tags in mesh.elements:
            for i in range(len(point_tags)):
                if i + 1 < len(point_tags):
                    line_point_tags = (point_tags[i+1], point_tags[i])
                else:
                    line_point_tags = (point_tags[0], point_tags[i])
                line_points = [mesh.points[line_point_tags[0]].tolist(), mesh.points[line_point_tags[1]].tolist()]

                marker_point_tags = line_point_tags if line_point_tags in marker_elements_to_name else line_point_tags[::-1]
                if marker_point_tags in marker_elements_to_name:
                    marker_name = marker_elements_to_name[marker_point_tags]
                    if marker_name not in marker_color_labels:
                        marker_color_labels[marker_name] = marker_colors[len(marker_color_labels)]
                    marker_color = marker_color_labels[marker_name]
                    marker_segment_colors.append([marker_color, marker_color])
                    marker_line_points.append(line_points)
                else:
                    non_marker_line_points.append(line_points)

        if not only_markers and len(non_marker_line_points) > 0:
            non_marker_lines = pythreejs.LineSegments2(
                cast(Any, pythreejs.LineSegmentsGeometry(positions=non_marker_line_points)),
                cast(Any, pythreejs.LineMaterial(linewidth=1, color=to_rgb_str(mesh_color)))
            )
            marker_line_segments.append(non_marker_lines)

        if len(marker_line_points) > 0:
            marker_lines = pythreejs.LineSegments2(
                cast(Any, pythreejs.LineSegmentsGeometry(positions=marker_line_points, colors=marker_segment_colors)),
                cast(Any, pythreejs.LineMaterial(linewidth=2, vertexColors='VertexColors'))
            )
            marker_line_segments.append(marker_lines)

        if len(mesh.elements) > 0:
            buffer_geom = pythreejs.BufferGeometry(attributes=dict(
                position=pythreejs.BufferAttribute(mesh.points, normalized=False),
                index=pythreejs.BufferAttribute(np.concatenate(mesh.elements), normalized=False),
            ))

            buffer_mesh = pythreejs.Mesh(
                geometry=buffer_geom,
                material=pythreejs.MeshLambertMaterial(color='white', side='DoubleSide'),
            )
            buffer_meshes.append(buffer_mesh)

    camera = pythreejs.PerspectiveCamera(position=[0, 0, 1], far=100000, near=0.001, aspect=cast(Any, view_width/view_height))
    scene = pythreejs.Scene(children=[*marker_line_segments, *buffer_meshes, *target_point_spheres, pythreejs.AmbientLight(intensity=cast(int, 0.8))], background="black")
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
    marker_legend_html = generate_color_legend_html("Markers", marker_color_labels)
    mesh_legend_html = generate_color_legend_html("Zones", mesh_color_labels)
    display(HTML(marker_legend_html+mesh_legend_html))
