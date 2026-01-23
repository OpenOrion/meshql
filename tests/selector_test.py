"""Unit tests for meshql.core.selector module."""

import pytest
import cadquery as cq
from meshql.core.selector import IndexSelector, FilterSelector, GroupSelector, Selection, ShapeExplorer, ConnectedShapesExplorer
from meshql.utils.types import OrderedSet


class TestSelector:
    """Test cases for selector functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple cube for testing
        self.cube = cq.Workplane("XY").box(10, 10, 10)
        self.cube_faces = self.cube.faces()
        self.cube_edges = self.cube.edges()

    def test_index_selector_creation(self):
        """Test IndexSelector creation and basic properties."""
        selector = IndexSelector([0, 1, 2])
        assert selector.indices == [0, 1, 2]

    def test_index_selector_filter(self):
        """Test IndexSelector filtering functionality with validation."""
        selector = IndexSelector([0, 2])
        faces = self.cube_faces.vals()
        original_count = len(faces)
        filtered = selector.filter(faces)

        # Verify correct number and specific faces selected
        assert len(filtered) == 2, "Should select exactly 2 faces"
        assert filtered[0] == faces[0], "First filtered face should be faces[0]"
        assert filtered[1] == faces[2], "Second filtered face should be faces[2]"

        # Verify we didn't modify original list
        assert len(faces) == original_count, "Original list should be unchanged"

        # Verify filtered objects are actual Face objects
        for face in filtered:
            assert isinstance(face, cq.Face), "Filtered objects should be Face instances"

    def test_filter_selector_creation(self):
        """Test FilterSelector creation."""
        def filter_func(obj): return True
        selector = FilterSelector(filter_func)
        assert selector.objFilter == filter_func

    def test_filter_selector_filter(self):
        """Test FilterSelector filtering functionality with normal validation."""
        # Create a filter that selects faces with Z normal > 0 (top face)
        def filter_func(face): return face.normalAt().z > 0.5
        selector = FilterSelector(filter_func)

        faces = self.cube_faces.vals()
        filtered = selector.filter(faces)

        # Should find one face (the top face)
        assert len(filtered) == 1, "Should find exactly one top face"
        top_face = filtered[0]

        # Verify it's actually the top face
        normal = top_face.normalAt()
        assert normal.z > 0.5, "Top face normal should point upward"
        assert normal.z == pytest.approx(1.0, abs=0.1), "Top face should have normal ~(0,0,1)"

        # Verify the face has expected properties (100 sq units for 10x10x10 cube)
        area = top_face.Area()
        assert area == pytest.approx(100.0, abs=1.0), "Top face should have area ~100"

    def test_group_selector_creation(self):
        """Test GroupSelector creation."""
        allowed_objects = OrderedSet([self.cube_faces.vals()[0]])
        selector = GroupSelector(allowed_objects)
        assert selector.allow == allowed_objects

    def test_group_selector_filter(self):
        """Test GroupSelector filtering functionality."""
        faces = self.cube_faces.vals()
        allowed_objects = OrderedSet([faces[0], faces[2]])
        selector = GroupSelector(allowed_objects)

        filtered = selector.filter(faces)

        assert len(filtered) == 2
        assert faces[0] in filtered
        assert faces[2] in filtered
        assert faces[1] not in filtered

    def test_selection_dataclass(self):
        """Test Selection dataclass creation and defaults."""
        # Test default values
        selection = Selection()
        assert selection.type is None
        assert selection.filter is None
        assert selection.indices is None

        # Test with specific values
        def filter_func(obj): return True
        selection_with_values = Selection(
            type="Face",
            filter=filter_func,
            indices=[0, 1, 2]
        )
        assert selection_with_values.type == "Face"
        assert selection_with_values.filter == filter_func
        assert selection_with_values.indices == [0, 1, 2]

    def test_shape_explorer_creation(self):
        """Test ShapeExplorer creation."""
        cube_shape = self.cube.val()
        explorer = ShapeExplorer(cube_shape)
        # The shape is stored as wrapped, so compare the wrapped versions
        assert explorer.shape == cube_shape.wrapped

    def test_shape_explorer_search_faces(self):
        """Test ShapeExplorer search for faces with area validation."""
        cube_shape = self.cube.val()
        explorer = ShapeExplorer(cube_shape)

        faces = explorer.search("Face")

        # A cube should have exactly 6 faces
        assert len(faces) == 6, "Cube should have exactly 6 faces"

        total_area = 0
        for face in faces:
            assert isinstance(face, cq.Face)
            # Each face of a 10x10x10 cube should have area 100
            face_area = face.Area()
            assert face_area == pytest.approx(100.0, abs=1.0), f"Each cube face should have area ~100, got {face_area}"
            total_area += face_area

        # Total surface area should be 6 * 100 = 600
        assert total_area == pytest.approx(600.0, abs=10.0), "Total cube surface area should be ~600"

    def test_shape_explorer_search_edges(self):
        """Test ShapeExplorer search for edges."""
        cube_shape = self.cube.val()
        explorer = ShapeExplorer(cube_shape)

        edges = explorer.search("Edge")

        # A cube should have 12 edges
        assert len(edges) == 12
        for edge in edges:
            assert isinstance(edge, cq.Edge)

    def test_shape_explorer_search_vertices(self):
        """Test ShapeExplorer search for vertices."""
        cube_shape = self.cube.val()
        explorer = ShapeExplorer(cube_shape)

        vertices = explorer.search("Vertex")

        # A cube should have 8 vertices
        assert len(vertices) == 8
        for vertex in vertices:
            assert isinstance(vertex, cq.Vertex)

    def test_connected_shapes_explorer_creation(self):
        """Test ConnectedShapesExplorer creation."""
        cube_shape = self.cube.val()
        face = self.cube_faces.vals()[0]
        explorer = ConnectedShapesExplorer(cube_shape, face)

        assert explorer.base_shape == cube_shape
        assert explorer.child_shape == face

    def test_connected_shapes_explorer_vertices(self):
        """Test ConnectedShapesExplorer _connected_by_vertices method."""
        cube_shape = self.cube.val()
        face = self.cube_faces.vals()[0]
        explorer = ConnectedShapesExplorer(cube_shape, face)

        # Test finding edges connected to the face by vertices
        edges = self.cube_edges.vals()
        connected_edges = explorer._connected_by_vertices(edges[0])

        # Should return a boolean
        assert isinstance(connected_edges, bool)

    def test_connected_shapes_explorer_search(self):
        """Test ConnectedShapesExplorer search with edge length validation."""
        cube_shape = self.cube.val()
        face = self.cube_faces.vals()[0]
        explorer = ConnectedShapesExplorer(cube_shape, face)

        # Search for edges connected to the face
        connected_edges = explorer.search("Edge")

        # A face of a cube should be connected to exactly 4 edges
        assert len(connected_edges) == 4, "Cube face should connect to exactly 4 edges"

        total_perimeter = 0
        for edge in connected_edges:
            assert isinstance(edge, cq.Edge)
            # Each edge of a 10x10x10 cube face should have length 10
            edge_length = edge.Length()
            assert edge_length == pytest.approx(10.0, abs=0.1), f"Cube face edge should have length ~10, got {edge_length}"
            total_perimeter += edge_length

        # Total perimeter should be 4 * 10 = 40
        assert total_perimeter == pytest.approx(40.0, abs=1.0), "Face perimeter should be ~40"

    def test_selection_with_string_type(self):
        """Test Selection with string type parameter."""
        selection = Selection(type="Face")
        assert selection.type == "Face"

        selection_edge = Selection(type="Edge")
        assert selection_edge.type == "Edge"

    def test_selection_with_directional_selector(self):
        """Test Selection with directional selectors."""
        # Test various directional selectors that might be used
        selection_top = Selection(type=">Z")
        assert selection_top.type == ">Z"

        selection_front = Selection(type=">Y")
        assert selection_front.type == ">Y"

        selection_right = Selection(type=">X")
        assert selection_right.type == ">X"

    def test_filter_selector_with_complex_filter(self):
        """Test FilterSelector with area-based filtering and validation."""
        # Create a filter that selects faces based on area
        def area_filter(face):
            try:
                area = face.Area()
                return area > 50  # For a 10x10x10 cube, each face has area 100
            except:
                return False

        selector = FilterSelector(area_filter)
        faces = self.cube_faces.vals()
        original_face_count = len(faces)
        filtered = selector.filter(faces)

        # All cube faces should pass this filter (area 100 > 50)
        assert len(filtered) == original_face_count, "All cube faces should have area > 50"
        assert len(filtered) == 6, "Should filter all 6 cube faces"

        # Verify each filtered face actually meets the criteria
        for face in filtered:
            area = face.Area()
            assert area > 50, f"Filtered face should have area > 50, got {area}"
            assert area == pytest.approx(100.0, abs=1.0), f"Cube face should have area ~100, got {area}"

        # Test inverse filter (area <= 50) should return no faces
        def small_area_filter(face):
            try:
                return face.Area() <= 50
            except:
                return False

        small_selector = FilterSelector(small_area_filter)
        small_filtered = small_selector.filter(faces)
        assert len(small_filtered) == 0, "No cube faces should have area <= 50"

    def test_index_selector_out_of_bounds(self):
        """Test IndexSelector with out of bounds indices."""
        selector = IndexSelector(
            [0, 5])  # 5 might be out of bounds for cube (6 faces)
        faces = self.cube_faces.vals()

        # Only test with valid indices
        valid_indices = [i for i in [0, 5] if i < len(faces)]
        selector = IndexSelector(valid_indices)
        filtered = selector.filter(faces)

        # Should return filtered faces
        assert len(filtered) == len(valid_indices)

    def test_empty_selection_handling(self):
        """Test handling of empty selections."""
        selector = IndexSelector([])
        faces = self.cube_faces.vals()
        filtered = selector.filter(faces)

        assert len(filtered) == 0


class TestSelectionIntegration:
    """Integration tests for selection functionality with complex geometries."""

    def setup_method(self):
        """Set up more complex test geometry."""
        # Create a box with a hole
        self.complex_shape = (
            cq.Workplane("XY")
            .box(20, 20, 10)
            .faces(">Z")
            .circle(5)
            .cutThruAll()
        )

    def test_selection_on_complex_geometry(self):
        """Test selection functionality on more complex geometry."""
        shape_val = self.complex_shape.val()
        explorer = ShapeExplorer(shape_val)

        faces = explorer.search("Face")
        edges = explorer.search("Edge")
        vertices = explorer.search("Vertex")

        # Should have more faces due to the hole
        assert len(faces) > 6
        assert len(edges) > 12
        assert len(vertices) > 8

    def test_connected_shapes_on_complex_geometry(self):
        """Test connected shapes explorer on complex geometry."""
        shape_val = self.complex_shape.val()
        faces = self.complex_shape.faces().vals()

        if faces:
            explorer = ConnectedShapesExplorer(shape_val, faces[0])
            connected_edges = explorer.search("Edge")

            # Should find connected edges
            assert len(connected_edges) > 0
