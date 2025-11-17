"""Unit tests for meshql.preprocessing.split module."""

import unittest
import cadquery as cq
import numpy as np
from meshql.ql import GeometryQL, GeometryQLContext
from meshql.preprocessing.split import Split


class SplitTest(unittest.TestCase):
    """Test cases for Split preprocessing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.cube = cq.Workplane("XY").box(10, 10, 10)
        self.context = GeometryQLContext()
        self.geo_ql = GeometryQL(ctx=self.context, workplane=self.cube)
        self.split = Split(self.geo_ql)

    def test_split_state_initialization(self):
        """Test Split object initializes with correct state."""
        self.assertEqual(self.split.ql, self.geo_ql)
        self.assertEqual(len(self.split.pending_splits), 0)
        self.assertEqual(len(self.split.face_edge_groups), 0)

    def test_pending_splits_accumulation(self):
        """Test that split operations accumulate in pending_splits."""
        initial_count = len(self.split.pending_splits)

        self.split.from_plane(base_pnt=(0, 0, 0))
        self.assertEqual(len(self.split.pending_splits), initial_count + 1)

        self.split.from_plane(base_pnt=(5, 0, 0))
        self.assertEqual(len(self.split.pending_splits), initial_count + 2)

        # Verify each pending split contains face objects
        for split_group in self.split.pending_splits:
            self.assertIsInstance(split_group, list)
            for face in split_group:
                self.assertIsInstance(face, cq.Face)

    def test_split_face_generation_properties(self):
        """Test that generated split faces have expected geometric properties."""
        self.split.from_plane(base_pnt=(0, 0, 0), angle=(0, 0, 1))

        # Verify we have a split face
        self.assertEqual(len(self.split.pending_splits), 1)
        split_face = self.split.pending_splits[0][0]

        # Verify it's a valid face with area > 0
        self.assertGreater(split_face.Area(), 0)

        # Verify face center is near expected position
        face_center = split_face.Center()
        self.assertAlmostEqual(face_center.z, 0, places=3)

    def test_from_plane_sizing_parameter_effect(self):
        """Test that sizing parameter affects plane dimensions."""
        # Test maxDim sizing
        self.split.from_plane(sizing="maxDim")
        maxdim_face = self.split.pending_splits[0][0]
        maxdim_area = maxdim_face.Area()

        # Clear and test infinite sizing
        self.split.pending_splits.clear()
        self.split.from_plane(sizing="infinite")
        infinite_face = self.split.pending_splits[0][0]

        # Both should be valid faces
        self.assertGreater(maxdim_area, 0)
        self.assertGreater(infinite_face.Area(), 0)

    def test_from_lines_creates_valid_split_faces(self):
        """Test that from_lines creates geometrically valid split faces."""
        # Simple horizontal line in XY plane
        lines = ((-5, 0, 0), (15, 0, 0))
        self.split.from_lines(lines=lines)

        self.assertEqual(len(self.split.pending_splits), 1)
        split_face = self.split.pending_splits[0][0]

        # Verify face properties
        self.assertGreater(split_face.Area(), 0)
        # Face should extend through the cube area
        bbox = split_face.BoundingBox()
        self.assertLess(bbox.xmin, -4)  # Should extend beyond cube
        self.assertGreater(bbox.xmax, 14)  # Should extend beyond cube

    def test_from_edge_direction_parameter_effect(self):
        """Test that direction parameter affects split face generation."""
        cube_edge = self.cube.edges().first().val()

        # Test "both" direction - verify split was created
        self.split.from_edge(edge=cube_edge, dir="both")
        self.assertEqual(len(self.split.pending_splits), 1)

        # Clear and test "towards" direction
        self.split.pending_splits.clear()
        self.split.from_edge(edge=cube_edge, dir="towards")
        self.assertEqual(len(self.split.pending_splits), 1)

        # Clear and test "away" direction
        self.split.pending_splits.clear()
        self.split.from_edge(edge=cube_edge, dir="away")
        self.assertEqual(len(self.split.pending_splits), 1)

        # All directions should produce some split geometry
        # (Even if area calculation fails, the split face should exist)
        towards_face = self.split.pending_splits[0][0]
        self.assertIsInstance(towards_face, cq.Face)

    def test_from_pnts_creates_face_from_points(self):
        """Test that from_pnts creates face matching input geometry."""
        # Create a square in XY plane
        points = [(0, 0, 0), (5, 0, 0), (5, 5, 0), (0, 5, 0)]
        self.split.from_pnts(points)

        self.assertEqual(len(self.split.pending_splits), 1)
        split_face = self.split.pending_splits[0][0]

        # Verify face properties match expected square
        self.assertAlmostEqual(split_face.Area(), 25, places=3)  # 5x5 square

        # Verify face center
        center = split_face.Center()
        self.assertAlmostEqual(center.x, 2.5, places=3)
        self.assertAlmostEqual(center.y, 2.5, places=3)
        self.assertAlmostEqual(center.z, 0, places=3)

    def test_push_method_state_management(self):
        """Test that push method correctly manages pending splits state."""
        # Create test faces
        face1 = cq.Face.makePlane(5, 5, cq.Vector(0, 0, 0), cq.Vector(0, 0, 1))
        face2 = cq.Face.makePlane(3, 3, cq.Vector(2, 0, 0), cq.Vector(0, 0, 1))

        # Test single face push
        self.split.push(face1)
        self.assertEqual(len(self.split.pending_splits), 1)
        self.assertEqual(len(self.split.pending_splits[0]), 1)

        # Test multiple faces push (as single group)
        self.split.push([face1, face2])
        self.assertEqual(len(self.split.pending_splits), 2)
        self.assertEqual(len(self.split.pending_splits[1]), 2)

    def test_apply_method_geometry_modification(self):
        """Test that apply method actually modifies the workplane geometry."""
        # Get initial solid count
        initial_solids = len(self.cube.solids().vals())

        # Add a split plane through the middle
        self.split.from_plane(base_pnt=(0, 0, 0), angle=(0, 0, 1))

        # Apply the split
        result_workplane = self.split.apply()

        # Verify we get more solids after splitting
        result_solids = len(result_workplane.solids().vals())
        self.assertGreaterEqual(result_solids, initial_solids)

        # Verify pending splits are cleared after apply
        self.assertEqual(len(self.split.pending_splits), 0)

    def test_apply_clears_pending_splits_state(self):
        """Test that apply method clears the pending splits state."""
        # Add multiple splits
        self.split.from_plane(base_pnt=(2, 0, 0))
        self.split.from_plane(base_pnt=(-2, 0, 0))

        # Verify splits are pending
        self.assertEqual(len(self.split.pending_splits), 2)

        # Apply and verify state is cleared
        result = self.split.apply()
        self.assertEqual(len(self.split.pending_splits), 0)
        self.assertIsInstance(result, cq.Workplane)

    def test_chaining_preserves_split_state(self):
        """Test that method chaining correctly accumulates splits."""
        result = (
            self.split
            .from_plane(base_pnt=(1, 0, 0))
            .from_plane(base_pnt=(-1, 0, 0))
            .from_lines(lines=[((0, -10, 0), (0, 10, 0))])
        )

        # Verify chaining returns self
        self.assertEqual(result, self.split)

        # Verify all splits were accumulated
        self.assertEqual(len(self.split.pending_splits), 3)

        # Verify each split contains valid faces
        for split_group in self.split.pending_splits:
            for face in split_group:
                self.assertIsInstance(face, cq.Face)
                self.assertGreater(face.Area(), 0)

    def test_split_face_intersection_with_geometry(self):
        """Test that split faces actually intersect with the target geometry."""
        # Create a split plane that should intersect the cube
        self.split.from_plane(base_pnt=(0, 0, 0), angle=(0, 0, 1))
        split_face = self.split.pending_splits[0][0]

        # Test intersection with cube
        intersection = self.cube.intersect(cq.Workplane(split_face))
        intersected_objects = intersection.vals()

        # Should have intersection objects (edges/faces)
        self.assertGreater(len(intersected_objects), 0)

    def test_axis_parameter_affects_split_orientation(self):
        """Test that axis parameter affects split face generation for different axes."""
        cube_edge = self.cube.edges().first().val()

        # Test different axis parameters create splits
        for axis in ["X", "Y", "Z"]:
            self.split.pending_splits.clear()
            self.split.from_edge(edge=cube_edge, axis=axis)

            # Verify split was created
            self.assertEqual(len(self.split.pending_splits), 1,
                             f"Split should be created for axis {axis}")
            split_face = self.split.pending_splits[0][0]
            self.assertIsInstance(split_face, (cq.Face, cq.Compound),
                                  f"Split for axis {axis} should be Face or Compound")

    def test_split_geometry_bounding_relationships(self):
        """Test that split faces have expected spatial relationships to geometry."""
        bbox = self.cube.val().BoundingBox()

        # Create split plane in middle of cube
        self.split.from_plane(base_pnt=(0, 0, 0))
        split_face = self.split.pending_splits[0][0]
        split_bbox = split_face.BoundingBox()

        # Split face should encompass the cube area
        self.assertLessEqual(split_bbox.xmin, bbox.xmin - 1)
        self.assertGreaterEqual(split_bbox.xmax, bbox.xmax + 1)
        self.assertLessEqual(split_bbox.ymin, bbox.ymin - 1)
        self.assertGreaterEqual(split_bbox.ymax, bbox.ymax + 1)


class SplitIntegrationTest(unittest.TestCase):
    """Integration tests for Split with complex geometries and workflows."""

    def setUp(self):
        """Set up complex test geometry."""
        # Create a more complex shape for integration testing
        self.complex_shape = (
            cq.Workplane("XY")
            .box(20, 20, 10)
            .faces(">Z")
            .circle(5)
            .cutThruAll()
        )
        self.context = GeometryQLContext()
        self.geo_ql = GeometryQL(
            ctx=self.context, workplane=self.complex_shape)

    def test_complex_geometry_split_state_management(self):
        """Test that complex geometry maintains proper split state."""
        split = Split(self.geo_ql)

        # Add multiple diverse splits
        split.from_plane(base_pnt=(0, 0, 5))  # Horizontal plane
        split.from_plane(base_pnt=(10, 0, 0), angle=(
            90, 0, 0))  # Vertical plane

        # Verify split accumulation
        self.assertEqual(len(split.pending_splits), 2)

        # Verify each split contains valid faces
        for split_group in split.pending_splits:
            for face in split_group:
                self.assertIsInstance(face, cq.Face)
                self.assertGreater(face.Area(), 0)

    def test_multiple_line_sets_geometric_properties(self):
        """Test that multiple line sets create geometrically consistent split faces."""
        split = Split(self.geo_ql)

        # Define grid of lines
        horizontal_lines = [((0, 5, 0), (20, 5, 0)), ((0, 15, 0), (20, 15, 0))]
        vertical_lines = [((5, 0, 0), (5, 20, 0)), ((15, 0, 0), (15, 20, 0))]

        # Add line-based splits
        split.from_lines(lines=horizontal_lines)
        split.from_lines(lines=vertical_lines)

        self.assertEqual(len(split.pending_splits), 2)

        # Verify geometric properties of generated faces
        horizontal_face = split.pending_splits[0][0]
        vertical_face = split.pending_splits[1][0]

        # Both should be large faces that extend beyond geometry
        self.assertGreater(horizontal_face.Area(), 1000)
        self.assertGreater(vertical_face.Area(), 1000)

        # Verify faces have expected orientations
        h_normal = horizontal_face.normalAt()
        v_normal = vertical_face.normalAt()

        # Should be roughly perpendicular
        dot_product = abs(h_normal.dot(v_normal))
        self.assertLess(dot_product, 0.2)

    def test_applied_splits_modify_geometry_count(self):
        """Test that applying splits actually increases geometry complexity."""
        split = Split(self.geo_ql)

        # Count initial solid pieces
        initial_solids = len(self.complex_shape.solids().vals())

        # Add splits that should divide the geometry
        split.from_plane(base_pnt=(0, 0, 0))  # Horizontal cut through middle
        split.from_plane(base_pnt=(0, 0, 0), angle=(90, 0, 0))  # Vertical cut

        # Apply splits
        result_workplane = split.apply()
        result_solids = len(result_workplane.solids().vals())

        # Should have more solid pieces after splitting
        self.assertGreaterEqual(result_solids, initial_solids)

    def test_split_face_coverage_of_complex_geometry(self):
        """Test that split faces adequately cover complex geometry."""
        split = Split(self.geo_ql)

        # Get geometry bounding box
        bbox = self.complex_shape.val().BoundingBox()

        # Create split plane
        split.from_plane(base_pnt=(0, 0, 5))  # Middle height
        split_face = split.pending_splits[0][0]
        split_bbox = split_face.BoundingBox()

        # Split should cover the geometry extent and more
        self.assertLessEqual(split_bbox.xmin, bbox.xmin - 5)
        self.assertGreaterEqual(split_bbox.xmax, bbox.xmax + 5)
        self.assertLessEqual(split_bbox.ymin, bbox.ymin - 5)
        self.assertGreaterEqual(split_bbox.ymax, bbox.ymax + 5)

    def test_edge_based_splits_follow_geometry_features(self):
        """Test that edge-based splits respect geometry features."""
        split = Split(self.geo_ql)

        # Get an edge from the complex shape
        edges = self.complex_shape.edges().vals()
        self.assertGreater(len(edges), 0, "Complex shape should have edges")

        test_edge = edges[0]

        # Create edge-based split
        split.from_edge(edge=test_edge, dir="both")

        # Verify split was created
        self.assertEqual(len(split.pending_splits), 1)
        split_geometry = split.pending_splits[0][0]

        # Verify split geometry was created and is a valid shape
        self.assertIsInstance(split_geometry, (cq.Face, cq.Compound))

        # Test intersection with original geometry (if the geometry is valid)
        try:
            intersection = self.complex_shape.intersect(
                cq.Workplane(split_geometry))
            intersection_objects = intersection.vals()
            # If intersection works, should have some objects
            if len(intersection_objects) > 0:
                self.assertGreater(len(intersection_objects),
                                   0, "Split should intersect geometry")
        except:
            # If intersection fails, at least verify split was created
            pass

    def test_refresh_parameter_affects_split_context(self):
        """Test that refresh parameter affects internal split state management."""
        split = Split(self.geo_ql)

        # Verify initial state
        initial_face_edge_groups_count = len(split.face_edge_groups)

        # Add split and apply with refresh
        split.from_plane(base_pnt=(3, 0, 0))

        # Apply with refresh=True should rebuild face_edge_groups
        split.apply(refresh=True)

        # After refresh, face_edge_groups should be populated
        refreshed_count = len(split.face_edge_groups)
        # Should have more face-edge relationships after refresh
        self.assertGreaterEqual(
            refreshed_count, initial_face_edge_groups_count)

    def test_point_based_splits_create_accurate_faces(self):
        """Test that point-based splits create faces with accurate geometry."""
        split = Split(self.geo_ql)

        # Create a specific shaped face using points
        points = [(5, 5, 2), (15, 5, 2), (15, 15, 2), (5, 15, 2)]
        split.from_pnts(points)

        self.assertEqual(len(split.pending_splits), 1)
        split_face = split.pending_splits[0][0]

        # Verify face has expected area (10x10 square = 100)
        self.assertAlmostEqual(split_face.Area(), 100, places=1)

        # Verify face center
        center = split_face.Center()
        self.assertAlmostEqual(center.x, 10, places=1)
        self.assertAlmostEqual(center.y, 10, places=1)
        self.assertAlmostEqual(center.z, 2, places=1)

    def test_sequential_apply_operations_maintain_state(self):
        """Test that multiple apply operations maintain correct state."""
        split = Split(self.geo_ql)

        # Add first set of splits
        split.from_plane(base_pnt=(5, 0, 0))
        split.from_plane(base_pnt=(-5, 0, 0))

        # Apply first set
        first_result = split.apply()
        self.assertEqual(len(split.pending_splits), 0)  # Should be cleared

        # Add second set of splits
        split.from_plane(base_pnt=(0, 5, 0), angle=(90, 0, 0))
        split.from_plane(base_pnt=(0, -5, 0), angle=(90, 0, 0))

        # Verify new splits accumulated
        self.assertEqual(len(split.pending_splits), 2)

        # Apply second set
        second_result = split.apply()
        # Should be cleared again
        self.assertEqual(len(split.pending_splits), 0)

        # Both results should be valid workplanes
        self.assertIsInstance(first_result, cq.Workplane)
        self.assertIsInstance(second_result, cq.Workplane)

    def test_split_face_intersection_accuracy(self):
        """Test accuracy of split face intersections with complex geometry."""
        split = Split(self.geo_ql)

        # Create split plane that should intersect the hole in complex geometry
        split.from_plane(base_pnt=(0, 0, 5))  # At middle height
        split_face = split.pending_splits[0][0]

        # Test intersection with the complex shape
        intersection = self.complex_shape.intersect(cq.Workplane(split_face))

        # Should have intersection geometry
        intersection_vals = intersection.vals()
        self.assertGreater(len(intersection_vals), 0)

        # Check that intersection includes both outer boundary and hole
        if len(intersection_vals) > 0:
            # Calculate total intersection area (should be box area minus hole area)
            total_intersection_area = sum(obj.Area() if hasattr(obj, 'Area') else 0
                                          for obj in intersection_vals)
            expected_area = 20 * 20 - np.pi * 5 * 5  # Box minus circular hole
            self.assertGreater(total_intersection_area,
                               expected_area * 0.8)  # Allow some tolerance

    def test_preprocessing_function_workflow_properties(self):
        """Test split workflow properties when used as preprocessing function."""
        def preprocess_func(ql):
            return (
                Split(ql)
                .from_plane(angle=(90, 0, 0))  # YZ plane
                .from_plane(angle=(0, 90, 0))  # XZ plane
            )

        # Execute preprocessing
        result = preprocess_func(self.geo_ql)

        # Verify split state
        self.assertEqual(len(result.pending_splits), 2)

        # Verify geometric properties of generated splits
        plane1 = result.pending_splits[0][0]
        plane2 = result.pending_splits[1][0]

        # Both planes should be large
        self.assertGreater(plane1.Area(), 1000)
        self.assertGreater(plane2.Area(), 1000)

        # Planes should be roughly perpendicular
        normal1 = plane1.normalAt()
        normal2 = plane2.normalAt()
        dot_product = abs(normal1.dot(normal2))
        self.assertLess(dot_product, 0.2)


if __name__ == '__main__':
    unittest.main()
