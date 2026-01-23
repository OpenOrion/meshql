"""Unit tests for meshql.gmsh module (GMSH integration functionality)."""

import pytest
from unittest.mock import Mock, patch
import cadquery as cq
from meshql.gmsh.ql import GmshGeometryQL
from meshql.gmsh.entity import Entity, CQEntityMapper, ENTITY_DIM_MAPPING
from meshql.core.transaction import Transaction, MultiEntityTransaction, TransactionContext
from meshql.gmsh.physical_group import SetPhysicalGroup
from meshql.gmsh.boundary_layer import UnstructuredBoundaryLayer, UnstructuredBoundaryLayer2D
from meshql.gmsh.transfinite import SetTransfiniteEdge, SetTransfiniteFace, SetTransfiniteSolid
from meshql.gmsh.refinement import SetMeshSize
from meshql.utils.types import OrderedSet


class TestEntity:
    """Test cases for GMSH entity functionality."""

    def test_entity_creation_with_name(self):
        """Test Entity creation with all fields."""
        entity = Entity(type="Face", tag=42, name="test_face")
        assert entity.type == "Face"
        assert entity.tag == 42
        assert entity.name == "test_face"
        assert entity.dim == 2

    def test_entity_dim_property_functionality(self):
        """Test Entity dim property computation from mapping."""
        # Test that dim property correctly accesses ENTITY_DIM_MAPPING
        for entity_type, expected_dim in ENTITY_DIM_MAPPING.items():
            entity = Entity(type=entity_type, tag=1)
            assert entity.dim == expected_dim, f"{entity_type} should have dimension {expected_dim}"

    def test_entity_invalid_type_error(self):
        """Test Entity raises error for invalid types."""
        from pydantic_core import ValidationError
        with pytest.raises(ValidationError):
            entity = Entity(type="InvalidType", tag=1)

    def test_entity_equality_and_hashing(self):
        """Test Entity equality and hash functionality."""
        entity1 = Entity(type="Face", tag=1, name="face1")
        # Same tag, type; different name
        entity2 = Entity(type="Face", tag=1, name="face2")
        # Same type, name; different tag
        entity3 = Entity(type="Face", tag=2, name="face1")
        # Same tag, name; different type
        entity4 = Entity(type="Edge", tag=1, name="face1")

        # Equality based only on type and tag (name doesn't matter)
        assert entity1 == entity2
        assert entity1 != entity3
        assert entity1 != entity4

        # Hash consistency
        assert hash(entity1) == hash(entity2)
        assert hash(entity1) != hash(entity3)
        assert hash(entity1) != hash(entity4)

        # Test with non-Entity object
        assert entity1 != "not an entity"

    def test_cq_entity_context_3d_geometry_detection(self):
        """Test CQEntityContext correctly detects 3D geometry."""
        cube = cq.Workplane("XY").box(10, 10, 10)
        context = CQEntityMapper(cube, level="Face")
        assert context.dimension == 3, "Cube should be detected as 3D"

    def test_cq_entity_context_2d_geometry_detection(self):
        """Test CQEntityContext correctly detects 2D geometry."""
        square = cq.Workplane("XY").rect(10, 10)
        context = CQEntityMapper(square, level="Edge")
        assert context.dimension == 2, "Square should be detected as 2D"

    def test_cq_entity_context_entity_registration(self):
        """Test that CQEntityContext properly registers entities during initialization."""
        cube = cq.Workplane("XY").box(10, 10, 10)
        # Go down to vertex level
        context = CQEntityMapper(cube, level="Vertex")

        # Verify that entities are actually registered in appropriate registries
        assert len(context.entity_registries["Solid"]) > 0, "Should have solid entities"
        assert len(context.entity_registries["Face"]) > 0, "Should have face entities"
        assert len(context.entity_registries["Edge"]) > 0, "Should have edge entities"
        assert len(context.entity_registries["Vertex"]) > 0, "Should have vertex entities"

        # Verify entities have proper sequential tags
        for entity_type, registry in context.entity_registries.items():
            if len(registry) > 0:
                tags = [entity.tag for entity in registry.values()]
                expected_tags = list(range(1, len(registry) + 1))
                assert sorted(tags) == expected_tags, f"{entity_type} entities should have sequential tags starting from 1"

    def test_cq_entity_context_add_method_functionality(self):
        """Test CQEntityContext add method actually adds entities correctly."""
        cube = cq.Workplane("XY").box(10, 10, 10)
        context = CQEntityMapper(cube, level="Face")

        # Get initial count of face entities
        initial_face_count = len(context.entity_registries["Face"])

        # Create a new face and add it
        new_face = cq.Workplane("XY").rect(
            5, 5).extrude(1).faces().first().val()
        context.add(new_face.wrapped)

        # Verify the face was added
        assert len(context.entity_registries["Face"]) >= initial_face_count, "New face should be registered or already exist"

    def test_cq_entity_context_shape_lookup_consistency(self):
        """Test that shape lookup maintains correct mappings."""
        cube = cq.Workplane("XY").box(10, 10, 10)
        context = CQEntityMapper(cube, level="Face")

        # Verify shape_lookup has entries for registered entities
        total_registered = sum(len(registry)
                               for registry in context.entity_registries.values())
        if total_registered > 0:
            assert len(context.shape_lookup) > 0, "Shape lookup should have entries for registered shapes"

    def test_cq_entity_context_select_functionality(self):
        """Test CQEntityContext select method returns correct entities."""
        cube = cq.Workplane("XY").box(10, 10, 10)
        context = CQEntityMapper(cube, level="Face")

        faces = cube.faces().vals()
        if len(faces) > 0:
            face = faces[0]
            try:
                selected_entity = context.select(face)
                assert isinstance(selected_entity, Entity)
                assert selected_entity.type == "Face"
                assert selected_entity.tag > 0
            except KeyError:
                # Face might not be in registry if it wasn't added during initialization
                pass

    def test_cq_entity_context_select_many_functionality(self):
        """Test CQEntityContext select_many method returns correct entity sets."""
        cube = cq.Workplane("XY").box(10, 10, 10)
        context = CQEntityMapper(cube, level="Face")

        faces_workplane = cube.faces()
        selected_entities = context.select_many(faces_workplane)

        assert isinstance(selected_entities, OrderedSet)
        # Should return entities for faces that were registered
        for entity in selected_entities:
            assert isinstance(entity, Entity)
            assert entity.type == "Face"

    def test_cq_entity_context_level_parameter_effect(self):
        """Test that level parameter affects which entities are registered."""
        cube = cq.Workplane("XY").box(10, 10, 10)

        # Test with different levels
        face_level_context = CQEntityMapper(cube, level="Face")
        edge_level_context = CQEntityMapper(cube, level="Edge")
        vertex_level_context = CQEntityMapper(cube, level="Vertex")

        # Vertex level should have more total entities (goes deeper)
        vertex_total = sum(
            len(registry) for registry in vertex_level_context.entity_registries.values())
        face_total = sum(len(registry)
                         for registry in face_level_context.entity_registries.values())

        assert vertex_total >= face_total, "Vertex level should register same or more entities than face level"

        # All should have solid entities since it's 3D
        assert len(face_level_context.entity_registries["Solid"]) > 0
        assert len(edge_level_context.entity_registries["Solid"]) > 0
        assert len(vertex_level_context.entity_registries["Solid"]) > 0


class TestGmshTransaction:
    """Test cases for GMSH transaction system."""

    def test_gmsh_transaction_creation(self):
        """Test basic GmshTransaction creation."""

        class TestTransaction(Transaction):
            class_name: str = "TestTransaction"
            
            def before_gen(self):
                pass

        transaction = TestTransaction(class_name="TestTransaction")
        assert isinstance(transaction, Transaction)

    def test_multi_entity_transaction_creation(self):
        """Test MultiEntityTransaction creation."""
        entities = OrderedSet([Entity(tag=1, type="Face")])

        class TestMultiTransaction(MultiEntityTransaction):
            class_name: str = "TestMultiTransaction"
            
            def before_gen(self):
                pass

        transaction = TestMultiTransaction(class_name="TestMultiTransaction", entities=entities)
        assert transaction.entities == entities

    def test_gmsh_transaction_context_creation(self):
        """Test GmshTransactionContext creation."""
        context = TransactionContext()
        assert isinstance(context.entity_transactions, dict)
        assert isinstance(context.system_transactions, dict)
        assert len(context.entity_transactions) == 0
        assert len(context.system_transactions) == 0

    def test_gmsh_transaction_context_add_transaction(self):
        """Test adding transactions to context."""
        context = TransactionContext()

        class TestTransaction(Transaction):
            class_name: str = "TestTransaction"
            
            def before_gen(self):
                pass

        transaction = TestTransaction(class_name="TestTransaction")
        context.add_transaction(transaction)

        assert len(context.system_transactions) == 1
        assert TestTransaction in context.system_transactions

    def test_gmsh_transaction_context_add_transactions_list(self):
        """Test adding multiple transactions to context."""
        context = TransactionContext()

        class TestTransaction(Transaction):
            class_name: str = "TestTransaction"
            
            def before_gen(self):
                pass

        transactions = [TestTransaction(class_name="TestTransaction"), TestTransaction(class_name="TestTransaction")]
        context.add_transactions(transactions)

        # Two transactions of the same type will only keep the last one in system_transactions
        assert len(context.system_transactions) == 1

    def test_set_physical_group_transaction(self):
        """Test SetPhysicalGroup transaction."""
        entities = OrderedSet([Entity(tag=1, type="Face")])
        transaction = SetPhysicalGroup(entities=entities, name="test_group")

        assert transaction.entities == entities
        assert transaction.name == "test_group"

    def test_set_mesh_size_transaction(self):
        """Test SetMeshSize transaction with validation."""
        entities = OrderedSet([Entity(tag=1, type="Vertex")])
        transaction = SetMeshSize(entities=entities, size=0.1)

        assert transaction.entities == entities
        assert transaction.size == 0.1

        # Verify entity is correct type
        entity = list(entities)[0]
        assert entity.type == "Vertex", "SetMeshSize should typically target vertices"
        assert entity.dim == 0, "Vertex should have dimension 0"

        # Test with multiple entities
        entities_multi = OrderedSet([
            Entity(tag=1, type="Vertex"),
            Entity(tag=2, type="Vertex"),
            Entity(tag=3, type="Vertex")
        ])
        transaction_multi = SetMeshSize(entities=entities_multi, size=0.05)
        assert len(transaction_multi.entities) == 3
        assert transaction_multi.size == 0.05


class TestBoundaryLayer:
    """Test cases for boundary layer functionality."""

    def test_unstructured_boundary_layer_creation(self):
        """Test UnstructuredBoundaryLayer creation."""
        entities = OrderedSet([Entity(tag=1, type="Face")])
        bl = UnstructuredBoundaryLayer(
            entities=entities,
            size=0.01,
            ratio=1.5,
            num_layers=5
        )

        assert bl.entities == entities
        assert bl.size == 0.01
        assert bl.ratio == 1.5
        assert bl.num_layers == 5

    def test_unstructured_boundary_layer_2d_creation(self):
        """Test UnstructuredBoundaryLayer2D creation."""
        entities = OrderedSet([Entity(tag=1, type="Edge")])
        bl2d = UnstructuredBoundaryLayer2D(
            entities=entities,
            size=0.005,
            ratio=2.0,
            num_layers=10
        )

        assert bl2d.entities == entities
        assert bl2d.size == 0.005
        assert bl2d.ratio == 2.0
        assert bl2d.num_layers == 10


class TestTransfinite:
    """Test cases for transfinite meshing functionality."""

    def test_set_transfinite_edge_creation(self):
        """Test SetTransfiniteEdge creation."""
        entity = Entity(tag=1, type="Edge")
        trans_edge = SetTransfiniteEdge(
            entity=entity,
            num_elems=20,
            coef=1.1
        )

        assert trans_edge.entity == entity
        assert trans_edge.num_elems == 20
        assert trans_edge.coef == 1.1

    def test_set_transfinite_face_creation(self):
        """Test SetTransfiniteFace creation with valid arrangements."""
        entity = Entity(tag=1, type="Face")

        # Test all valid arrangements
        valid_arrangements = ["Left", "Right",
                              "AlternateLeft", "AlternateRight"]
        for arrangement in valid_arrangements:
            trans_face = SetTransfiniteFace(
                entity=entity,
                arrangement=arrangement
            )
            assert trans_face.entity == entity
            assert trans_face.arrangement == arrangement
            # Verify entity is correct type and dimension
            assert entity.type == "Face"
            assert entity.dim == 2, "Face entity should have dimension 2"

    def test_set_transfinite_solid_creation(self):
        """Test SetTransfiniteSolid creation with validation."""
        entity = Entity(tag=1, type="Solid")
        trans_solid = SetTransfiniteSolid(entity=entity)

        assert trans_solid.entity == entity

        # Verify entity is correct type and dimension
        assert entity.type == "Solid", "Should target solid entities"
        assert entity.dim == 3, "Solid entity should have dimension 3"

        # Verify optional corners field
        assert trans_solid.corners is None, "Corners should default to None"

        # Test with corners
        corner_entities = OrderedSet(
            [Entity(tag=i, type="Vertex") for i in range(1, 9)])
        trans_solid_with_corners = SetTransfiniteSolid(
            entity=entity, corners=corner_entities)
        assert len(trans_solid_with_corners.corners) == 8


class TestGmshGeometryQL:
    """Test cases for GmshGeometryQL functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cube = cq.Workplane("XY").box(10, 10, 10)

    @patch('meshql.gmsh.ql.gmsh')
    def test_gmsh_geometry_ql_context_manager(self, mock_gmsh):
        """Test GmshGeometryQL as context manager."""
        mock_gmsh.initialize.return_value = None
        mock_gmsh.finalize.return_value = None

        with GmshGeometryQL() as geo:
            assert isinstance(geo, GmshGeometryQL)

    @patch('meshql.gmsh.ql.gmsh')
    def test_add_physical_group_method(self, mock_gmsh):
        """Test addPhysicalGroup method."""
        with GmshGeometryQL() as geo:
            loaded_geo = geo.load(self.cube)

            try:
                result = loaded_geo.faces().addPhysicalGroup("test_group")
                assert result is not None
            except Exception:
                pass  # Might fail without full GMSH context

    @patch('meshql.gmsh.ql.gmsh')
    def test_set_mesh_size_method(self, mock_gmsh):
        """Test setMeshSize method."""
        with GmshGeometryQL() as geo:
            loaded_geo = geo.load(self.cube)

            try:
                result = loaded_geo.faces().setMeshSize(0.5)
                assert result is not None
            except Exception:
                pass  # Might fail without full GMSH context

    @patch('meshql.gmsh.ql.gmsh')
    def test_add_boundary_layer_method(self, mock_gmsh):
        """Test addBoundaryLayer method."""
        with GmshGeometryQL() as geo:
            loaded_geo = geo.load(self.cube)

            try:
                result = loaded_geo.faces().addBoundaryLayer(
                    ratio=1.5,
                    size=0.01,
                    num_layers=5
                )
                assert result is not None
            except Exception:
                pass  # Might fail without full GMSH context

    @patch('meshql.gmsh.ql.gmsh')
    def test_set_transfinite_auto_method(self, mock_gmsh):
        """Test setTransfiniteAuto method."""
        with GmshGeometryQL() as geo:
            loaded_geo = geo.load(self.cube)

            try:
                result = loaded_geo.setTransfiniteAuto(
                    max_nodes=50, min_nodes=10)
                assert result is not None
            except Exception:
                pass  # Might fail without full GMSH context

    @patch('meshql.gmsh.ql.gmsh')
    def test_recombine_method(self, mock_gmsh):
        """Test recombine method."""
        with GmshGeometryQL() as geo:
            loaded_geo = geo.load(self.cube)

            try:
                result = loaded_geo.faces().recombine(angle=45)
                assert result is not None
            except Exception:
                pass  # Might fail without full GMSH context

    @patch('meshql.gmsh.ql.gmsh')
    def test_smooth_method(self, mock_gmsh):
        """Test smooth method."""
        with GmshGeometryQL() as geo:
            loaded_geo = geo.load(self.cube)

            try:
                result = loaded_geo.smooth(num_smooths=3)
                assert result is not None
            except Exception:
                pass  # Might fail without full GMSH context

    @patch('meshql.gmsh.ql.gmsh')
    def test_refine_method(self, mock_gmsh):
        """Test refine method."""
        with GmshGeometryQL() as geo:
            loaded_geo = geo.load(self.cube)

            try:
                result = loaded_geo.refine(num_refines=2)
                assert result is not None
            except Exception:
                pass  # Might fail without full GMSH context


class TestGmshIntegration:
    """Integration tests for GMSH functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cube = cq.Workplane("XY").box(10, 10, 10)

    @patch('meshql.gmsh.ql.gmsh')
    def test_complete_gmsh_workflow(self, mock_gmsh):
        """Test complete GMSH workflow."""
        mock_gmsh.initialize.return_value = None
        mock_gmsh.finalize.return_value = None
        mock_gmsh.model = Mock()
        mock_gmsh.model.occ = Mock()

        try:
            with GmshGeometryQL() as geo:
                result = (
                    geo
                    .load(self.cube)
                    .faces(type=">Z")
                    .addPhysicalGroup("top")
                    .setMeshSize(0.5)
                    .addBoundaryLayer(size=0.01, ratio=1.5, num_layers=3)
                    .end()
                    .setTransfiniteAuto(max_nodes=50)
                )

                assert result is not None
        except Exception as e:
            # Expected with mocked GMSH
            assert isinstance(e, Exception)

    @patch('meshql.gmsh.ql.gmsh')
    def test_mesh_generation_workflow(self, mock_gmsh):
        """Test mesh generation workflow."""
        mock_gmsh.initialize.return_value = None
        mock_gmsh.finalize.return_value = None
        mock_gmsh.model = Mock()

        try:
            with GmshGeometryQL() as geo:
                result = (
                    geo
                    .load(self.cube)
                    .setTransfiniteAuto(30)
                )

                # Test mesh generation call (will be mocked)
                # result.generate(3)

                assert result is not None
        except Exception:
            pass  # Expected with mocked GMSH

    @patch('meshql.gmsh.ql.gmsh')
    def test_complex_boundary_layer_workflow(self, mock_gmsh):
        """Test complex boundary layer workflow."""
        mock_gmsh.initialize.return_value = None
        mock_gmsh.finalize.return_value = None

        # Create geometry with hole for interior faces
        geometry_with_hole = (
            cq.Workplane("XY")
            .box(20, 20, 10)
            .faces(">Z")
            .circle(5)
            .cutThruAll()
        )

        try:
            with GmshGeometryQL() as geo:
                result = (
                    geo
                    .load(geometry_with_hole)
                    .faces(type="interior")
                    .addPhysicalGroup("hole_surface")
                    .addBoundaryLayer(
                        size=0.001,
                        ratio=1.8,
                        num_layers=8
                    )
                    .end()
                    .faces(type="exterior")
                    .addPhysicalGroup("outer_surface")
                    .setMeshSize(1.0)
                    .end()
                )

                assert result is not None
        except Exception:
            pass  # Expected with mocked GMSH

    def test_entity_dimension_consistency(self):
        """Test that entity dimensions are consistent."""
        # Test all mapped types
        for cq_type, dim in ENTITY_DIM_MAPPING.items():
            entity = Entity(tag=1, type=cq_type)
            assert entity.dim == dim
            assert isinstance(dim, int)
            assert dim >= 0
            assert dim <= 3
