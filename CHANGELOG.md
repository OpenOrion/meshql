# Changelog

## Release 2.0.0 (Enhancement/Split Branch)

### Major Refactoring & Architecture Changes
- [x] Complete package restructure with new modular architecture
  - [x] Split core functionality into `meshql/core/` (ql.py, selector.py, transaction.py)
  - [x] Organized GMSH operations into `meshql/gmsh/` module
  - [x] Consolidated mesh operations in `meshql/mesh/` module
  - [x] Enhanced preprocessing capabilities in `meshql/preprocessing/`
- [x] Migrated from setup.py to modern pyproject.toml build system
- [x] Added comprehensive Makefile for build and installation management

### New Features
- [x] **GeometryQL.gmsh() Static Method**: New entry point for creating GMSH-based geometries
- [x] **compute_mesh() Method**: Enhanced mesh generation with integrated transactions and preprocessing
- [x] **Mesh Marker Support**: Load and export meshes with physical group markers
  - [x] Discrete entity creation for marker elements
  - [x] Automatic physical group assignment from markers
  - [x] Marker exclusion from main mesh when loading
- [x] **Enhanced Mesh Loading**: Direct import of meshly meshes into GMSH
- [x] **Ratio-based Geometry Splitting**: Split geometries using ratio parameters
- [x] **Advanced Split Operations**: 
  - [x] `from_plane()`: Split by plane with angle specification
  - [x] `from_ratios()`: Split by ratio values along edges
  - [x] `from_normals()`: Split by normal vectors
  - [x] `from_anchor()`: Split from anchor points
  - [x] `from_edge()`: Split from specific edges
  - [x] `from_lines()`: Split using line definitions
- [x] **Python Notebook Visualizer**: Improved in-notebook mesh visualization with pythreejs
- [x] **print_val() Method**: Debug helper to print Workplane values in GeometryQL and Split

### Improvements
- [x] **Performance Optimizations**:
  - [x] Replaced CQCache utility with new CacheService implementation
  - [x] Direct workplane handling in Split class (no GeometryQL dependency)
  - [x] Improved entity selection and filtering logic
  - [x] Enhanced checksum generation for geometry tracking
  - [x] Multi-level caching system for split operations and region groups
    - [x] BREP caching for split workplane geometry (77% faster on subsequent runs)
    - [x] Region group caching with deterministic checksums across sessions
    - [x] Automatic cache invalidation and updating for nested splits
    - [x] `CacheService.clear_cache()` method for cache management
- [x] **Refactored Entity Management**:
  - [x] Replaced CQEntityContext with CQEntityMapper for better clarity
  - [x] Introduced GeometryQLContext class with enhanced properties
  - [x] Improved selection logic for complex entity filtering
- [x] **Enhanced Preprocessing**:
  - [x] Flexible preprocessing mechanism using (Type, callback) tuples
  - [x] Split class works independently with workplanes
  - [x] Better region group handling and visualization
- [x] **Mesh Visualization Improvements**:
  - [x] Refactored visualize_mesh function for better mesh extraction
  - [x] Enhanced wireframe rendering for various mesh types
  - [x] Simplified surface and volume mesh creation
  - [x] Removed unused coordinate display and buffer mesh code
- [x] **Transaction System**:
  - [x] Migrated from Python dataclasses to Pydantic models
  - [x] Improved transaction context handling
  - [x] Better lookup mechanism for previous transactions
  - [x] Updated import structure for cleaner organization

### Bug Fixes
- [x] Fixed interior/exterior edge classification bug
- [x] Fixed issue with multiple physical groups having the same name
- [x] Corrected wire orientation to ensure consistent winding
- [x] Resolved split operation bugs for 2D geometries
- [x] Fixed `from_ratio()` to work correctly with reversed wires
- [x] Added None check for ql in `end()` method
- [x] Fixed visualizer compatibility with meshly

### Testing & Documentation
- [x] Comprehensive unit test suite added:
  - [x] `tests/selector_test.py`: IndexSelector, FilterSelector, GroupSelector, Selection tests
  - [x] `tests/shapes_test.py`: NACA airfoil, circle generation, sampling tests
  - [x] `tests/split_test.py`: Split operation tests
  - [x] `tests/cq_utils_test.py`: CadQuery utility function tests
  - [x] `tests/entity_test.py`: Entity management tests
  - [x] `tests/mesh_test.py`: Mesh loading and conversion tests
  - [x] `tests/integration_test.py`: End-to-end workflow tests
- [x] Updated example notebooks:
  - [x] `cube.ipynb`: Structured meshing with boundary layers
  - [x] `naca0012.ipynb`: Airfoil meshing examples
  - [x] `inviscid_wedge.ipynb`: Inviscid flow setup
  - [x] `turbo.ipynb`: Turbomachinery with STEP import
  - [x] `progression.ipynb`: Mesh refinement examples
- [x] Added `visualize_example.py` for standalone visualization demos
- [x] Created `scripts/gmsh_host_client.py` for opening and watching .msh files

### Dependencies
- [x] Updated to meshly==1.3.0a0
- [x] Updated to su2fmt@2.0.0
- [x] Added pythreejs==2.4.2 for notebook visualization
- [x] Maintained Python 3.8+ compatibility

### Removed/Deprecated
- [x] Removed setup.py in favor of pyproject.toml
- [x] Removed requirements_dev.txt
- [x] Removed deprecated `exporters.py` and `importers.py`
- [x] Consolidated functionality into `loaders.py`
- [x] Removed old transaction structure from `meshql/transactions/`
- [x] Removed standalone `entity.py`, `transaction.py`, and `ql.py` from root meshql/

---

## Release 1.0.0

- [x] Add su2fmt 3D mesh output
- [x] Add 3D import and export support
- [x] Add proper filtering for specific faces based on diffs (fromTagged support)
- [x] Add 3D boundary layer support
- [x] Add plot visualization for entities including physical groups
- [x] Convert CQ objects to gmsh OCC
- [x] Define point sizes from selectors
- [x] Add multi group tagging in 1 command
- [x] Allow multiple solids to be split
- [x] Add batch operations for things such as transfinite fields
- [x] Fix interior faces for multi-solid meshes
- [x] All in one function that will automatically do structured meshing
    - [x] Adjust the refinement by interior edges
    - [x] Structured boundary layer, Adjust refinement based on high level parms (wall height, etc.)
    - [x] Structured boundary layer for 2D based on num layers
    - [x] Handle >4 edge transfinite faces (get this from partitions) - pretty much pass in edge corners of solid
        - [x] Group edges and set corners and node counts
        - [x] Group faces and set corners
    - [x] Split faces based on partitions, make sure interior entities are updates accordingly
- [x] Preprocessing to auto slice into transfinite faces
    - [x] Auto slice into transfinite faces
    - [x] Applying preprocessing to faces (should work but make sure what's the problem)
    - [x] Caching for slice to be faster after preprocessing
    - [x] Custom cell counts for different transfinite regions
    - [x] Solve issue with boundary condtion on shared edge solids
- [x] For manual setTransiniteFace auto set cell count to all the edges same way as initial auto
- [ ] Documentation and testing

Nice to haves
- [ ] Automatically set the group angle
- [ ] Make visualizer auto-scale initial view and color coded better
