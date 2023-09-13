Release 1.0.0

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
