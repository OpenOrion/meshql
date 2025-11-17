import cadquery as cq
from meshql import GeometryQL, Split
from meshql.utils.cq_cache import CQCache

# clears the cache, comment out for performance
# CQCache.clear_cache()

with GeometryQL.gmsh() as geo:
    ql = (
        geo
        .load(
            (
                cq.Workplane("XY")
                .box(10, 10, 10)
                .rect(2, 2)
                .cutThruAll()
            ),
            on_preprocess=lambda ql: (
                Split(ql)
                .from_plane(angle=(90, 90, 0))
                .from_plane(angle=(-90, 90, 0))
            ),
        )
        .setTransfiniteAuto(max_nodes=300)
        .generate(3)
        # .write("mesh.msh")
        .show("mesh")
    )
