

from setuptools import setup

setup(
   name='meshql',
   version='1.1.0',
   description='query based meshing on top of GMSH',
   author='Afshawn Lotfi',
   author_email='',
   packages=['meshql', 'meshql.mesh', 'meshql.preprocessing', 'meshql.gmsh', 'meshql.utils'],
   install_requires=[
    "numpy",
    "gmsh",
    "ipywidgets",
    "ipython_genutils",
    "pythreejs==2.4.2",
    "su2fmt @ git+https://github.com/OpenOrion/su2fmt.git",
    "shapely",
    "scipy",
    "jupyter_cadquery @ git+https://github.com/bernhard-42/jupyter-cadquery.git",
    "plotly",
   ]
)