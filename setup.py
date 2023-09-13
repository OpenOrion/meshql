

from setuptools import setup

setup(
   name='meshql',
   version='1.0.0',
   description='query based meshing on top of GMSH',
   author='Afshawn Lotfi',
   author_email='',
   packages=['meshql', 'meshql.mesh', 'meshql.preprocessing', 'meshql.transactions', 'meshql.utils'],
   install_requires=[
    "numpy",
    "gmsh",
    "ipywidgets==7.6",
    "ipython_genutils",
    "pythreejs==2.4.2",
    "su2fmt @ git+https://github.com/Turbodesigner/su2fmt.git",
    "shapely",
    "scipy",
    "jupyter_cadquery"
   ]
)