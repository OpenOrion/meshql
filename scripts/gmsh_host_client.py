# Install dependencies with:
#   pip install gmsh watchfiles

import argparse
import sys
import os
import gmsh
from watchfiles import watch
import multiprocessing

def open_in_gmsh_python(msh_file):
    gmsh.initialize()
    gmsh.open(msh_file)
    gmsh.fltk.run()
    gmsh.finalize()

def gmsh_process(msh_file):
    open_in_gmsh_python(msh_file)

def main():
    parser = argparse.ArgumentParser(description="Open a .msh file in Gmsh FLTK window and reload on changes.")
    parser.add_argument('msh_file', nargs='?', default='output.msh', help="Path to the .msh file (default: output.msh)")
    args = parser.parse_args()
    msh_file = args.msh_file

    if not os.path.isfile(msh_file):
        print(f"Error: File '{msh_file}' does not exist.")
        sys.exit(1)

    proc = multiprocessing.Process(target=gmsh_process, args=(msh_file,))
    proc.start()

    print(f"Watching '{msh_file}' for changes...")
    for changes in watch(msh_file):
        print("File changed, restarting Gmsh window...")
        proc.terminate()
        proc.join()
        proc = multiprocessing.Process(target=gmsh_process, args=(msh_file,))
        proc.start()

if __name__ == "__main__":
    main()
