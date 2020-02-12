# FSF
Fracture-Separation-Analysis (three-dimensional separation of the skeleton of individual planar-like connected features)

Before running the program:
1) Open "fsf.py" in Spyder (if you have Anaconda for Python) and, at the bottom, put the PATH of the folder where the gray-scale image stack is stored (images must be ".tif" files) and define the parameters for the analysis based on the irregular shape of your features:
  - tol_or = Orientation tolerance for 2D pairing of segments (in degrees)
  - loc_tol_or = Orientation tolerance between 2D segments close to the interconnection (in degrees)
  - vx = Size of the window to place at each interconnection to record the "segments" present
  - small_segm = [a, b] tol_or becomes "tol_or x a" for labels smaller than b
  - tol_dist_i = Maximum distance for the intersection in the previous image for 3D continuity
  - tol_sp_base = Distance tolerance for 3D pairing of segments (in pixels)
  - tol_or_base = Orientation tolerance for 3D pairing of segments (in degrees)

2) RUN the script
