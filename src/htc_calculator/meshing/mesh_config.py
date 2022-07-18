# https://www.openfoam.com/documentation/user-guide/4-mesh-generation-and-conversion/4.4-mesh-generation-with-the-snappyhexmesh-utility

# ----------------------------------------------------------------------------------------------------------------------
# Cell splitting at feature edges and surfaces
# ----------------------------------------------------------------------------------------------------------------------

merge_tolerance = 1e-6                      # Merge tolerance as fraction of bounding box of initial mesh
max_local_cells = int(2e+7)                 # max number of cellsMaximum number of cells per processor during refinement
max_global_cells = int(3e+7)                # Overall cell limit during refinement (i.e. before removal)
min_refinement_cells = 0                    # If ≥ number of cells to be refined, surface refinement stops
max_load_unbalance = 0.2                    # Maximum processor imbalance during refinement where a value of 0 represents a perfect balance
n_cells_between_levels = 2                  # Number of buffer layers of cells between different levels of refinement
resolve_feature_angle = 30                  # Applies maximum level of refinement to cells that can see intersections whose angle exceeds this
allow_free_standing_zone_faces = False      # Allow the generation of free-standing zone faces


# ----------------------------------------------------------------------------------------------------------------------
# Snapping
# ----------------------------------------------------------------------------------------------------------------------

n_smooth_patch = 1                          # Number of patch smoothing iterations before finding correspondence to surface
tolerance = 4.0                             # The tolerance between_levels
n_solve_iter = 30                           # Number of mesh displacement relaxation iterations
n_relax_iter = 5                            # Maximum number of snapping relaxation iterations
n_feature_snap_iter = 10                    # Number of feature edge snapping iterations
implicit_feature_snap = False               # Detect (geometric only) features by sampling the surface
explicit_feature_snap = True                # Use castellatedMeshControls features
multi_region_feature_snap = False           # Detect features between multiple surfaces when using the explicitFeatureSnap


# ----------------------------------------------------------------------------------------------------------------------
# other
# ----------------------------------------------------------------------------------------------------------------------

feature_edges_level = 0                     # level of refinement of feature
pipe_section_default_cell_size = 50         # Default cell size for pipe sections
