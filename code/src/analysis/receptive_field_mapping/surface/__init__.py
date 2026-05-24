"""Surface geometry sub-package: 3D-to-2D projection, mesh utilities, and SLIM UV."""

from .tangent_plane_alignment import align_points, camera_settings_to_rotation
from .rf_surface_utils import (
    load_or_build_forearm_mesh,
    map_scalars_to_mesh,
    mesh_to_pyvista,
    build_delaunay_mesh,
    apply_rotation_to_mesh,
)
from .slim_helpers import (
    clean_mesh,
    boundary_loop,
    canonicalise_uv,
    flatten_slim,
    compute_face_distortion,
)
from .slim_qc_figures import (
    plot_slim_uv_panel,
    plot_slim_distortion_panel,
    save_slim_qc_figures,
)
from .forearm_slim_uv import (
    SlimUvCache,
    precompute_forearm_slim_uv,
    load_slim_uv_cache,
    barycentric_uv_lookup,
    uv_points_to_xyz,
)
from .slim_uv_config_io import (
    SlimUvConfig,
    SlimUvCleanSteps,
    load_slim_uv_config,
    save_slim_uv_config,
    config_path_for_session,
    config_hash,
    make_default_config,
    DEFAULT_CLEAN_STEPS,
    DEFAULT_MESH_METHOD,
    DEFAULT_MAX_EDGE_MM,
    DEFAULT_N_ITER,
    DEFAULT_SAVE_DIAGNOSTICS,
)
from .rf_projection import (
    project_tangent_plane,
    project_cylindrical_unwrap,
    project_slim,
    PROJECTION_METHODS,
    project_to_2d,
)

__all__ = [
    # tangent_plane_alignment
    "align_points",
    "camera_settings_to_rotation",
    # rf_surface_utils
    "load_or_build_forearm_mesh",
    "map_scalars_to_mesh",
    "mesh_to_pyvista",
    "build_delaunay_mesh",
    "apply_rotation_to_mesh",
    # slim_helpers
    "clean_mesh",
    "boundary_loop",
    "canonicalise_uv",
    "flatten_slim",
    "compute_face_distortion",
    # slim_qc_figures
    "plot_slim_uv_panel",
    "plot_slim_distortion_panel",
    "save_slim_qc_figures",
    # forearm_slim_uv
    "SlimUvCache",
    "precompute_forearm_slim_uv",
    "load_slim_uv_cache",
    "barycentric_uv_lookup",
    "uv_points_to_xyz",
    # slim_uv_config_io
    "SlimUvConfig",
    "SlimUvCleanSteps",
    "load_slim_uv_config",
    "save_slim_uv_config",
    "config_path_for_session",
    "config_hash",
    "make_default_config",
    "DEFAULT_CLEAN_STEPS",
    "DEFAULT_MESH_METHOD",
    "DEFAULT_MAX_EDGE_MM",
    "DEFAULT_N_ITER",
    "DEFAULT_SAVE_DIAGNOSTICS",
    # rf_projection
    "project_tangent_plane",
    "project_cylindrical_unwrap",
    "project_slim",
    "PROJECTION_METHODS",
    "project_to_2d",
]
