"""Tests for the per-vertex contact depth field.

Three families, in increasing order of what they buy:

1. **Known-answer tests** — closed-form geometry (point-vs-box-face,
   sphere-vs-sphere) proves the signed distance means what the module says it
   means, in the units it says.
2. **Contract and fail-fast tests** — the postconditions and every raise site.
3. **Characterisation tests** — the legacy scalar algorithm, transcribed
   verbatim from the pre-refactor ``_calculate_intersection_volume``, is run
   alongside the refactored processor over a deterministic pose sweep and the
   two must agree *bit-identically* on every field.  This is what makes the
   extraction safe; it needs no external data and therefore always runs.

A fourth test regresses against a real recording's reference CSV.  Neither the
recording geometry nor its output CSV is committable (size, participant data),
so that test skips with an explicit reason when the fixture bundle is absent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

o3d = pytest.importorskip("open3d", reason="Open3D is required for contact geometry tests")

from preprocessing.motion_analysis.tactile_quantification.model.contact_depth_field import (  # noqa: E402
    EPSILON,
    MAX_PLAUSIBLE_DEPTH_MM,
    ContactDepthFrame,
    signed_contact_depth_mm,
    validate_pose_transform,
)


# ---------------------------------------------------------------------------
# Importing the processor
# ---------------------------------------------------------------------------
# ``code/tests/conftest.py`` replaces ``preprocessing.forearm_extraction`` and its
# ``registration`` subpackage with bare stubs so the rest of the suite runs
# without the Kinect/OpenGL/Qt SDKs.  The processor imports
# ``serialize_contact_points`` from the package's ``__init__``, which re-exports
# it from ``.registration``, so *both* stubs have to be lifted here — leaving the
# subpackage stubbed makes the parent's own ``__init__`` fail on import.  If the
# real package cannot be imported (SDK-free environment) the stubs are put back
# and the processor-level tests skip with a recorded reason rather than silently
# passing.

_STUBBED_FOR_PROCESSOR = (
    "preprocessing.forearm_extraction",
    "preprocessing.forearm_extraction.registration",
)


def _load_processor():
    """Return ``ObjectsInteractionProcessor``, or ``None`` with a skip reason."""
    lifted = {}
    for dotted in _STUBBED_FOR_PROCESSOR:
        module = sys.modules.get(dotted)
        if module is not None and getattr(module, "__file__", None) is None:
            lifted[dotted] = sys.modules.pop(dotted)
    try:
        from preprocessing.motion_analysis.tactile_quantification.model.objects_interaction_processor import (
            ObjectsInteractionProcessor,
        )
    except ImportError:
        sys.modules.update(lifted)
        return None
    return ObjectsInteractionProcessor


_PROCESSOR_CLS = _load_processor()
_needs_processor = pytest.mark.skipif(
    _PROCESSOR_CLS is None,
    reason=(
        "ObjectsInteractionProcessor needs the real preprocessing.forearm_extraction "
        "package (open3d / PyQt5 / pyk4a SDKs), which this environment stubs out."
    ),
)


def _load_controller():
    """Return ``ObjectsInteractionController``, or ``None`` with a skip reason.

    Same stub-lifting dance as :func:`_load_processor` — the controller imports
    the processor, which imports ``preprocessing.forearm_extraction``.
    """
    if _PROCESSOR_CLS is None:
        return None
    try:
        from preprocessing.motion_analysis.tactile_quantification.core.objects_interaction_controller import (
            ObjectsInteractionController,
        )
    except ImportError:
        return None
    return ObjectsInteractionController


_CONTROLLER_CLS = _load_controller()
_needs_controller = pytest.mark.skipif(
    _CONTROLLER_CLS is None,
    reason=(
        "ObjectsInteractionController needs the real preprocessing.forearm_extraction "
        "package (open3d / PyQt5 / pyk4a SDKs), which this environment stubs out."
    ),
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _planar_grid_mesh(half_extent_mm: float, divisions: int, z_mm: float = 0.0):
    """A flat square terrain patch in the z = ``z_mm`` plane, +z normals.

    Stands in for the forearm terrain: a 2.5D triangulated surface with vertex
    normals, which is what ``ObjectsInteractionProcessor`` guarantees.
    """
    axis = np.linspace(-half_extent_mm, half_extent_mm, divisions + 1)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    vertices = np.column_stack(
        [xx.ravel(), yy.ravel(), np.full(xx.size, float(z_mm))]
    ).astype(np.float64)

    triangles = []
    stride = divisions + 1
    for i in range(divisions):
        for j in range(divisions):
            a = i * stride + j
            b = a + 1
            c = a + stride
            d = c + 1
            triangles.append([a, c, b])
            triangles.append([b, c, d])

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)),
    )
    mesh.compute_vertex_normals()
    return mesh


def _box_mesh(size_mm: float, centre_mm) -> "o3d.geometry.TriangleMesh":
    """An axis-aligned cube of side ``size_mm`` centred on ``centre_mm``.

    A cube *is* its own axis-aligned bounding box, so the broad phase cannot
    retain any query vertex outside it.  That makes it useless as a hand proxy
    for ordinary contact fixtures — it is only used here to construct the
    all-vertices-negative case on purpose.
    """
    mesh = o3d.geometry.TriangleMesh.create_box(size_mm, size_mm, size_mm)
    mesh.translate(np.asarray(centre_mm, dtype=np.float64) - size_mm / 2.0, relative=True)
    mesh.compute_vertex_normals()
    return mesh


def _sphere_mesh(radius_mm: float, centre_mm, resolution: int = 60):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius_mm, resolution=resolution)
    mesh.translate(np.asarray(centre_mm, dtype=np.float64), relative=True)
    mesh.compute_vertex_normals()
    return mesh


#: Geometry of the standard contact fixture: a flat-capped cylinder standing on
#: z in [0, 100] with radius 60, pressed onto a terrain plane below its cap.
#: The cylinder's bounding box is strictly larger than the cylinder, so the
#: broad phase retains terrain vertices *outside* the hand as well as inside —
#: the situation a real hand-on-forearm frame produces.
_CYL_RADIUS_MM = 60.0
_CYL_HEIGHT_MM = 100.0
_CYL_TOP_Z_MM = 100.0
#: Half-extent of the terrain square: just inside the cylinder's bounding box so
#: the crop keeps the corners, which lie outside the cylinder itself.
_TERRAIN_HALF_EXTENT_MM = 58.0


def _cylinder_hand(resolution: int = 60):
    """The standard hand proxy: a closed cylinder occupying z in [0, 100]."""
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=_CYL_RADIUS_MM, height=_CYL_HEIGHT_MM, resolution=resolution
    )
    mesh.translate(np.array([0.0, 0.0, _CYL_HEIGHT_MM / 2.0]), relative=True)
    mesh.compute_vertex_normals()
    return mesh


def _under_the_flat_cap(points: np.ndarray, margin_mm: float = 20.0) -> np.ndarray:
    """Mask of points far enough from the cylinder wall that the cap is nearest."""
    radial = np.linalg.norm(points[:, :2], axis=1)
    return radial < (_CYL_RADIUS_MM - margin_mm)


# ---------------------------------------------------------------------------
# 1. Known-answer tests
# ---------------------------------------------------------------------------

def test_point_vs_plane_matches_closed_form():
    """Terrain under a flat cap: depth is the distance to that plane.

    The hand cylinder's flat top cap sits at z = 100 and the terrain plane at
    z = 90.  For every terrain vertex far enough from the cylinder wall, the cap
    is the nearest surface, so the closed-form signed distance is exactly
    ``-(100 - 90) = -10`` mm.
    """
    hand = _cylinder_hand()
    forearm = _planar_grid_mesh(
        half_extent_mm=_TERRAIN_HALF_EXTENT_MM, divisions=20, z_mm=90.0
    )

    frame = signed_contact_depth_mm(hand, forearm)

    assert frame is not None
    under_cap = _under_the_flat_cap(frame.points)
    assert under_cap.sum() > 10, "fixture must exercise many under-cap vertices"
    np.testing.assert_allclose(
        frame.signed_depth_mm[under_cap],
        np.full(int(under_cap.sum()), -10.0),
        # RaycastingScene is a float32 engine; atol covers its representation
        # limit at this magnitude, rtol is the value the plan specifies.
        rtol=1e-6,
        atol=1e-4,
    )
    assert frame.max_penetration_depth_mm == pytest.approx(10.0, rel=1e-6, abs=1e-4)


def test_point_vs_plane_depth_tracks_plane_height():
    """Sweeping the terrain plane sweeps the depth by the same amount."""
    hand = _cylinder_hand()

    for plane_z, expected_depth in ((95.0, 5.0), (90.0, 10.0), (80.0, 20.0)):
        forearm = _planar_grid_mesh(
            half_extent_mm=_TERRAIN_HALF_EXTENT_MM, divisions=20, z_mm=plane_z
        )
        frame = signed_contact_depth_mm(hand, forearm)
        assert frame is not None
        assert frame.max_penetration_depth_mm == pytest.approx(
            expected_depth, rel=1e-6, abs=1e-4
        )


#: Sphere-vs-sphere fixture: overlapping by ``30 + 25 - 50 = 5`` mm.
_SPHERE_R_HAND_MM = 30.0
_SPHERE_R_FOREARM_MM = 25.0
_SPHERE_SEPARATION_MM = 50.0
_SPHERE_EXPECTED_DEPTH_MM = (
    _SPHERE_R_HAND_MM + _SPHERE_R_FOREARM_MM - _SPHERE_SEPARATION_MM
)


def _sphere_sag_bound_mm(resolution: int) -> float:
    """Leading-order under-report of an inscribed sphere mesh, in millimetres.

    ``create_sphere`` produces a polyhedron *inscribed* in the ideal sphere.  Its
    pole vertex is exact, but the triangles fanning out from that pole form a
    shallow cone whose faces pass *closer* to an interior axis point than the
    pole itself does.  With latitude step ``dt = pi / resolution``, the
    perpendicular distance from a point ``d`` above the pole to those faces is
    ``d * sqrt(1 - dt**2 / 4) ~= d * (1 - dt**2 / 8)`` — so the measured
    penetration falls short of the closed form by ``d * dt**2 / 8``.

    This is a property of the *mesh*, not of the code under test, which is why
    the assertions below are stated against this bound rather than against an
    arbitrarily loosened tolerance.
    """
    delta_theta = np.pi / resolution
    return _SPHERE_EXPECTED_DEPTH_MM * delta_theta**2 / 8.0


def _sphere_vs_sphere_depth(resolution: int) -> float:
    hand = _sphere_mesh(_SPHERE_R_HAND_MM, (0.0, 0.0, 0.0), resolution=resolution)
    forearm = _sphere_mesh(
        _SPHERE_R_FOREARM_MM,
        (0.0, 0.0, -_SPHERE_SEPARATION_MM),
        resolution=resolution,
    )
    frame = signed_contact_depth_mm(hand, forearm)
    assert frame is not None
    return frame.max_penetration_depth_mm


@pytest.mark.parametrize("resolution", [60, 120])
def test_sphere_vs_sphere_penetration_matches_closed_form(resolution):
    """Penetration depth of two overlapping spheres is ``r1 + r2 - |c1 - c2|``.

    Matched to within the mesh's own discretisation sag; see
    :func:`_sphere_sag_bound_mm`.
    """
    depth = _sphere_vs_sphere_depth(resolution)

    assert depth == pytest.approx(
        _SPHERE_EXPECTED_DEPTH_MM, abs=2.0 * _sphere_sag_bound_mm(resolution)
    )
    # One-sided: an inscribed mesh can only ever under-report the penetration.
    assert depth <= _SPHERE_EXPECTED_DEPTH_MM


def test_sphere_vs_sphere_converges_to_the_closed_form():
    """Doubling the resolution quarters the error — the signature of ``dt**2``.

    Agreement at a single resolution could be coincidence; second-order
    convergence to the analytic value cannot.
    """
    coarse_error = _SPHERE_EXPECTED_DEPTH_MM - _sphere_vs_sphere_depth(60)
    fine_error = _SPHERE_EXPECTED_DEPTH_MM - _sphere_vs_sphere_depth(120)

    assert coarse_error == pytest.approx(_sphere_sag_bound_mm(60), rel=0.05)
    assert fine_error == pytest.approx(_sphere_sag_bound_mm(120), rel=0.05)
    assert coarse_error / fine_error == pytest.approx(4.0, rel=0.05)


def test_vertices_inside_the_hand_report_negative():
    """Sign convention: inside the hand mesh is negative, not positive."""
    hand = _cylinder_hand()
    forearm = _planar_grid_mesh(
        half_extent_mm=_TERRAIN_HALF_EXTENT_MM, divisions=20, z_mm=90.0
    )

    frame = signed_contact_depth_mm(hand, forearm)

    assert frame is not None
    assert np.all(frame.signed_depth_mm < 0.0)


# ---------------------------------------------------------------------------
# 2. Contract and fail-fast tests
# ---------------------------------------------------------------------------

def test_field_is_index_aligned_with_contact_points():
    hand = _cylinder_hand()
    forearm = _planar_grid_mesh(
        half_extent_mm=_TERRAIN_HALF_EXTENT_MM, divisions=20, z_mm=90.0
    )

    frame = signed_contact_depth_mm(hand, forearm)

    assert frame is not None
    assert isinstance(frame, ContactDepthFrame)
    assert frame.points.shape == (len(frame.signed_depth_mm), 3)
    assert frame.points.dtype == np.float64
    assert frame.signed_depth_mm.dtype == np.float64
    assert frame.normals.shape == frame.points.shape
    assert frame.mean_location.shape == (3,)
    assert isinstance(frame.total_area_mm2, float)


def test_frame_identity_defaults_to_absent_not_zero():
    """An unlabelled frame carries ``None``, never a fabricated index 0."""
    hand = _cylinder_hand()
    forearm = _planar_grid_mesh(
        half_extent_mm=_TERRAIN_HALF_EXTENT_MM, divisions=20, z_mm=90.0
    )

    unlabelled = signed_contact_depth_mm(hand, forearm)
    labelled = signed_contact_depth_mm(hand, forearm, frame_index=17, time_s=0.5667)

    assert unlabelled.frame_index is None
    assert unlabelled.time_s is None
    assert labelled.frame_index == 17
    assert labelled.time_s == pytest.approx(0.5667)


def test_no_contact_returns_none_not_an_empty_frame():
    """Separated geometry is a legitimate empty result, expressed as ``None``."""
    hand = _box_mesh(size_mm=20.0, centre_mm=(0.0, 0.0, 500.0))
    forearm = _planar_grid_mesh(half_extent_mm=50.0, divisions=6, z_mm=0.0)

    assert signed_contact_depth_mm(hand, forearm) is None


def test_terrain_outside_the_hand_bounding_box_returns_none():
    """The broad phase alone can resolve "no contact"; that is not a failure."""
    hand = _cylinder_hand()
    # 1 mm above the cylinder's cap, so the AABB crop keeps nothing.
    forearm = _planar_grid_mesh(
        half_extent_mm=_TERRAIN_HALF_EXTENT_MM, divisions=6, z_mm=101.0
    )

    assert signed_contact_depth_mm(hand, forearm) is None


def test_single_triangle_patch_yields_three_vertices():
    """The smallest non-degenerate patch: one triangle, three contact vertices.

    Two disjoint triangles at the same height: one on the cylinder axis (inside),
    one out at radius ~74 mm (outside the cylinder but still inside its bounding
    box, so the broad phase keeps it and the winding sentinel stays quiet).
    """
    hand = _cylinder_hand()
    vertices = np.array(
        [
            [-5.0, -5.0, 90.0],
            [5.0, -5.0, 90.0],
            [0.0, 5.0, 90.0],
            [50.0, 50.0, 90.0],
            [58.0, 50.0, 90.0],
            [54.0, 58.0, 90.0],
        ],
        dtype=np.float64,
    )
    forearm = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)),
    )
    forearm.compute_vertex_normals()

    frame = signed_contact_depth_mm(hand, forearm)

    assert frame is not None
    assert len(frame.points) == 3
    assert len(frame.signed_depth_mm) == 3
    np.testing.assert_allclose(np.sort(frame.points[:, 0]), [-5.0, 0.0, 5.0])


def test_absent_hand_pose_raises_rather_than_reporting_zero_depth():
    forearm = _planar_grid_mesh(half_extent_mm=20.0, divisions=4, z_mm=0.0)
    with pytest.raises(ValueError, match="absent"):
        signed_contact_depth_mm(None, forearm)


def test_nan_in_hand_vertices_raises():
    hand = _cylinder_hand()
    corrupted = np.asarray(hand.vertices).copy()
    corrupted[0, 2] = np.nan
    hand.vertices = o3d.utility.Vector3dVector(corrupted)
    forearm = _planar_grid_mesh(
        half_extent_mm=_TERRAIN_HALF_EXTENT_MM, divisions=6, z_mm=90.0
    )

    with pytest.raises(ValueError, match="non-finite"):
        signed_contact_depth_mm(hand, forearm)


def test_empty_hand_mesh_raises():
    hand = o3d.geometry.TriangleMesh()
    forearm = _planar_grid_mesh(half_extent_mm=20.0, divisions=4, z_mm=90.0)

    with pytest.raises(ValueError, match="no triangles"):
        signed_contact_depth_mm(hand, forearm)


def test_all_queried_vertices_negative_raises_winding_sentinel():
    """A terrain wholly enclosed by the hand is the winding-inversion signature.

    A cube is used deliberately: it coincides with its own bounding box, so the
    broad phase cannot retain a single vertex outside the hand — the same shape
    the field takes when an inverted winding order reports the whole cropped
    region as penetrating.
    """
    hand = _box_mesh(size_mm=200.0, centre_mm=(0.0, 0.0, 0.0))
    forearm = _planar_grid_mesh(half_extent_mm=40.0, divisions=4, z_mm=0.0)

    with pytest.raises(ValueError, match="winding"):
        signed_contact_depth_mm(hand, forearm)


def test_depth_beyond_the_sanity_band_raises():
    """A 300 mm penetration is a units/coordinate-space error, not a measurement."""
    radius = 300.0
    hand = _sphere_mesh(radius, (0.0, 0.0, 0.0), resolution=30)
    # Spans past the sphere equator so the corner vertices stay outside it and
    # the all-negative sentinel does not pre-empt the sanity band.
    forearm = _planar_grid_mesh(half_extent_mm=290.0, divisions=8, z_mm=0.0)

    with pytest.raises(ValueError, match="sanity band"):
        signed_contact_depth_mm(hand, forearm)

    assert MAX_PLAUSIBLE_DEPTH_MM == 200.0


def test_sanity_band_is_overridable_per_call():
    hand = _cylinder_hand()
    forearm = _planar_grid_mesh(
        half_extent_mm=_TERRAIN_HALF_EXTENT_MM, divisions=6, z_mm=90.0
    )

    with pytest.raises(ValueError, match="sanity band"):
        signed_contact_depth_mm(hand, forearm, max_plausible_depth_mm=1.0)


def test_epsilon_constant_is_the_historical_tolerance():
    assert EPSILON == 1e-5


@pytest.mark.parametrize(
    "matrix, message",
    [
        (np.diag([1.0, 1.0, -1.0, 1.0]), "determinant"),
        (np.diag([-2.0, 1.0, 1.0, 1.0]), "determinant"),
        (np.full((4, 4), np.nan), "non-finite"),
        (np.eye(3), "4, 4"),
    ],
)
def test_validate_pose_transform_rejects_bad_transforms(matrix, message):
    with pytest.raises(ValueError, match=message):
        validate_pose_transform(matrix)


def test_validate_pose_transform_accepts_a_rigid_transform():
    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [12.0, -3.0, 4.5]

    validate_pose_transform(matrix)  # must not raise


# ---------------------------------------------------------------------------
# 3. Determinism
# ---------------------------------------------------------------------------

def test_same_frame_computed_twice_is_bit_identical():
    hand = _cylinder_hand()
    forearm = _planar_grid_mesh(
        half_extent_mm=_TERRAIN_HALF_EXTENT_MM, divisions=19, z_mm=88.0
    )

    first = signed_contact_depth_mm(hand, forearm)
    second = signed_contact_depth_mm(hand, forearm)

    assert first is not None and second is not None
    assert np.array_equal(first.signed_depth_mm, second.signed_depth_mm)
    assert np.array_equal(first.points, second.points)
    assert np.array_equal(first.mean_location, second.mean_location)
    assert first.total_area_mm2 == second.total_area_mm2


# ---------------------------------------------------------------------------
# 4. Characterisation: the refactor changed nothing
# ---------------------------------------------------------------------------

def _legacy_intersection_volume(ref_mesh, input_mesh, serialize_contact_points):
    """The pre-refactor ``_calculate_intersection_volume``, transcribed verbatim.

    Kept byte-for-byte faithful to the algorithm the reference CSVs were
    produced with, so it can serve as the oracle for the extraction.  It is
    deliberately *not* refactored, tidied, or DRY-ed against the production
    code — its whole value is that it is an independent copy.
    """
    empty_contact = {
        "contact_detected": 0,
        "contact_depth": 0.0,
        "contact_area": 0.0,
        "contact_location_x": np.nan,
        "contact_location_y": np.nan,
        "contact_location_z": np.nan,
        "contact_points": "[]",
    }
    empty_viz = {"contact_points": np.array([]), "contact_normals": np.array([])}

    aabb = input_mesh.get_axis_aligned_bounding_box()
    cropped_terrain = ref_mesh.crop(aabb)
    if len(cropped_terrain.triangles) == 0:
        return empty_contact, empty_viz

    t_object = o3d.t.geometry.TriangleMesh.from_legacy(input_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_object)

    cropped_vertices_np = np.asarray(cropped_terrain.vertices)
    t_terrain_verts = o3d.core.Tensor.from_numpy(cropped_vertices_np.astype(np.float32))
    distances_np = scene.compute_signed_distance(t_terrain_verts).numpy()

    EPS = 1e-5
    cropped_triangles = np.asarray(cropped_terrain.triangles)
    tri_distances = distances_np[cropped_triangles]
    inside_mask = np.all(tri_distances < EPS, axis=1)
    if not np.any(inside_mask):
        return empty_contact, empty_viz

    active_tris = cropped_triangles[inside_mask]
    v0 = cropped_vertices_np[active_tris[:, 0]]
    v1 = cropped_vertices_np[active_tris[:, 1]]
    v2 = cropped_vertices_np[active_tris[:, 2]]
    cross_prod = np.cross(v1 - v0, v2 - v0)
    active_areas = 0.5 * np.linalg.norm(cross_prod, axis=1)
    total_contact_area = np.sum(active_areas)
    centroids = (v0 + v1 + v2) / 3.0
    mean_location = np.mean(centroids, axis=0)

    active_vertex_distances = tri_distances[inside_mask]
    contact_depth = np.max(np.abs(active_vertex_distances))

    unique_contact_indices = np.unique(active_tris)
    contact_points = cropped_vertices_np[unique_contact_indices]

    contact_quantities = {
        "contact_detected": 1,
        "contact_points": serialize_contact_points([tuple(pt) for pt in contact_points]),
        "contact_depth": float(contact_depth),
        "contact_area": float(total_contact_area),
        "contact_location_x": float(mean_location[0]),
        "contact_location_y": float(mean_location[1]),
        "contact_location_z": float(mean_location[2]),
    }
    contact_info = {
        "contact_points": contact_points,
        "contact_normals": np.asarray(cropped_terrain.vertex_normals)[unique_contact_indices]
        if cropped_terrain.has_vertex_normals()
        else np.array([]),
    }
    return contact_quantities, contact_info


_SWEEP_TERRAIN_Z_MM = 90.0
_SWEEP_FINGERTIP_RADIUS_MM = 18.0


def _pose_sweep(n_frames: int = 40, seed: int = 20260812):
    """A deterministic sequence of fingertip poses sweeping across the terrain.

    A sphere stands in for a fingertip: like a real hand, and unlike a cube, it
    is strictly smaller than its bounding box, so each frame's broad phase keeps
    terrain vertices on both sides of the surface.  The vertical profile starts
    and ends clear of the terrain, so the sweep exercises the contact and the
    no-contact branch.  Seeded explicitly (guide 03 §2) for reproducibility.
    """
    rng = np.random.default_rng(seed)
    for index in range(n_frames):
        phase = np.pi * index / (n_frames - 1)
        travel = -45.0 + 90.0 * index / (n_frames - 1)
        height = _SWEEP_TERRAIN_Z_MM + 22.0 - 18.0 * np.sin(phase)
        jitter = rng.normal(scale=0.4, size=3)
        centre = np.array([travel, 0.0, height]) + jitter
        yield _sphere_mesh(_SWEEP_FINGERTIP_RADIUS_MM, centre, resolution=30)


@_needs_processor
def test_refactored_processor_reproduces_the_legacy_algorithm_bit_identically():
    """Every CSV field, every frame — not a spot check on depth alone."""
    from preprocessing.forearm_extraction import serialize_contact_points

    forearm = _planar_grid_mesh(half_extent_mm=60.0, divisions=24, z_mm=90.0)
    processor = _PROCESSOR_CLS(reference_geometry=forearm)

    contacting_frames = 0
    empty_frames = 0

    for index, hand in enumerate(_pose_sweep()):
        expected_quantities, expected_viz = _legacy_intersection_volume(
            forearm, hand, serialize_contact_points
        )
        actual_quantities, actual_viz, _ = processor.process_single_frame(
            current_mesh=hand, frame_index=index, time_s=index / 30.0
        )

        assert set(actual_quantities) == set(expected_quantities)
        for key, expected in expected_quantities.items():
            actual = actual_quantities[key]
            if isinstance(expected, float) and np.isnan(expected):
                assert np.isnan(actual), key
            else:
                # Bit-identical, not approximate: the refactor must not have
                # moved a single least-significant bit.
                assert actual == expected, key

        assert np.array_equal(
            actual_viz["contact_points"], expected_viz["contact_points"]
        )

        if expected_quantities["contact_detected"] == 1:
            contacting_frames += 1
            assert np.array_equal(
                actual_viz["contact_normals"], expected_viz["contact_normals"]
            )
        else:
            empty_frames += 1

    # Guard against a vacuous pass: the sweep must exercise both branches.
    assert contacting_frames > 0, "pose sweep produced no contacting frames"
    assert empty_frames > 0, "pose sweep produced no empty frames"


@_needs_processor
def test_max_abs_field_equals_the_processor_scalar_bit_identically():
    """The critical invariant, stated directly on the two quantities."""
    forearm = _planar_grid_mesh(half_extent_mm=60.0, divisions=24, z_mm=90.0)
    processor = _PROCESSOR_CLS(reference_geometry=forearm)

    checked = 0
    for index, hand in enumerate(_pose_sweep()):
        quantities, _, surfaced = processor.process_single_frame(
            current_mesh=hand, frame_index=index, time_s=index / 30.0
        )
        frame = signed_contact_depth_mm(hand, forearm)

        if frame is None:
            assert quantities["contact_detected"] == 0
            assert surfaced is None
            continue

        checked += 1
        assert quantities["contact_detected"] == 1
        assert np.array_equal(
            np.float64(quantities["contact_depth"]),
            np.max(np.abs(frame.signed_depth_mm)),
        )
        assert len(frame.signed_depth_mm) == len(frame.points)

        # The processor must surface the very field it summarised, stamped with
        # the identity it was given — not a recomputation and not an unlabelled one.
        assert surfaced is not None
        assert surfaced.frame_index == index
        assert surfaced.time_s == pytest.approx(index / 30.0)
        assert np.array_equal(surfaced.signed_depth_mm, frame.signed_depth_mm)
        assert np.array_equal(surfaced.points, frame.points)

    assert checked > 0


@_needs_processor
def test_processor_refuses_an_absent_hand_pose():
    forearm = _planar_grid_mesh(half_extent_mm=20.0, divisions=4, z_mm=90.0)
    processor = _PROCESSOR_CLS(reference_geometry=forearm)

    with pytest.raises(ValueError, match="absent"):
        processor.process_single_frame(current_mesh=None, frame_index=0, time_s=0.0)


# ---------------------------------------------------------------------------
# 4b. The controller surfaces the field series
# ---------------------------------------------------------------------------


def _controller_over_the_sweep():
    """Run the real controller over the deterministic pose sweep."""
    forearm = _planar_grid_mesh(half_extent_mm=60.0, divisions=24, z_mm=90.0)
    hands = list(_pose_sweep())
    timestamps = [index / 30.0 for index in range(len(hands))]
    controller = _CONTROLLER_CLS(
        hand_meshes=hands,
        timestamps=timestamps,
        references_mesh={0: forearm},
    )
    return controller.run()


@_needs_controller
def test_controller_returns_a_field_series_matching_the_dataframe():
    """The series holds exactly the contacting frames, labelled consistently."""
    df, series, vis_artifacts = _controller_over_the_sweep()

    assert vis_artifacts is not None
    contacting = df.loc[df["contact_detected"] == 1]

    # Guard against a vacuous pass: the sweep must exercise both branches.
    assert len(contacting) > 0
    assert len(contacting) < len(df)

    assert len(series) == len(contacting)
    assert [f.frame_index for f in series] == contacting["frame_index"].tolist()
    np.testing.assert_allclose(
        [f.time_s for f in series], contacting["time"].to_numpy()
    )

    # Frame identity is populated, never left unlabelled.
    assert all(f.frame_index is not None and f.time_s is not None for f in series)


@_needs_controller
def test_controller_field_series_agrees_with_the_csv_scalar_bit_identically():
    """``max(|field|)`` per frame is the CSV's ``contact_depth``, to the bit."""
    df, series, _ = _controller_over_the_sweep()

    indexed = df.set_index("frame_index")
    for frame in series:
        recovered = np.max(np.abs(frame.signed_depth_mm))
        assert np.array_equal(
            recovered, np.float64(indexed.loc[frame.frame_index, "contact_depth"])
        ), frame.frame_index

    # A frame absent from the series is zero contact, not an unrecorded frame.
    covered = {f.frame_index for f in series}
    for frame_index, row in indexed.iterrows():
        if frame_index not in covered:
            assert row["contact_detected"] == 0


# ---------------------------------------------------------------------------
# 5. Regression against a real recording
# ---------------------------------------------------------------------------
# The reference bundle is not committed: a recording's hand-pose NPZ, forearm
# mesh and somatosensory CSV are far too large for the repository and carry
# participant data.  Point SOCIAL_TOUCH_CONTACT_REFERENCE_DIR at a directory
# holding
#     reference_contact.csv   the somatosensory CSV produced before this refactor
#     hand_meshes.npz         'vertices' (F, V, 3) float64, 'triangles' (T, 3) int32
#     forearm_mesh.obj        the block's Space-1 forearm mesh
# to run it.  Absent that, the test skips loudly rather than passing vacuously.

_REFERENCE_DIR_ENV = "SOCIAL_TOUCH_CONTACT_REFERENCE_DIR"


def _reference_bundle():
    root = os.environ.get(_REFERENCE_DIR_ENV)
    if root is None:
        return None
    path = Path(root)
    required = {
        "csv": path / "reference_contact.csv",
        "hand": path / "hand_meshes.npz",
        "forearm": path / "forearm_mesh.obj",
    }
    missing = [str(p) for p in required.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{_REFERENCE_DIR_ENV} is set to {path} but these are missing: {missing}"
        )
    return required


@_needs_processor
def test_recording_regression_against_reference_csv():
    """Recompute a whole recording and match the committed reference CSV exactly."""
    bundle = _reference_bundle()
    if bundle is None:
        pytest.skip(
            f"No reference recording bundle: set {_REFERENCE_DIR_ENV} to a directory "
            "containing reference_contact.csv, hand_meshes.npz and forearm_mesh.obj. "
            "The bundle is not committed (size and participant data)."
        )

    pd = pytest.importorskip("pandas")

    # float_precision="round_trip" is mandatory, not decoration: pandas' default
    # CSV float parser perturbs ~9% of the values in a real recording by one ULP,
    # which would fail the bit-identity assertion below on values the pipeline
    # actually wrote correctly. Measured on the ST14-01 block-order-01 reference.
    reference = pd.read_csv(bundle["csv"], float_precision="round_trip")
    forearm = o3d.io.read_triangle_mesh(str(bundle["forearm"]))
    if not forearm.has_vertex_normals():
        forearm.compute_vertex_normals()

    payload = np.load(bundle["hand"])
    vertices_per_frame = payload["vertices"]
    triangles = o3d.utility.Vector3iVector(payload["triangles"])

    assert len(vertices_per_frame) == len(reference), (
        f"{len(vertices_per_frame)} hand poses vs {len(reference)} reference rows"
    )

    processor = _PROCESSOR_CLS(reference_geometry=forearm)

    recomputed_depth = np.empty(len(vertices_per_frame), dtype=np.float64)
    recomputed_area = np.empty(len(vertices_per_frame), dtype=np.float64)
    recomputed_detected = np.empty(len(vertices_per_frame), dtype=np.int64)

    for index, vertices in enumerate(vertices_per_frame):
        hand = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
            triangles,
        )
        quantities, _, _ = processor.process_single_frame(
            current_mesh=hand, frame_index=index, time_s=index / 30.0
        )
        recomputed_depth[index] = quantities["contact_depth"]
        recomputed_area[index] = quantities["contact_area"]
        recomputed_detected[index] = quantities["contact_detected"]

        frame = signed_contact_depth_mm(hand, processor.ref_mesh)
        if frame is not None:
            assert len(frame.signed_depth_mm) == len(frame.points), index

    assert np.array_equal(
        recomputed_detected, reference["contact_detected"].to_numpy(dtype=np.int64)
    )
    assert np.array_equal(
        recomputed_depth, reference["contact_depth"].to_numpy(dtype=np.float64)
    )
    assert np.array_equal(
        recomputed_area, reference["contact_area"].to_numpy(dtype=np.float64)
    )
