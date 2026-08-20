"""Per-vertex signed contact depth between a hand mesh and a forearm mesh.

This module holds the **single** definition of "interpenetration" in the
pipeline.  :func:`signed_contact_depth_mm` computes the per-vertex signed
depth field for one frame; every scalar summary the pipeline reports (maximum
penetration depth, contact area, contact centroid) is *derived* from that
field rather than computed independently, so the two can never drift apart.

Purity contract
---------------
This module knows about geometry and nothing else.  It must never import or
reference file paths, config objects, DataFrames, PyVista, Qt, or
session/block/recording identity.  It depends only on ``numpy`` and the
geometry/compute half of ``open3d`` (never ``open3d.visualization``).

Decision record — the three things a future maintainer will otherwise
re-derive incorrectly
---------------------------------------------------------------------
**Query direction.**  The ``o3d.t.geometry.RaycastingScene`` is built from the
**hand** mesh, and the **forearm** vertices are the query points.  The reverse
pairing answers a different question and yields different numbers.

**Sign convention.**  ``RaycastingScene.compute_signed_distance`` returns a
*negative* value for points inside the scene geometry.  Therefore
``signed_depth_mm < 0`` means the forearm vertex lies inside the hand, i.e. it
is **penetrating**.  "Penetration depth" is ``-signed_depth_mm``, a positive
magnitude.  Storage stays signed.

**Units.**  Millimetres, Kinect-native.  There is no unit conversion anywhere
in this pipeline.  A maximum depth of ``0.03`` where ``30`` was expected is the
metres-vs-millimetres tell; :data:`MAX_PLAUSIBLE_DEPTH_MM` guards against it.

Precision note
--------------
``RaycastingScene`` is a float32 engine: query points are cast to float32 and
distances come back as float32.  The field is widened to float64 for storage,
which is exact, so ``max(|signed_depth_mm|)`` is bit-identical to the float32
maximum the legacy scalar was computed from.

Non-watertight geometry
-----------------------
The hand mesh is not guaranteed watertight (vertices are removed before
contact processing).  Signed distance is formally undefined on an open shell
and returns a *plausible* number rather than erroring.  That is a known,
inherited limitation which this module deliberately does not repair; the
per-vertex field is what makes it visible (isolated sign flips) where the
legacy ``max()`` hid it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d

__all__ = [
    "EPSILON",
    "MAX_PLAUSIBLE_DEPTH_MM",
    "ContactDepthFrame",
    "signed_contact_depth_mm",
    "validate_pose_transform",
]


#: Grazing-contact tolerance, in millimetres.  A forearm triangle counts as
#: contacting when *all three* of its vertices satisfy ``signed_distance <
#: EPSILON``.  This is the historical patch definition and is deliberately
#: unchanged so ``contact_area`` stays comparable across reprocessings.
EPSILON: float = 1e-5

#: Upper sanity bound on ``max(|signed_depth_mm|)``, in millimetres.  A hand
#: cannot plausibly penetrate a forearm by 20 cm; exceeding this means the
#: geometry is wrong (bad pose fit, mismatched coordinate space, or a unit
#: error) and the frame must not be reported as a measurement.
MAX_PLAUSIBLE_DEPTH_MM: float = 200.0


@dataclass(frozen=True)
class ContactDepthFrame:
    """The per-vertex contact depth field for a single frame.

    Data only — this is the inter-stage contract between the geometry stage
    that produces it and any consumer (scalar summariser, viewer, writer).  It
    knows about nothing.

    Attributes:
        frame_index: Kinect frame index, or ``None`` when the producer was not
            given frame identity.  ``None`` is an explicit "unlabelled" marker;
            it is never substituted with ``0``.
        time_s: Frame timestamp in seconds, or ``None`` when unlabelled.
        points: ``(N, 3)`` float64 contact-patch vertex positions in
            millimetres, Kinect Space 1.
        signed_depth_mm: ``(N,)`` float64 signed distance of each contact-patch
            vertex to the hand surface, in millimetres.  Negative = penetrating.
            Index-aligned with :attr:`points`.
        normals: ``(N, 3)`` float64 forearm vertex normals at the contact-patch
            vertices, index-aligned with :attr:`points`.  Shape ``(0, 3)`` when
            the forearm mesh carried no vertex normals.
        total_area_mm2: Summed surface area of the contacting forearm
            triangles, in square millimetres.
        mean_location: ``(3,)`` float64 mean of the contacting triangle
            centroids, in millimetres.
    """

    frame_index: Optional[int]
    time_s: Optional[float]
    points: np.ndarray
    signed_depth_mm: np.ndarray
    normals: np.ndarray
    total_area_mm2: float
    mean_location: np.ndarray

    @property
    def max_penetration_depth_mm(self) -> float:
        """Largest penetration magnitude in the patch, in millimetres.

        This is the quantity the pipeline historically reported as
        ``contact_depth``.  It is derived here so exactly one place in the
        codebase turns the field into that scalar.
        """
        return float(np.max(np.abs(self.signed_depth_mm)))


def validate_pose_transform(matrix: np.ndarray, *, label: str = "pose transform") -> None:
    """Raise unless ``matrix`` is a finite, orientation-preserving 4x4 transform.

    A negative determinant inverts the face winding order of the transformed
    mesh.  ``RaycastingScene.compute_signed_distance`` resolves inside/outside
    by winding number, so an inverted transform flips the sign of *every*
    vertex in the field and reports the whole forearm as penetrating.  See
    ``docs/development/knowledge-base/bug-contact-detection-winding-inversion.md``.

    This validator is exposed for callers that hold the transform (the pose
    loader / driver); :func:`signed_contact_depth_mm` receives an already
    transformed mesh and cannot perform the check itself.

    Args:
        matrix: A ``(4, 4)`` homogeneous transform.
        label: Name used in error messages to identify the offending transform.

    Raises:
        ValueError: If the matrix is not 4x4, is not finite, or has a
            non-positive rotation-block determinant.
    """
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (4, 4):
        raise ValueError(f"{label}: expected a (4, 4) matrix, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label}: contains non-finite values.")

    # Scalar triple product rather than np.linalg.det: a 3x3 determinant needs
    # no LAPACK, and the LAPACK-backed path is not reliably loadable in this
    # project's Open3D/NumPy environment.
    rotation = array[:3, :3]
    determinant = float(np.dot(rotation[0], np.cross(rotation[1], rotation[2])))
    if determinant <= 0.0:
        raise ValueError(
            f"{label}: det(M[:3, :3]) = {determinant!r} <= 0. A non-positive "
            "determinant inverts the mesh winding order, which inverts the sign "
            "of every signed-distance value. Refusing to compute a contact "
            "depth field from it."
        )


def signed_contact_depth_mm(
    hand_mesh: o3d.geometry.TriangleMesh,
    forearm_mesh: o3d.geometry.TriangleMesh,
    *,
    epsilon: float = EPSILON,
    max_plausible_depth_mm: float = MAX_PLAUSIBLE_DEPTH_MM,
    frame_index: Optional[int] = None,
    time_s: Optional[float] = None,
) -> Optional[ContactDepthFrame]:
    """Return the per-vertex contact depth field, or ``None`` if no contact.

    ``None`` means "no contact this frame" — a legitimate, documented empty
    result.  It never means "computation failed"; failures raise.

    Broad phase: the forearm mesh is cropped to the hand's axis-aligned
    bounding box.  Narrow phase: an ``o3d.t.geometry.RaycastingScene`` built
    from the hand mesh is queried at every cropped forearm vertex.  A forearm
    triangle is *contacting* when all three of its vertices report
    ``signed_distance < epsilon``; the contact patch is the set of vertices
    belonging to at least one such triangle.

    Args:
        hand_mesh: The dynamic hand mesh, already transformed into world space
            (Kinect Space 1, millimetres).  Builds the raycasting scene.
        forearm_mesh: The static forearm terrain mesh in the same space.
            Supplies the query vertices and the contact areas.
        epsilon: Grazing-contact tolerance in millimetres.
        max_plausible_depth_mm: Upper sanity bound on the resulting maximum
            penetration magnitude.
        frame_index: Optional frame identity, copied onto the result.
        time_s: Optional frame timestamp in seconds, copied onto the result.

    Returns:
        A :class:`ContactDepthFrame`, or ``None`` when the hand does not touch
        the forearm this frame.

    Raises:
        ValueError: If either mesh is absent, degenerate, or non-finite; if the
            computed distances are non-finite; if *every* queried forearm vertex
            reports a negative distance (the winding-inversion signature); or if
            the maximum penetration magnitude leaves the sanity band.
        AssertionError: If the returned arrays violate the index-alignment
            contract.
    """
    _validate_input_meshes(hand_mesh, forearm_mesh)

    # --- Broad phase: crop the forearm terrain to the hand's AABB. -----------
    # Acts as a spatial hash, bounding the number of signed-distance queries.
    aabb = hand_mesh.get_axis_aligned_bounding_box()
    cropped_terrain = forearm_mesh.crop(aabb)

    if len(cropped_terrain.triangles) == 0:
        return None

    cropped_vertices_np = np.asarray(cropped_terrain.vertices)
    if not np.all(np.isfinite(cropped_vertices_np)):
        raise ValueError(
            "Forearm mesh contains non-finite vertices inside the hand's bounding box."
        )

    # --- Narrow phase: signed distance, watertightness-independent. ----------
    # Occupancy would require a closed hand mesh; signed distance does not.
    t_object = o3d.t.geometry.TriangleMesh.from_legacy(hand_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_object)

    # RaycastingScene is a float32 engine; the cast is required, not a choice.
    t_terrain_verts = o3d.core.Tensor.from_numpy(cropped_vertices_np.astype(np.float32))
    distances_np = scene.compute_signed_distance(t_terrain_verts).numpy()

    if not np.all(np.isfinite(distances_np)):
        raise ValueError(
            "Signed-distance query returned non-finite values — the hand mesh "
            "geometry is degenerate."
        )
    if bool(np.all(distances_np < 0.0)):
        raise ValueError(
            f"All {distances_np.size} queried forearm vertices report a negative "
            "signed distance. A whole-bounding-box contact is not physically "
            "plausible; this is the signature of an inverted hand-mesh winding "
            "order (negative-determinant pose transform). See "
            "bug-contact-detection-winding-inversion.md."
        )

    # --- Contact patch: triangles whose three vertices are all penetrating. --
    cropped_triangles = np.asarray(cropped_terrain.triangles)
    tri_distances = distances_np[cropped_triangles]
    inside_mask = np.all(tri_distances < epsilon, axis=1)

    if not np.any(inside_mask):
        return None

    active_tris = cropped_triangles[inside_mask]

    # --- Area and centroid of the contacting triangles. ----------------------
    v0 = cropped_vertices_np[active_tris[:, 0]]
    v1 = cropped_vertices_np[active_tris[:, 1]]
    v2 = cropped_vertices_np[active_tris[:, 2]]

    cross_prod = np.cross(v1 - v0, v2 - v0)
    active_areas = 0.5 * np.linalg.norm(cross_prod, axis=1)
    total_contact_area = float(np.sum(active_areas))

    centroids = (v0 + v1 + v2) / 3.0
    mean_location = np.mean(centroids, axis=0)

    # --- The field itself, one value per contact-patch vertex. ---------------
    contact_indices = np.unique(active_tris)
    points = cropped_vertices_np[contact_indices]
    # float32 -> float64 widening is exact, so max(|field|) is bit-identical to
    # the float32 maximum the legacy scalar was derived from.
    signed_depth = distances_np[contact_indices].astype(np.float64)

    if cropped_terrain.has_vertex_normals():
        normals = np.asarray(cropped_terrain.vertex_normals)[contact_indices]
    else:
        normals = np.empty((0, 3), dtype=np.float64)

    _validate_field(points, signed_depth, max_plausible_depth_mm)

    return ContactDepthFrame(
        frame_index=frame_index,
        time_s=time_s,
        points=points,
        signed_depth_mm=signed_depth,
        normals=normals,
        total_area_mm2=total_contact_area,
        mean_location=mean_location,
    )


def _validate_input_meshes(
    hand_mesh: o3d.geometry.TriangleMesh,
    forearm_mesh: o3d.geometry.TriangleMesh,
) -> None:
    """Reject inputs that cannot yield a meaningful field.

    In particular, an absent hand pose is refused outright: a missing pose and
    a zero-depth frame are different facts, and once this field weights neural
    firing rates, collapsing them would silently down-weight real spikes.
    """
    if hand_mesh is None:
        raise ValueError(
            "hand_mesh is None. An absent hand pose must be recorded as absent by "
            "the caller — never as a zero-depth contact frame."
        )
    if forearm_mesh is None:
        raise ValueError("forearm_mesh is None; a reference terrain is required.")

    if len(hand_mesh.triangles) == 0:
        raise ValueError("hand_mesh has no triangles; cannot build a raycasting scene.")
    if len(forearm_mesh.triangles) == 0:
        raise ValueError("forearm_mesh has no triangles; cannot accumulate contact area.")

    # Checked before the AABB crop: a NaN vertex poisons the bounding box, and a
    # NaN bounding box crops to nothing, which would masquerade as "no contact".
    hand_vertices = np.asarray(hand_mesh.vertices)
    if not np.all(np.isfinite(hand_vertices)):
        raise ValueError(
            "hand_mesh contains non-finite vertices — a failed pose fit is "
            "propagating into contact detection."
        )


def _validate_field(
    points: np.ndarray,
    signed_depth: np.ndarray,
    max_plausible_depth_mm: float,
) -> None:
    """Enforce the :class:`ContactDepthFrame` postconditions before returning."""
    if len(points) != len(signed_depth):
        raise AssertionError(
            f"Contract violation: {len(points)} contact points vs "
            f"{len(signed_depth)} depth values; the field must be index-aligned "
            "with the contact points."
        )
    if not np.all(np.isfinite(signed_depth)):
        raise ValueError("Contact depth field contains non-finite values.")

    max_depth = float(np.max(np.abs(signed_depth)))
    if max_depth > max_plausible_depth_mm:
        raise ValueError(
            f"max(|signed_depth_mm|) = {max_depth} mm exceeds the sanity band of "
            f"{max_plausible_depth_mm} mm. The geometry is wrong: check that the "
            "hand and forearm meshes share the same coordinate space (Kinect "
            "Space 1) and that both are in millimetres."
        )
