from __future__ import annotations
from collections.abc import Callable
from dataclasses import MISSING

import numpy as np
import trimesh
from typing import TYPE_CHECKING

import isaacsim.core.utils.prims as prim_utils
import omni.kit.commands
import omni.log

from pxr import Gf, Sdf, Semantics, Usd
from pxr import Gf, Sdf, Semantics, Usd, UsdPhysics
from isaaclab.sim import converters, schemas
from isaaclab.sim.utils import bind_physics_material, bind_visual_material, clone, select_usd_variants
from isaaclab.sim.spawners.materials import DeformableBodyMaterialCfg, RigidBodyMaterialCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import FileCfg

from isaaclab.utils import configclass

@clone
def spawn_from_mesh_file(
    prim_path: str,
    cfg: MeshFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Create a USD-Mesh prim from the given mesh file.
    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.
    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
    Returns:
        The created prim.
    Raises:
        ValueError: If a prim already exists at the given path.
    """
    # load the mesh from file
    mesh = trimesh.load_mesh(cfg.file_path)
    # spawn geometry if it doesn't exist.
    if not prim_utils.is_prim_path_valid(prim_path):
        prim_utils.create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)
    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    # check that invalid schema types are not used
    if cfg.deformable_props is not None and cfg.rigid_props is not None:
        raise ValueError("Cannot use both deformable and rigid properties at the same time.")
    if cfg.deformable_props is not None and cfg.collision_props is not None:
        raise ValueError("Cannot use both deformable and collision properties at the same time.")
    # check material types are correct
    if cfg.deformable_props is not None and cfg.physics_material is not None:
        if not isinstance(cfg.physics_material, DeformableBodyMaterialCfg):
            raise ValueError("Deformable properties require a deformable physics material.")
    if cfg.rigid_props is not None and cfg.physics_material is not None:
        if not isinstance(cfg.physics_material, RigidBodyMaterialCfg):
            raise ValueError("Rigid properties require a rigid physics material.")

    # create all the paths we need for clarity
    geom_prim_path = prim_path + "/geometry"
    mesh_prim_path = geom_prim_path + "/mesh"

    # create the mesh prim
    mesh_prim = prim_utils.create_prim(
        mesh_prim_path,
        prim_type="Mesh",
        scale=cfg.scale,
        attributes={
            "points": mesh.vertices,
            "faceVertexIndices": mesh.faces.flatten(),
            "faceVertexCounts": np.asarray([3] * len(mesh.faces)),
            "subdivisionScheme": "loop",
        },
    )

    # note: in case of deformable objects, we need to apply the deformable properties to the mesh prim.
    #   this is different from rigid objects where we apply the properties to the parent prim.
    if cfg.deformable_props is not None:
        # apply mass properties
        if cfg.mass_props is not None:
            schemas.define_mass_properties(mesh_prim_path, cfg.mass_props)
        # apply deformable body properties
        schemas.define_deformable_body_properties(mesh_prim_path, cfg.deformable_props)
    elif cfg.collision_props is not None:
        collision_approximation = "convexHull"
        # apply collision approximation to mesh
        # note: for primitives, we use the convex hull approximation -- this should be sufficient for most cases.
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        mesh_collision_api.GetApproximationAttr().Set(collision_approximation)
        # apply collision properties
        schemas.define_collision_properties(mesh_prim_path, cfg.collision_props)

    # apply visual material
    if cfg.visual_material is not None:
        if not cfg.visual_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.visual_material_path}"
        else:
            material_path = cfg.visual_material_path
        # create material
        cfg.visual_material.func(material_path, cfg.visual_material)
        #this a hacky way to deal with colon shader path
        material_path += "/OmniSurface"

        # apply material
        bind_visual_material(mesh_prim_path, material_path)

    # apply physics material
    if cfg.physics_material is not None:
        if not cfg.physics_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.physics_material_path}"
        else:
            material_path = cfg.physics_material_path
        # create material
        cfg.physics_material.func(material_path, cfg.physics_material)
        # apply material
        bind_physics_material(mesh_prim_path, material_path)

    # note: we apply the rigid properties to the parent prim in case of rigid objects.
    if cfg.rigid_props is not None:
        # apply mass properties
        if cfg.mass_props is not None:
            schemas.define_mass_properties(prim_path, cfg.mass_props)
        # apply rigid properties
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props)
    # return the prim
    return prim_utils.get_prim_at_path(prim_path)


@configclass
class MeshFileCfg(FileCfg):
    """Mesh file to spawn asset from.
    See :meth:`spawn_mesh_file` for more information.
    .. note::
        The configuration parameters include various properties. If not `None`, these properties
        are modified on the spawned prim in a nested manner.
        If they are set to a value, then the properties are modified on the spawned prim in a nested manner.
        This is done by calling the respective function with the specified properties.
    """

    func: Callable = spawn_from_mesh_file

    file_path: str = MISSING
    """Path to the mesh file (in OBJ, STL, or FBX format)."""

    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Scale of the mesh (in m)."""

    physics_material: RigidBodyMaterialCfg = RigidBodyMaterialCfg()
    """Physics material properties. Defaults to the default rigid body material."""

    physics_material_path: str = "PhysicsMaterial"

