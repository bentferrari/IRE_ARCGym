import logging

import torch
import omni.usd
from pxr import UsdGeom, Gf, Vt

def create_simple_sphere_markers(start_pos, target_positions, env_id):
    """
    Debugging function for creating 
    """
    if isinstance(start_pos, torch.Tensor):
        start_pos = start_pos.cpu().numpy().tolist()
    if isinstance(target_positions, torch.Tensor):
        target_positions = target_positions.cpu().numpy().tolist()
    
    stage = omni.usd.get_context().get_stage()
    
    # Create start marker (green sphere)
    start_prim = stage.DefinePrim(f"/World/start_marker_{env_id}", "Sphere")
    start_sphere = UsdGeom.Sphere(start_prim)
    start_sphere.GetRadiusAttr().Set(0.01)
    
    start_sphere.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.0, 1.0, 0.0)]))
    sphere_objects = [start_sphere]

    xform = UsdGeom.Xformable(start_prim)
    xform.AddTranslateOp().Set(Gf.Vec3f(*start_pos))

    if not isinstance(target_positions[0], list):
        target_positions = [target_positions]
    
    # Create target markers (red spheres)
    for i, target_pos in enumerate(target_positions):
        target_prim = stage.DefinePrim(f"/World/target_marker_{env_id}_{i}", "Sphere") 
        target_sphere = UsdGeom.Sphere(target_prim)
        target_sphere.GetRadiusAttr().Set(0.01)
        
        target_sphere.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(1.0, 0.0, 0.0)]))
        
        xform = UsdGeom.Xformable(target_prim)
        xform.AddTranslateOp().Set(Gf.Vec3f(*target_pos))
        sphere_objects.append(target_sphere)
    
    logging.info(f"Created colored markers: Start (green) at {start_pos}, Target (red) at {target_pos}")
    return sphere_objects