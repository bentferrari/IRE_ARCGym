import os

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObjectCfg, RigidObjectCfg, DeformableObject, RigidObject
from isaaclab.sim.spawners.shapes import CapsuleCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.sim.spawners.materials import DeformableBodyMaterialCfg
import isaacsim.core.utils.stage as stage_utils

from pxr import PhysxSchema
import torch

from arcgym.utils.isaac_mesh import MeshFileCfg
import arcgym.utils.colon_utils as colon_utils


model_folder = os.path.dirname(__file__)
model_full_path = os.path.join(model_folder, "noncollapsed_0000_origincollid.usd")
# Rotation to lay colon horizontally
FLAT_ROTATION_Y = (0.7071068, 0, 0.7071068, 0)  # 90° rotation around Y-axis

obj_model_full_path = os.path.join(model_folder, "noncollapsed_0000_shell.obj")
shader_full_path = os.path.join(model_folder, "materials/colon_surface_material.usd")


COLON_GEOM_USD_CFG = UsdFileCfg(
                usd_path=model_full_path,
                scale=(0.001, 0.001, 0.001),
                deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, 
                                                                       contact_offset=0.001, 
                                                                       self_collision=False, 
                                                                       collision_simplification=False,
                                                                       simulation_hexahedral_resolution=8,     #simulation mesh resolution, default 10
                                                                       ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.3, 0.3)),
        )

COLON_GEOM_MESH_CFG = MeshFileCfg(
                file_path=obj_model_full_path,
                scale=(0.01, 0.01, 0.01),
                mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
                deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, 
                                                                       contact_offset=0.001, 
                                                                       self_collision=False, 
                                                                       collision_simplification=False,
                                                                       simulation_hexahedral_resolution=6, #16,    #simulation mesh resolution, default 10
                                                                       #sleep_damping=0.5,
                                                                       vertex_velocity_damping=5.0,
                                                                       ),
                #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.3, 0.3), opacity=1),  #seems not easy to get semi-transparent vis, have to turn on interactive rendering?
                visual_material=UsdFileCfg(usd_path=shader_full_path),
                physics_material=DeformableBodyMaterialCfg(
                        youngs_modulus=1000,
                        poissons_ratio=0.49, 
                        elasticity_damping=0.6,
                        ),
                
        )

COLON_CFG = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/Colon",
        spawn=COLON_GEOM_MESH_CFG,
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.1, 0, 2)),
        #rot=FLAT_ROTATION_Y,
        debug_vis=False, #this can be turned on/off for debugging visual markers, for instance, white spheres will be created if kinematic targets are set     
)

COLON_GEOM_MESH_RIGID_CFG = MeshFileCfg(
    file_path=obj_model_full_path,
    scale=(0.01, 0.01, 0.01),
    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=True,
        disable_gravity=True
    ),
    collision_props=sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=0.001,
        rest_offset=0.0,
    ),
    visual_material=UsdFileCfg(usd_path=shader_full_path),
    physics_material=sim_utils.RigidBodyMaterialCfg(
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.0,
    ),
)

COLON_RIGID_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Colon",
    spawn=COLON_GEOM_MESH_RIGID_CFG,
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.1, 0, 2)),
    #rot=FLAT_ROTATION_Y,
)

COLON_ENV_ATTACH_RECTUM_CFG = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Colon/AttachRectum",
        spawn=CapsuleCfg(
            radius=0.01,
            height=0.01,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True,),
            mass_props=sim_utils.MassPropertiesCfg(mass=2),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),  #no collisions will be incurred
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 1.0)),
            axis='Z',
            visible=True,
        )
)

COLON_ENV_ATTACH_DECEND_CFG = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Colon/AttachDescending",
        spawn=CapsuleCfg(
            radius=0.01,
            height=0.05,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True,),
            mass_props=sim_utils.MassPropertiesCfg(mass=2),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),  #no collisions will be incurred
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 1.0)),
            axis='Z',
            visible=True,
        )
)

COLON_ENV_ATTACH_SPLENIC_CFG = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Colon/AttachSplenic",
        spawn=CapsuleCfg(
            radius=0.01,
            height=0.01,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True,),
            mass_props=sim_utils.MassPropertiesCfg(mass=2),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),  #no collisions will be incurred
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 1.0)),
            axis='Z',
            visible=True,
        )
)

COLON_ENV_ATTACH_HEPATIC_CFG = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Colon/AttachHepatic",
        spawn=CapsuleCfg(
            radius=0.01,
            height=0.01,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True,),
            mass_props=sim_utils.MassPropertiesCfg(mass=2),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),  #no collisions will be incurred
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 1.0)),
            axis='Z',
            visible=True,
        )
)

COLON_ENV_ATTACH_CECUM_CFG = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Colon/AttachCecum",
        spawn=CapsuleCfg(
            radius=0.01,
            height=0.01,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True,),
            mass_props=sim_utils.MassPropertiesCfg(mass=2),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),  #no collisions will be incurred
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 1.0)),
            axis='Z',
            visible=True,
        )
)

class ColonModelCfg:
    is_rigid = False
    
    geom_mesh_cfg = COLON_GEOM_MESH_CFG
    colon_body_cfg = COLON_CFG
    colon_body_rigid_cfg = COLON_RIGID_CFG

    # colon_attach_rectum_cfg = COLON_ENV_ATTACH_RECTUM_CFG
    # colon_attach_descend_cfg = COLON_ENV_ATTACH_DECEND_CFG
    # colon_attach_splenic_cfg = COLON_ENV_ATTACH_SPLENIC_CFG
    # colon_attach_hepatic_cfg = COLON_ENV_ATTACH_HEPATIC_CFG
    # colon_attach_cecum_cfg = COLON_ENV_ATTACH_CECUM_CFG


class ColonModel:
    def __init__(self, scene, cfg: ColonModelCfg, init_pos=(0,0,2), init_rot=FLAT_ROTATION_Y, is_rigid=False):
        self.scene = scene
        self.cfg = cfg
        self.is_rigid = is_rigid
        self.colon_body = None
        self.init_pos = init_pos
        self.init_rot = init_rot
        self._env_translation = None

        self._setup()

    def _setup(self):
        if self.colon_body is None:
            if self.is_rigid:
                # Remove this line: from isaaclab.assets import RigidObject
                self.colon_body = RigidObject(cfg=self.cfg.colon_body_rigid_cfg.replace(
                    init_state=RigidObjectCfg.InitialStateCfg(pos=self.init_pos, rot=self.init_rot)
                ))
                self.scene.rigid_objects['colon'] = self.colon_body
            else:
                self.colon_body = DeformableObject(cfg=self.cfg.colon_body_cfg.replace(
                    init_state=DeformableObjectCfg.InitialStateCfg(pos=self.init_pos, rot=self.init_rot)
                ))
                self.scene.deformable_objects['colon'] = self.colon_body

    def _spawn_colon_env_objs(self):
        self.cfg.colon_attach_rectum_cfg.replace(init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 2, 1)))
        self.colon_attach_rectum = RigidObject(cfg=self.cfg.colon_attach_rectum_cfg)

        self.cfg.colon_attach_descend_cfg = COLON_ENV_ATTACH_DECEND_CFG.replace(init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 2, 2.5)))
        self.colon_attach_descend = RigidObject(cfg=self.cfg.colon_attach_descend_cfg)

        self.cfg.colon_attach_splenic_cfg = COLON_ENV_ATTACH_SPLENIC_CFG.replace(init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 2, 4)))
        self.colon_attach_splenic = RigidObject(cfg=self.cfg.colon_attach_splenic_cfg)

        self.cfg.colon_attach_hepatic_cfg = COLON_ENV_ATTACH_HEPATIC_CFG.replace(init_state=RigidObjectCfg.InitialStateCfg(pos=(5, 2, 4)))
        self.colon_attach_hepatic = RigidObject(cfg=self.cfg.colon_attach_hepatic_cfg)

        self.cfg.colon_attach_cecum_cfg = COLON_ENV_ATTACH_CECUM_CFG.replace(init_state=RigidObjectCfg.InitialStateCfg(pos=(2.5, 2, 2.5)))
        self.colon_attach_cecum = RigidObject(cfg=self.cfg.colon_attach_cecum_cfg)
    
    def _attach_colon_env_objs(self):
        # this is another way of using environment objects to create attachment, 
        # might be more intuitive than setting kinematic target as that was for nodals subject to simulation resolution
        # this might not be very efficient at each reset under a large number of envs though
        # can we create attachment at the isaaclab layer to avoid search matched colons?
        stage = stage_utils.get_current_stage()
        colon_prim_lst = sim_utils.utils.find_matching_prims(self.colon.cfg.prim_path, stage)

        for colon_prim in colon_prim_lst:
            colon_prim_path = colon_prim.GetPath()
            attachment_path = colon_prim_path.AppendElementString("attachment")
            attachment = PhysxSchema.PhysxPhysicsAttachment.Define(self.stage, attachment_path)
            attachment.GetActor0Rel().SetTargets([colon_prim_path])
            attachment.GetActor1Rel().SetTargets([
                colon_prim_path.AppendElementString("AttachRectum"),
                colon_prim_path.AppendElementString("AttachDescending"),
                colon_prim_path.AppendElementString("AttachSplenic"),
                colon_prim_path.AppendElementString("AttachHepatic"),
                colon_prim_path.AppendElementString("AttachCecum"),
            ])
            api = PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())
            api.CreateDeformableVertexOverlapOffsetAttr(0.001)

    def _attach_colon_nodals(self):
        if self.is_rigid:
            return  # Skip for rigid bodies
        #try to attach nodal of colon
        nodal_state = self.colon_body._data.default_nodal_state_w.clone()
        nodal_kinematic_target = self.colon_body._data.nodal_kinematic_target.clone()

        nodal_kinematic_target[..., :3] = nodal_state[..., :3]  #initialize target with initial nodal pos
        nodal_kinematic_target[..., 3] = 1                      #free for all of them

        #now attach rectum, descend, splenic, hectic, cecum nodals
        attaching_nodal_idx = colon_utils.extract_attach_nodals(nodal_pos=nodal_state[0, :, :3])
        
        # Get bottom vertices indices (same logic as entry position)
        xyz = nodal_state[0, :, :3]
        z = xyz[:, 2]
        bottom_mask = z < torch.quantile(z, 0.033)  # Same threshold as get_entry_pos
        bottom_nodal_idx = torch.where(bottom_mask)[0]
        
        # Ensure both tensors are on the same device
        attaching_nodal_idx = torch.tensor(attaching_nodal_idx, device=nodal_state.device)
        
        # Combine anatomical attachments with bottom vertex attachments
        all_attach_idx = torch.cat([attaching_nodal_idx, bottom_nodal_idx])
        
        # print("nodal to attach", all_attach_idx)
        nodal_kinematic_target[..., all_attach_idx, 3] = 0                 
        self.colon_body.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)
        self.colon_body.write_data_to_sim()

    @property
    def env_translation(self):
        if self._env_translation is None:
            self._env_translation = self.colon_body.data.root_pos_w - self.colon_body.data.root_pos_w[0]
        return self._env_translation

    def get_entry_pos(self, env_ids: torch.Tensor = None) -> torch.Tensor:
        """Get entry positions based on lowest mesh vertices."""
        # Get device from appropriate data attribute based on rigid/deformable
        if self.is_rigid:
            device = self.colon_body.data.body_state_w.device
        else:
            device = self.colon_body.data.nodal_state_w.device

        #bottom_center = torch.tensor([0.2882, 0.2539, 0.3108]).to(device)
        bottom_center = torch.tensor([0.2882, 0.2539, 0.3108]).to(device)
        bottom_center -= torch.tensor([0.2927, 0.1686, 0.4606]).to(device)

        #entry_pos = torch.tensor([0.7893,  0.0091,  0.2963]).to(device)
        entry_pos = torch.tensor([2,  0.5,  0.3]).to(device)
        delta_trans = torch.tensor([[0.0, 0.0, 0.0],
                                    # [0.0, 0.5, 0.0],
                                    # [-0.5, 0.0, 0.0],
                                    # [-0.5, 0.5, 0.0],
                                    # [-1.0, 0.0, 0.0]
                                    ]).to(device)
        #print("entry_pos + delta_trans:", entry_pos + delta_trans)
        #print("self.colon_body.data.root_pos_w:", self.colon_body.data.root_pos_w)
        
        #return bottom_center + self.colon_body.data.root_pos_w#self.env_translation
        return entry_pos + delta_trans
        #return entry_pos
    
    def get_targets(self, env_ids: torch.Tensor = None) -> torch.Tensor:
        entry_positions = self.get_entry_pos(env_ids)
        #print("entry_positions:", entry_positions)
        #env_translation = self.colon_body.data.nodal_state_w[:, 0, :3] - xyz[0, :]
        # base_target = torch.tensor(
        #         [[entry_positions[0,0], entry_positions[0,1], entry_positions[0,2]],
        #          [0.2849, 0.2097, 0.3526], # unfinished additional coordinates, please ignore
        #          [0.2977, 0.1968, 0.3533]]).to(entry_positions.device)
        base_target = torch.tensor([
            # [0.2882, 0.2539, 0.3108],
            # [0.2946, 0.2506, 0.3341],
            # [0.2972, 0.2269, 0.3446],
            # [0.2917, 0.2176, 0.3476],
            # [0.2899, 0.2150, 0.3484],
            # [0.2842, 0.2044, 0.3543],
            # [0.2877, 0.1887, 0.3600],
            # [0.3075, 0.1809, 0.3694],
            # [0.3356, 0.1627, 0.3941],
            # [0.3378, 0.1568, 0.3575],
            # [0.3340, 0.1559, 0.3386],
            [0.7795, -0.0453,  0.3561],
            ]).to(entry_positions.device)


        targets = []
        for translation in self.env_translation:
            targets.append(base_target + translation)
        print("targets", targets)
        return torch.stack(targets)

    def reset(self, env_ids: torch.Tensor = None):
        """Reset colon to initial state."""
        if env_ids is None:
            env_ids = torch.arange(self.colon_body.num_instances, device=self.colon_body.device)
        
        # Reset the existing deformable object
        self.colon_body.reset(env_ids)
        
        # Reapply nodal attachments
        #self._attach_colon_nodals()