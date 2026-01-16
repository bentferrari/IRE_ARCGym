import os
import logging

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

#obj_model_full_path = os.path.join(model_folder, "noncollapsed_0000_shell.obj")
#obj_model_full_path = os.path.join(model_folder, "shell_hole_0000.stl")
obj_model_full_path = os.path.join(model_folder, "outputconv_shell_hole_0000.obj")
#shader_full_path = os.path.join(model_folder, "materials/colon_surface_material_realistic.usd")  # Use realistic material
shader_full_path = os.path.join(model_folder, "materials/colon_surface_material.usd")  # Original material

COLON_GEOM_MESH_CFG = MeshFileCfg(
                file_path=obj_model_full_path,
                scale=(0.001, 0.001, 0.001),
                mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
                deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                                                                       rest_offset=0.0,        # Increased from 0.0 to prevent tunneling
                                                                       contact_offset=0.001,     # Increased from 0.0001 for better collision detection
                                                                       self_collision=False,
                                                                       collision_simplification=False,
                                                                       simulation_hexahedral_resolution=8, #16,    #simulation mesh resolution, default 10
                                                                       #sleep_damping=0.5,
                                                                       vertex_velocity_damping=5.0,
                                                                       ),
                #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.3, 0.3), opacity=1),  #seems not easy to get semi-transparent vis, have to turn on interactive rendering?
                visual_material=UsdFileCfg(usd_path=shader_full_path),
                physics_material=DeformableBodyMaterialCfg(
                        youngs_modulus=2e6,
                        poissons_ratio=0.49,
                        elasticity_damping=30,
                        ),

        )

COLON_GEOM_MESH_CFG_endoscope = MeshFileCfg(
                file_path=obj_model_full_path,
                scale=(0.01, 0.01, 0.01),
                mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
                deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                                                                       rest_offset=0.0,        # Negative value allows robot to get closer before collision
                                                                       contact_offset=0.00001,     # Reduced to 0 to eliminate invisible collision boundary
                                                                       self_collision=False,
                                                                       collision_simplification=False,
                                                                       simulation_hexahedral_resolution=6, #16,    #simulation mesh resolution, default 10
                                                                       #sleep_damping=0.5,
                                                                       vertex_velocity_damping=50.0,
                                                                       ),
                #visual_material=sim_utils.PreviewSurfaceCfg(
                #    diffuse_color=(0.95, 0.6, 0.55),  # More realistic pink/flesh color for colon
                #    roughness=0.7,  # Add surface roughness for organic appearance
                #    metallic=0.05,  # Slight wetness/shininess
                #    opacity=1.0
                #),
                visual_material=UsdFileCfg(usd_path=shader_full_path),  # Original shader - uncomment to use
                physics_material=DeformableBodyMaterialCfg(
                        youngs_modulus=2e6,
                        poissons_ratio=0.49,
                        elasticity_damping=60,
                        ),

        )

COLON_CFG = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/Colon",
        spawn=COLON_GEOM_MESH_CFG,
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.1, 0, 2)),
        #rot=FLAT_ROTATION_Y,
        debug_vis=False, #this can be turned on/off for debugging visual markers, for instance, white spheres will be created if kinematic targets are set     
)

COLON_CFG_endoscope = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/Colon",
        spawn=COLON_GEOM_MESH_CFG_endoscope,
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
    
    colon_body_cfg = COLON_CFG
    colon_body_cfg_endoscope = COLON_CFG_endoscope
    colon_body_rigid_cfg = COLON_RIGID_CFG

    # colon_attach_rectum_cfg = COLON_ENV_ATTACH_RECTUM_CFG
    # colon_attach_descend_cfg = COLON_ENV_ATTACH_DECEND_CFG
    # colon_attach_splenic_cfg = COLON_ENV_ATTACH_SPLENIC_CFG
    # colon_attach_hepatic_cfg = COLON_ENV_ATTACH_HEPATIC_CFG
    # colon_attach_cecum_cfg = COLON_ENV_ATTACH_CECUM_CFG


class ColonModel:
    def __init__(self, scene, cfg: ColonModelCfg, cfg1: dict, init_pos=(0,0,2), init_rot=FLAT_ROTATION_Y, is_rigid=False):
        self.scene = scene
        self.cfg = cfg
        self.cfg1 = cfg1
        self.robot_config = cfg1["robot_config"]  # Store robot_config as instance attribute
        self.is_rigid = is_rigid
        self.colon_body = None
        self.init_pos = init_pos
        self.init_rot = init_rot
        self._env_translation = None

        # Get attachment configuration from env_config
        self.use_txt_files_for_attachments = cfg1.get("env_config", {}).get("use_txt_files_for_attachments", True)

        self._setup()

    def _setup(self):
        robot_type = self.robot_config.get("robot_type", None)
        print("robot_type", robot_type)
        if self.colon_body is None:
            if self.is_rigid:
                # Remove this line: from isaaclab.assets import RigidObject
                self.colon_body = RigidObject(cfg=self.cfg.colon_body_rigid_cfg.replace(
                    init_state=RigidObjectCfg.InitialStateCfg(pos=self.init_pos, rot=self.init_rot)
                ))
                self.scene.rigid_objects['colon'] = self.colon_body
            else:
                if robot_type == "capsule":
                    self.colon_body = DeformableObject(cfg=self.cfg.colon_body_cfg.replace(
                        init_state=DeformableObjectCfg.InitialStateCfg(pos=self.init_pos, rot=self.init_rot)
                    ))
                    self.scene.deformable_objects['colon'] = self.colon_body
                else:
                    self.colon_body = DeformableObject(cfg=self.cfg.colon_body_cfg_endoscope.replace(
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

    def _attach_colon_nodals(self, env_ids: torch.Tensor = None, use_txt_files: bool = True):
        if self.is_rigid:
            return  # Skip for rigid bodies

        # If no env_ids specified, apply to all environments
        if env_ids is None:
            env_ids = torch.arange(self.colon_body.num_instances, device=self.colon_body.device)

        #try to attach nodal of colon
        nodal_state = self.colon_body._data.default_nodal_state_w.clone()
        nodal_kinematic_target = self.colon_body._data.nodal_kinematic_target.clone()

        # Initialize target with initial nodal pos for specified environments
        nodal_kinematic_target[env_ids, :, :3] = nodal_state[env_ids, :, :3]
        nodal_kinematic_target[env_ids, :, 3] = 1  # free for all of them

        if use_txt_files:
            # Load attachment indices from txt files (these are visual mesh indices)
            visual_vertex_indices = colon_utils.get_attachment_nodal_indices(
                model_id="0000",
                device=nodal_state.device
            )

            # Map visual mesh vertex indices to simulation mesh node indices
            # The txt files reference the high-res visual mesh, but we need simulation node indices
            all_attach_idx = colon_utils.map_visual_to_simulation_nodes(
                visual_mesh_path=obj_model_full_path,
                visual_vertex_indices=visual_vertex_indices,
                simulation_nodal_positions=nodal_state[0, :, :3]  # Use first env's nodal positions
            )
            print(f"Loaded {len(visual_vertex_indices)} visual vertices from txt files")
            print(f"Mapped to {len(all_attach_idx)} simulation nodes (out of {nodal_state.shape[1]} total nodes)")
        else:
            #now attach rectum, descend, splenic, hectic, cecum nodals (old method)
            # Use first environment's nodal positions to determine attachment indices
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

        # Apply attachments only to specified environments
        # print("nodal to attach", all_attach_idx)
        # Use advanced indexing to set attachments for specific environments
        for env_id in env_ids:
            nodal_kinematic_target[env_id, all_attach_idx, 3] = 0
        self.colon_body.write_nodal_kinematic_target_to_sim(nodal_kinematic_target[env_ids], env_ids)
        self.colon_body.write_data_to_sim()

    @property
    def env_translation(self):
        if self._env_translation is None:
            self._env_translation = self.colon_body.data.root_pos_w - self.colon_body.data.root_pos_w[0]
        return self._env_translation
    
    def get_accumulated_stress(self, env_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Calculates the total accumulated Von Mises stress on the colon.
        
        For Deformable: Returns sum of Von Mises stress across all mesh elements.
        For Rigid: Returns 0.0 (requires ContactSensor for true force).
        
        Returns:
            torch.Tensor: Shape (num_envs,), containing the scalar stress score.
        """
        if self.is_rigid:
            # Rigid bodies do not have internal stress. 
            # You must use a ContactSensor to get forces for rigid bodies.
            if env_ids is None:
                return torch.zeros(self.colon_body.num_instances, device=self.colon_body.device)
            return torch.zeros(len(env_ids), device=self.colon_body.device)

        else:
            # 1. Access the Stress Tensor
            # Shape: (num_instances, num_elements, 3, 3)
            # The property 'sim_element_stress_w' automatically fetches data from the PhysX view
            stress_tensor = self.colon_body.data.sim_element_stress_w
            
            if env_ids is not None:
                stress_tensor = stress_tensor[env_ids]

            # 2. Extract Components for Von Mises Calculation
            s11 = stress_tensor[..., 0, 0]
            s22 = stress_tensor[..., 1, 1]
            s33 = stress_tensor[..., 2, 2]
            s12 = stress_tensor[..., 0, 1]
            s23 = stress_tensor[..., 1, 2]
            s31 = stress_tensor[..., 2, 0]

            # 3. Calculate Von Mises Stress
            # Formula: sqrt(0.5 * [(s11-s22)^2 + (s22-s33)^2 + (s33-s11)^2 + 6*(s12^2 + s23^2 + s31^2)])
            von_mises = torch.sqrt(0.5 * (
                (s11 - s22)**2 + 
                (s22 - s33)**2 + 
                (s33 - s11)**2 + 
                6.0 * (s12**2 + s23**2 + s31**2)
            ))

            # 4. Sum over all elements to get total "impact" score per environment
            # Shape: (num_envs,)
            total_stress = torch.sum(von_mises, dim=1)
            
            return total_stress

    def get_entry_pos(self, env_ids: torch.Tensor = None, csv_filepath: str = None) -> torch.Tensor:
        """Get entry positions based on lowest mesh vertices or from CSV file.

        IMPORTANT: When loading from CSV, we load the position from env_0's row and
        translate it for other environments using env_origins. This ensures all robots
        start with the same relative position to their colon.

        Args:
            env_ids: Environment IDs to get positions for
            csv_filepath: Optional path to CSV file containing root positions

        Returns:
            Entry positions tensor of shape (num_envs, 3)
        """
        # Get device from appropriate data attribute based on rigid/deformable
        if self.is_rigid:
            device = self.colon_body.data.body_state_w.device
        else:
            device = self.colon_body.data.nodal_state_w.device

        # Determine env_ids
        if env_ids is not None:
            env_ids_tensor = env_ids if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device=device, dtype=torch.long)
        else:
            env_ids_tensor = torch.arange(self.colon_body.num_instances, device=device)
        env_count = int(env_ids_tensor.numel())

        # If CSV file is provided, load entry position from env_0 and translate for others
        if csv_filepath is not None and csv_filepath.endswith('.csv'):
            import pandas as pd
            import os

            if not os.path.exists(csv_filepath):
                raise FileNotFoundError(f"CSV file not found: {csv_filepath}")

            df = pd.read_csv(csv_filepath)

            # ALWAYS load from env_0 row - this is the reference position
            if 0 not in df['env_id'].values:
                raise ValueError(f"env_id 0 not found in CSV file: {csv_filepath}")

            env0_row = df[df['env_id'] == 0].iloc[0]
            env0_pos = torch.tensor([
                env0_row['root_pos_x'],
                env0_row['root_pos_y'],
                env0_row['root_pos_z']
            ], dtype=torch.float32, device=device)

            # Get environment origins for translation (relative to env_0)
            env_origins = self.env_translation

            # Create entry positions for all requested environments
            entry_positions = torch.zeros((env_count, 3), dtype=torch.float32, device=device)

            for idx, env_id in enumerate(env_ids_tensor.tolist()):
                # Get translation for this environment (relative to env_0)
                if env_id < len(env_origins):
                    translation = env_origins[env_id]
                else:
                    translation = torch.zeros(3, device=device)

                # Position: env_0's position + translation
                entry_positions[idx] = env0_pos + translation

            logging.info(f"Loaded entry position from env_0 in CSV: {csv_filepath}")
            logging.info(f"env_0 position: {env0_pos}")
            logging.info(f"Translated entry positions: {entry_positions}")
            return entry_positions

        # Default behavior - use hardcoded positions
        #bottom_center = torch.tensor([0.2882, 0.2539, 0.3108]).to(device)
        bottom_center = torch.tensor([0.2882, 0.2539, 0.3108]).to(device)
        bottom_center -= torch.tensor([0.2927, 0.1686, 0.4606]).to(device)
        
        # Colon lowest point
        # Env 0: x=4.8371, y=-0.8524, z=-2.0192
        # Env 1: x=4.8371, y=4.1476, z=-2.0192
        # Env 2: x=-0.1629, y=-0.8524, z=-2.0192
        # Env 3: x=-0.1629, y=4.1476, z=-2.0192
        # Env 4: x=-5.1629, y=-0.8524, z=-2.0192

        # 5_envs_endoscope: 7.4029,  0.5015,  0.5500
        # 1_env_endoscope: 2.3976, 2.9910, 0.5599
        # 5_envs_capsule: 1.0783, 0.5391, 0.1505
        # 5_envs_new_mesh: 5.6430,  -1.3124, -2.2
        # 1_env_new_mesh: x=0.643, 1.119, -2.017
        robot_type = self.robot_config.get("robot_type", None)
        if robot_type == "capsule":
            entry_pos = torch.tensor([1.1899, 0.4953, 0.1229]).to(device)
            delta_trans = torch.tensor([[0.0, 0.0, 0.0],
                                        [0.0, 0.5, 0.0],
                                        [-0.5, 0.0, 0.0],
                                        [-0.5, 0.5, 0.0],
                                        [-1.0, 0.0, 0.0]
                                        ]).to(device)
        else:
            entry_pos = torch.tensor([0.643, 1.119, -2.017]).to(device)
            delta_trans = torch.tensor([[0.0, 0.0, 0.0],
                                        [0.0, 5, 0.0],
                                        [-5, 0.0, 0.0],
                                        [-5, 5, 0.0],
                                        [-10, 0.0, 0.0]
                                        ]).to(device)

        #print("entry_pos + delta_trans:", entry_pos + delta_trans)
        #print("self.colon_body.data.root_pos_w:", self.colon_body.data.root_pos_w)

        #return bottom_center + self.colon_body.data.root_pos_w#self.env_translation
        entry_positions = entry_pos + delta_trans
        if env_ids is not None:
            env_ids_tensor = env_ids if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device=device, dtype=torch.long)
            entry_positions = entry_positions[env_ids_tensor]
        return entry_positions
        #return entry_pos
    
    def get_targets(self, env_ids: torch.Tensor = None, csv_filepath: str = None) -> torch.Tensor:
        entry_positions = self.get_entry_pos(env_ids)

        # Get device from appropriate data attribute based on rigid/deformable
        if self.is_rigid:
            device = self.colon_body.data.body_state_w.device
        else:
            device = self.colon_body.data.nodal_state_w.device

        # If CSV file is provided, load base_target from it
        if csv_filepath is not None and csv_filepath.endswith('.csv'):
            import pandas as pd
            import os

            if not os.path.exists(csv_filepath):
                raise FileNotFoundError(f"CSV file not found: {csv_filepath}")

            df = pd.read_csv(csv_filepath)

            # Extract root_xyz (root_pos_x, root_pos_y, root_pos_z) from CSV
            # Each row represents a different environment's target configuration
            base_targets = []
            for _, row in df.iterrows():
                # Skip rows with NaN values
                if pd.isna(row['root_pos_x']) or pd.isna(row['root_pos_y']) or pd.isna(row['root_pos_z']):
                    continue
                pos = [row['root_pos_x'], row['root_pos_y'], row['root_pos_z']]
                base_targets.append(pos)

            if not base_targets:
                raise ValueError(f"No valid target positions found in CSV: {csv_filepath}")

            # Use the first valid row as the base target
            base_target = torch.tensor([base_targets[0]], dtype=torch.float32, device=device)
            logging.info(f"Loaded base_target from CSV: {csv_filepath}")
            logging.info(f"Base target: {base_target}")
        else:
            # Default hardcoded base_target
            base_target = torch.tensor([
                [0.7795, -0.0453,  0.3561],
            ]).to(device)

        targets = []
        translations = self.env_translation
        if env_ids is not None:
            env_ids_tensor = env_ids if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device=device, dtype=torch.long)
            translations = translations[env_ids_tensor]
        for translation in translations:
            targets.append(base_target + translation)
        print("targets", targets)
        return torch.stack(targets)

    def get_lowest_position(self, env_id: int = 0) -> torch.Tensor:
        """Get the lowest (minimum Z coordinate) position of the colon.

        Args:
            env_id: Environment index to query. Default is 0.

        Returns:
            Tensor of shape (3,) containing [x, y, z] of the lowest point.
        """
        if self.is_rigid:
            # For rigid body, use body state
            # This might not give us individual vertices, so we use the body position
            body_pos = self.colon_body.data.body_state_w[env_id, :3]
            return body_pos
        else:
            # For deformable body, get all nodal positions
            nodal_positions = self.colon_body.data.nodal_state_w[env_id, :, :3]  # Shape: (num_nodes, 3)

            # Find the node with minimum Z coordinate
            z_coords = nodal_positions[:, 2]
            min_z_idx = torch.argmin(z_coords)
            lowest_pos = nodal_positions[min_z_idx]

            return lowest_pos

    def print_lowest_position(self, env_id: int = 0):
        """Print the lowest position of the colon.

        Args:
            env_id: Environment index to query. Default is 0.
        """
        lowest_pos = self.get_lowest_position(env_id)
        print(f"Colon lowest position (env {env_id}): x={lowest_pos[0]:.4f}, y={lowest_pos[1]:.4f}, z={lowest_pos[2]:.4f}")
        return lowest_pos

    def get_all_lowest_positions(self) -> torch.Tensor:
        """Get the lowest position for all environments.

        Returns:
            Tensor of shape (num_envs, 3) containing [x, y, z] of lowest point for each env.
        """
        if self.is_rigid:
            # For rigid body
            return self.colon_body.data.body_state_w[:, :3]
        else:
            # For deformable body
            num_envs = self.colon_body.data.nodal_state_w.shape[0]
            lowest_positions = torch.zeros((num_envs, 3), device=self.colon_body.device)

            for env_id in range(num_envs):
                nodal_positions = self.colon_body.data.nodal_state_w[env_id, :, :3]
                z_coords = nodal_positions[:, 2]
                min_z_idx = torch.argmin(z_coords)
                lowest_positions[env_id] = nodal_positions[min_z_idx]

            return lowest_positions

    def print_all_lowest_positions(self):
        """Print the lowest position for all environments."""
        lowest_positions = self.get_all_lowest_positions()
        print("All colons' lowest positions:")
        for env_id in range(lowest_positions.shape[0]):
            pos = lowest_positions[env_id]
            print(f"  Env {env_id}: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
        return lowest_positions

    def reset(self, env_ids: torch.Tensor = None):
        """Reset colon to initial state."""
        if env_ids is None:
            env_ids = torch.arange(self.colon_body.num_instances, device=self.colon_body.device)

        # Reset the existing deformable object
        self.colon_body.reset(env_ids)

        # For deformable bodies, explicitly restore nodal state to default
        # This ensures blown-away colons return to their original position
        if not self.is_rigid:
            # Get the default nodal state (position and velocity)
            default_nodal_state = self.colon_body._data.default_nodal_state_w.clone()

            # Check if any colons have been "blown away" (nodes moved too far from default)
            current_nodal_pos = self.colon_body._data.nodal_state_w[env_ids, :, :3]  # Positions only
            default_nodal_pos = default_nodal_state[env_ids, :, :3]

            # Calculate max displacement per environment
            displacement = torch.norm(current_nodal_pos - default_nodal_pos, dim=2)  # (num_envs, num_nodes)
            max_displacement_per_env = torch.max(displacement, dim=1)[0]  # (num_envs,)

            # Threshold for detecting blown-away colon (in meters)
            BLOWN_AWAY_THRESHOLD = 0.5  # Adjust based on your scene scale

            # Detect which environments have blown-away colons
            blown_away_mask = max_displacement_per_env > BLOWN_AWAY_THRESHOLD
            blown_away_env_ids = env_ids[blown_away_mask]

            if len(blown_away_env_ids) > 0:
                print(f"Detected blown-away colons in environments: {blown_away_env_ids.cpu().tolist()}")
                print(f"Max displacements: {max_displacement_per_env[blown_away_mask].cpu().tolist()}")

            # Write the default state back to the specified environments
            # This includes both position ([:3]) and velocity ([:3:6])
            self.colon_body._data.nodal_state_w[env_ids] = default_nodal_state[env_ids]

            # Write the nodal state to simulation
            self.colon_body.write_nodal_state_to_sim(self.colon_body._data.nodal_state_w[env_ids], env_ids)

        # Reapply nodal attachments (pass env_ids to only reset specified environments)
        #self._attach_colon_nodals(env_ids, use_txt_files=self.use_txt_files_for_attachments)
        self._attach_colon_nodals(env_ids, use_txt_files=False)
        # Print the lowest position after reset (for environment 0)
        if 0 in env_ids or env_ids.numel() == self.colon_body.num_instances:
            self.print_all_lowest_positions()
        print("reset!!!!!!!!!!!!!!!!!!")
