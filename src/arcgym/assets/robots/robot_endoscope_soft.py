import logging
import math
import torch
import pdb
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, UsdShade
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors import CameraCfg, TiledCamera, TiledCameraCfg
from isaaclab.sim.spawners.lights import SphereLightCfg
from isaaclab.sensors import ContactSensorCfg, ContactSensor
from isaaclab.utils.math import quat_from_euler_xyz, combine_frame_transforms
from arcgym.assets.robots.base_robot import BaseRobot
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.math import quat_apply
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

class SoftEndoscopeChain(BaseRobot):
    def __init__(self, scene, config, isaac_cfg, init_pos, init_rot, device):
        self.config = config
        self.isaac_cfg = isaac_cfg
        self.scene = scene
        self.device = device
        self.use_camera = config["env_config"].get("use_camera", False)
        self.num_envs = scene.num_envs

        logging.info(f"Initializing Continuum Snake Robot at pos: {init_pos}")

        # Initial pose
        self.init_pos = torch.tensor(init_pos, device=device, dtype=torch.float32)
        self.init_rot = torch.tensor(init_rot, device=device, dtype=torch.float32)  # (w, x, y, z)
        self.env_init_pos = None
        
        # Track initialization state
        self._is_initialized = False

        # Build robot USD and create articulation config (but don't instantiate yet)
        self._build_robot()

        # Sensors
        if self.use_camera:
            logging.info("Creating egocentric camera...")
            self.egocamera = TiledCamera(self.isaac_cfg.camera_cfg)
            self.scene.sensors['egocamera'] = self.egocamera
            logging.info("Camera created")

        logging.info("Creating front light...")
        light_paths = f"/World/envs/env_.*/Robot/tip_light"
        self.egolight = sim_utils.spawn_light(
            prim_path=light_paths,
            cfg=self.isaac_cfg.light_cfg
        )
        logging.info("Light created")

        # Register - The scene will call spawn() on the articulation during its setup
        self.scene.articulations['robot'] = self.robot
        logging.info("Continuum robot registered with scene")

    def _build_robot(self):
        """Build multi-segment continuum robot with passive + active sections."""
        robot_config = self.config["robot_config"]
        PASSIVE_COLOR = Gf.Vec3f(0.5, 0.5, 0.5)  # Grey
        BASE_COLOR    = Gf.Vec3f(0.8, 0.2, 0.2)  # Red
        ACTIVE_COLOR  = Gf.Vec3f(0.2, 0.8, 0.2)  # Green
        num_passive = robot_config.get("num_passive_segments", robot_config.get("num_passive_links", 20))
        num_active = robot_config.get("num_active_segments", robot_config.get("num_active_links", 5))
        
        logging.info(f"Building robot with num_passive={num_passive}, num_active={num_active}")
        
        link_radius = robot_config.get("link_radius", 0.01)
        link_height = robot_config.get("link_height", 0.05)
        passive_stiffness = robot_config.get("passive_stiffness", 1e5)
        passive_damping = robot_config.get("passive_damping", 1e3)
        active_stiffness = robot_config.get("active_stiffness", 1e6)
        active_damping = robot_config.get("active_damping", 1e4)

        self.num_joints = num_passive - 1 + num_active
        self.active_joint_indices = list(range(num_passive - 1, num_passive - 1 + num_active))
        
        logging.info(f"Active joint indices: {self.active_joint_indices}, count: {len(self.active_joint_indices)}")

        prim_paths = f"/World/envs/env_.*/Robot"

        # === Define Actuators ===
        actuators = {}

        # Passive joints
        for i in range(1, num_passive):
            joint_name = f"passive_{i}_joint"
            actuators[joint_name] = ImplicitActuatorCfg(
                joint_names_expr=[f".*{joint_name}"],
                effort_limit=1e10,
                velocity_limit=100.0,
                stiffness=passive_stiffness,
                damping=passive_damping,
            )

        # Base joint
        actuators["base_joint"] = ImplicitActuatorCfg(
            joint_names_expr=[".*base_joint"],
            effort_limit=1e10,
            velocity_limit=100.0,
            stiffness=passive_stiffness,
            damping=passive_damping,
        )

        # Active joints
        for i in range(1, num_active + 1):
            joint_name = f"link_{i}_joint"
            actuators[joint_name] = ImplicitActuatorCfg(
                joint_names_expr=[f".*{joint_name}"],
                effort_limit=1e10,
                velocity_limit=100.0,
                stiffness=active_stiffness,
                damping=active_damping,
            )

        # === Build USD ===
        stage = self.scene.stage

        # Build USD for each environment
        for env_id in range(self.num_envs):
            env_path = f"/World/envs/env_{env_id}/Robot"

            # Root - Make it a rigid body so articulation root is properly defined
            root_prim = UsdGeom.Xform.Define(stage, env_path)
            root_prim_obj = root_prim.GetPrim()
            UsdPhysics.ArticulationRootAPI.Apply(root_prim_obj)
            
            # The root should NOT be a rigid body if the first link is the actual rigid body
            # This was causing the issue - remove these lines:
            # UsdPhysics.RigidBodyAPI.Apply(root_prim_obj)
            # mass_api = UsdPhysics.MassAPI.Apply(root_prim_obj)
            # mass_api.CreateDensityAttr().Set(100.0)

            prev_link_path = None

            # --- Passive Backbone ---
            for i in range(num_passive):
                link_name = f"passive_{i}"
                link_path = f"{env_path}/{link_name}"
                pos = Gf.Vec3f(-link_height * (num_passive - i - 0.5), 0.0, link_radius)
                self._create_link(stage, link_path, link_radius, link_height, pos,color=PASSIVE_COLOR)

                if i > 0:
                    axis = "Z" if i % 2 == 1 else "Y"
                    self._create_revolute_joint(
                        stage=stage,
                        joint_name=f"{link_name}_joint",
                        body0_path=prev_link_path,
                        body1_path=link_path,
                        local_pos0=Gf.Vec3f(link_height / 2, 0, 0),
                        local_pos1=Gf.Vec3f(-link_height / 2, 0, 0),
                        axis=axis,
                        stiffness=passive_stiffness,
                        damping=passive_damping,
                        target_pos=0.0
                    )
                prev_link_path = link_path

            # --- Base Link ---
            base_path = f"{env_path}/base_link"
            base_pos = Gf.Vec3f(0.0, 0.0, link_radius)
            self._create_link(stage, base_path, link_radius, link_height, base_pos,color=BASE_COLOR)

            axis = "Z" if num_passive % 2 == 1 else "Y"
            self._create_revolute_joint(
                stage=stage,
                joint_name="base_joint",
                body0_path=prev_link_path,
                body1_path=base_path,
                local_pos0=Gf.Vec3f(link_height / 2, 0, 0),
                local_pos1=Gf.Vec3f(-link_height / 2, 0, 0),
                axis=axis,
                stiffness=passive_stiffness,
                damping=passive_damping,
                target_pos=0.0
            )
            prev_link_path = base_path

            # --- Active Links ---
            for i in range(1, num_active + 1):
                link_name = f"link_{i}"
                link_path = f"{env_path}/{link_name}"
                pos = Gf.Vec3f(link_height * i, 0.0, link_radius)
                self._create_link(stage, link_path, link_radius, link_height, pos, color=ACTIVE_COLOR)

                axis = "Z" if i % 2 == 1 else "Y"
                self._create_revolute_joint(
                    stage=stage,
                    joint_name=f"{link_name}_joint",
                    body0_path=prev_link_path,
                    body1_path=link_path,
                    local_pos0=Gf.Vec3f(link_height / 2, 0, 0),
                    local_pos1=Gf.Vec3f(-link_height / 2, 0, 0),
                    axis=axis,
                    stiffness=active_stiffness,
                    damping=active_damping,
                    target_pos=0.0
                )
                prev_link_path = link_path

            # --- Tip Mount ---
            # Attach tip to the last active link with a fixed joint
            tip_path = f"{env_path}/tip"
            tip_prim = UsdGeom.Xform.Define(stage, tip_path)
            tip_prim.AddTranslateOp().Set(Gf.Vec3f((num_active + 0.5) * link_height, 0.0, link_radius))

            # Make tip a rigid body so it can be part of articulation
            tip_prim_obj = tip_prim.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(tip_prim_obj)

            # Create fixed joint to attach tip to last active link
            fixed_joint_path = f"{prev_link_path}/tip_fixed_joint"
            fixed_joint = UsdPhysics.FixedJoint.Define(stage, fixed_joint_path)
            fixed_joint.CreateBody0Rel().SetTargets([prev_link_path])
            fixed_joint.CreateBody1Rel().SetTargets([tip_path])
            fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(link_height / 2, 0, 0))  # +X end of last link
            fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))  # Center of tip
            fixed_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
            fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

            # # --- Create a hollow tube around the robot for this env ---
            # tube_path = f"{env_path}/Tube"
            # # position: center the tube on robot base; adjust Z or other axis if needed
            # tube_pos = Gf.Vec3f(0.2882,  0.2539, 0.0)  # tweak so tube encloses the robot
            # # choose sizes to enclose your robot (example values - tune to your robot dimensions)
            # inner_r = link_radius + 0.009    # slightly larger than robot outer radius
            # thickness = 0.003
            # length = (num_passive + num_active+15) * link_height  # span whole robot
            # self._create_hollow_tube(stage, tube_path,
            #                          inner_radius=inner_r,
            #                          thickness=thickness,
            #                          length=length,
            #                          axis="Z",
            #                          resolution=48,
            #                          segments_along=16,
            #                          position=tube_pos,
            #                          color=Gf.Vec3f(0.9, 0.9, 0.9),
            #                          static=True)


            # Diagnostic: list children under this environment to help detect missing prims
            try:
                env_parent_path = f"/World/envs/env_{env_id}"
                env_parent_prim = stage.GetPrimAtPath(env_parent_path)
                if env_parent_prim:
                    children = env_parent_prim.GetChildren()
                    logging.info(f"Env {env_id} children: {[c.GetName() for c in children]}")
                else:
                    logging.warning(f"Env parent prim not found at {env_parent_path}")
            except Exception as e:
                logging.warning(f"Failed listing children for env {env_id}: {e}")

        # === ArticulationCfg ===
        # The spawn=None tells IsaacLab that USD prims already exist
        articulation_cfg = ArticulationCfg(
            prim_path=prim_paths,
            spawn=None,  # USD already created above
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={".*": 0.0},
            ),
            actuators=actuators,
        )

        # Create the Articulation object
        # The scene's setup will properly initialize it later
        self.robot = Articulation(articulation_cfg)

        logging.info(f"Articulation created with {len(actuators)} actuator groups")

    def _create_link(self, stage, path, radius, height, position, color: Gf.Vec3f):
        cyl = UsdGeom.Cylinder.Define(stage, path)
        cyl.CreateRadiusAttr().Set(radius)
        cyl.CreateHeightAttr().Set(height)
        cyl.CreateAxisAttr().Set("X")
        cyl.AddTranslateOp().Set(position)
        cyl.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 0))

        prim = cyl.GetPrim()
        primvars_api = UsdGeom.PrimvarsAPI(prim)
        primvars_api.CreatePrimvar("displayColor", 
                                 Sdf.ValueTypeNames.Color3fArray, 
                                 UsdGeom.Tokens.constant).Set([color])
        
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateDensityAttr().Set(1000.0)
        
        # Enable contact reporting
        prim.SetCustomData({"physxContactReport:enabled": True})

    def _create_revolute_joint(self, stage, joint_name, body0_path, body1_path,
                               local_pos0, local_pos1, axis, stiffness, damping, target_pos):
        joint_path = f"{body0_path}/{joint_name}"
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([body0_path])
        joint.CreateBody1Rel().SetTargets([body1_path])
        joint.CreateAxisAttr().Set(axis)
        joint.CreateLocalPos0Attr().Set(local_pos0)
        joint.CreateLocalPos1Attr().Set(local_pos1)
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr().Set("position")
        drive.CreateTargetPositionAttr().Set(target_pos)
        drive.CreateStiffnessAttr().Set(stiffness)
        drive.CreateDampingAttr().Set(damping)
        drive.CreateMaxForceAttr().Set(1e10)

# ----------------------- paste below _create_revolute_joint -----------------------
    def _create_hollow_tube(self, stage, path, inner_radius=0.02, thickness=0.002,
                            length=1.0, axis="X", resolution=64, segments_along=8,
                            position=Gf.Vec3f(0.0, 0.0, 0.0), color=Gf.Vec3f(0.9,0.9,0.9),
                            static=True):
        """
        Create a hollow tube as a triangulated mesh and attach physics collision/rigid body.
        - path: prim path under stage (e.g., "/World/envs/env_0/Tube")
        - inner_radius: inner radius (meters)
        - thickness: shell thickness (meters)
        - length: tube length along chosen axis (meters)
        - axis: 'X'|'Y'|'Z' - axis the tube runs along (defaults to X to match robot)
        - resolution: number of vertices around circumference
        - segments_along: subdivisions along tube length (increases mesh smoothness)
        - position: translate the tube (Gf.Vec3f)
        - color: display color
        - static: if True the rigid body is static (non-moving)
        """
        from pxr import UsdGeom, Usd, Sdf, Gf, UsdPhysics

        outer_r = inner_radius + thickness
        # Build points: (segments_along+1) slices along length, each with resolution points for outer and inner
        points = []
        uvs = []  # optional
        faceVertexCounts = []
        faceVertexIndices = []

        # param along axis
        for seg in range(segments_along + 1):
            t = seg / float(segments_along)
            # x position along length centered at 0
            coord = (t - 0.5) * length
            for ring in (outer_r, inner_radius):
                for i in range(resolution):
                    theta = 2.0 * math.pi * i / resolution
                    y = math.cos(theta) * ring
                    z = math.sin(theta) * ring
                    if axis == "X":
                        p = (coord, y, z)
                    elif axis == "Y":
                        p = (y, coord, z)
                    else:
                        p = (y, z, coord)
                    points.append(Gf.Vec3f(*p))

        # Build faces: connect quads between successive segments, outer then inner (but we must maintain consistent winding)
        # Vertex indexing scheme:
        # slice_index * (2*resolution) + (ring_idx * resolution) + i
        def vid(slice_idx, ring_idx, i):
            return slice_idx * (2 * resolution) + (ring_idx * resolution) + i

        for s in range(segments_along):
            # connect outer shell (ring 0)
            for i in range(resolution):
                i_next = (i + 1) % resolution
                # Quad on outer: v0, v1, v2, v3  (we'll triangulate to two triangles)
                v0 = vid(s, 0, i)
                v1 = vid(s+1, 0, i)
                v2 = vid(s+1, 0, i_next)
                v3 = vid(s, 0, i_next)
                # triangle 1
                faceVertexCounts.append(3)
                faceVertexIndices.extend([v0, v1, v2])
                # triangle 2
                faceVertexCounts.append(3)
                faceVertexIndices.extend([v0, v2, v3])

            # connect inner shell (ring 1) - note winding reversed to make interior face normals point inward
            for i in range(resolution):
                i_next = (i + 1) % resolution
                v0 = vid(s, 1, i)
                v1 = vid(s, 1, i_next)
                v2 = vid(s+1, 1, i_next)
                v3 = vid(s+1, 1, i)
                faceVertexCounts.append(3)
                faceVertexIndices.extend([v0, v1, v2])
                faceVertexCounts.append(3)
                faceVertexIndices.extend([v0, v2, v3])

            # connect between outer and inner at ends to close shell thickness (create quads between outer/inner edges)
            for ring_end in (0, 1):
                # we create faces on the side (thin face) by connecting outer->inner at this slice to close thickness
                # For each i, connect outer(i) -> outer(i_next) -> inner(i_next) -> inner(i)
                # We'll do this per slice (s and s+1 will be handled by their own loops). Handle slice s here to generate thin faces along circumference.
                pass
        # The above creates the longitudinal shell faces; we also need radial faces closing thickness along circumference (between outer and inner)
        # We'll now add faces for each slice to connect outer->inner (closing shell thickness along circumference)
        for s in range(segments_along + 1):
            for i in range(resolution):
                i_next = (i + 1) % resolution
                outer_i = vid(s, 0, i)
                outer_next = vid(s, 0, i_next)
                inner_i = vid(s, 1, i)
                inner_next = vid(s, 1, i_next)
                # create two triangles for quad (outer_i, outer_next, inner_next, inner_i)
                faceVertexCounts.append(3)
                faceVertexIndices.extend([outer_i, outer_next, inner_next])
                faceVertexCounts.append(3)
                faceVertexIndices.extend([outer_i, inner_next, inner_i])

        # Define mesh prim
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr().Set(points)
        mesh.CreateFaceVertexCountsAttr().Set(faceVertexCounts)
        mesh.CreateFaceVertexIndicesAttr().Set(faceVertexIndices)
        # Optionally set normals/uvs; USD can compute normals if needed
        mesh.AddTranslateOp().Set(position)
        # visual color
        prim = mesh.GetPrim()
        primvars_api = UsdGeom.PrimvarsAPI(prim)
        primvars_api.CreatePrimvar("displayColor",
                                  Sdf.ValueTypeNames.Color3fArray,
                                  UsdGeom.Tokens.constant).Set([color])
        prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        # Physics: make static rigid body
        if static:
            # Static collision object
            UsdPhysics.CollisionAPI.Apply(prim)

            # PhysX: enable triangle mesh collision (needed for hollow geometry)
            from pxr import PhysxSchema

            physx = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            



        return mesh

    @staticmethod
    def make_isaac_config(config: dict):
        env_config = config["env_config"]
        robot_config = config["robot_config"]
        simulation_config = config["simulation_config"]
        discrete_action_space = env_config.get("discrete_action_space", False)
        camera_update_period = simulation_config["render_interval"] * simulation_config["dt"]
        camera_height, camera_width = robot_config["camera_resolution"]

        num_active_segments = robot_config.get("num_active_segments", robot_config.get("num_active_links", 5))

        CAMERA_CFG = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/tip/front_cam",
            update_period=camera_update_period,
            height=camera_height,
            width=camera_width,
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=robot_config["front_camera_focal_length"],
                focus_distance=robot_config["front_camera_focus_distance"],
                horizontal_aperture=robot_config["front_camera_horizontal_aperture"],
                clipping_range=robot_config["front_camera_clipping_range"],
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.03, 0.0, 0.0),
                rot=(1, 0, 0, 0),
                convention="world"
            ),
        )

        LIGHT_CFG = SphereLightCfg(
            color=robot_config["front_light_color"],
            color_temperature=robot_config["front_light_color_temperature"],
            intensity=robot_config["front_light_intensity"],
            radius=robot_config["front_light_radius"],
            exposure=robot_config["front_light_exposure"],
            visible=True,
        )

        class RobotContinuumSnakeCfg:
            camera_cfg = CAMERA_CFG
            light_cfg = LIGHT_CFG
            action_space = Discrete(6) if discrete_action_space else Box(low=-1.0, high=1.0, shape=(6,))
            observation_space = Dict({"rgb": Box(0, 255, shape=(3, camera_height, camera_width), dtype=np.uint8)})

        return RobotContinuumSnakeCfg()

    def apply_action(self, actions: torch.Tensor, action_scale: float = 1.0, **kwargs) -> None:
        """
        Apply 6-DoF action to continuum robot.

        Actions:
            [0] left/right (NOT USED)
            [1] up/down (NOT USED)
            [2] forward/backward (NOT IMPLEMENTED - needs prismatic joint)
            [3] Δpitch_total → total pitch bend (rad/s)
            [4] Δyaw_total → total yaw bend (rad/s)
            [5] roll/twist (NOT IMPLEMENTED - needs root rotation control)

        Args:
            actions: Action tensor
            action_scale: Scaling factor for actions
            **kwargs: Additional parameters for compatibility
        """
        if not self._is_initialized:
            logging.warning("Cannot apply action - robot not initialized yet")
            return
        
        # Extract action components
        forward_velocity = actions[:, 2]  # m/s (positive = forward, negative = backward)
        forward_velocity = forward_velocity * action_scale
        pitch_rate = actions[:, 3]    # rad/s
        yaw_rate = actions[:, 4]      # rad/s
        roll_rate = actions[:, 5]         # rad/s
        # Time step for velocity integration
        dt = 0.1  # Adjust based on your control frequency
        
        # ============================================================
        # 1. FORWARD/BACKWARD TRANSLATION
        # ============================================================
        # Get current root position and orientation

        current_root_state = self.robot.data.root_state_w.clone()
        current_pos = current_root_state[:, :3]  # (num_envs, 3)
        current_quat = current_root_state[:, 3:7]  # (num_envs, 4) - (w, x, y, z)
        
        # Convert quaternion to rotation matrix to get forward direction
        # Forward direction is along the robot's X-axis in local frame
        
        # Local X-axis (forward direction in robot frame)
        local_forward = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        
        # Transform to world frame
        world_forward = quat_apply(current_quat, local_forward)
        
        # Calculate translation
        translation = world_forward * forward_velocity.unsqueeze(1) * dt
        new_pos = current_pos + translation
        
        # Update root position
        new_root_state = current_root_state.clone()
        new_root_state[:, :3] = new_pos

        # ============================================================
        # 2. ROLL/TWIST ROTATION
        # ============================================================
        # Create incremental roll quaternion around X-axis (longitudinal axis)

        roll_angle = roll_rate * dt  # Convert rate to angle
        roll_quat = quat_from_euler_xyz(
            roll_angle, 
            torch.zeros_like(roll_angle), 
            torch.zeros_like(roll_angle)
        )  # (num_envs, 4)
        
        # Combine with current orientation
        new_quat = quat_mul(current_quat, roll_quat)
        new_root_state[:, 3:7] = new_quat
        
        # Write the new root state to simulation
        self.robot.write_root_state_to_sim(new_root_state)

        # ============================================================
        # 3. PITCH AND YAW BENDING (existing logic)
        # ============================================================
        # Get current joint positions
        current_joint_pos = self.robot.data.joint_pos[:, self.active_joint_indices]

        # Identify Y-axis (pitch) and Z-axis (yaw) joints
        # Note: Y-axis joints control pitch (up/down), Z-axis joints control yaw (left/right)
        y_joints = []  # Y-axis rotation joints (pitch control)
        z_joints = []  # Z-axis rotation joints (yaw control)

        for local_idx in range(len(self.active_joint_indices)):
            link_num = local_idx + 1
            if link_num % 2 == 1:
                z_joints.append(local_idx)  # Odd links have Z-axis joints
            else:
                y_joints.append(local_idx)  # Even links have Y-axis joints

        # Calculate target position increments
        new_joint_pos = current_joint_pos.clone()

        if len(y_joints) > 0:
            pitch_per_joint = (pitch_rate * dt) / len(y_joints)
            for idx in y_joints:
                new_joint_pos[:, idx] += pitch_per_joint

        if len(z_joints) > 0:
            yaw_per_joint = (yaw_rate * dt) / len(z_joints)
            for idx in z_joints:
                new_joint_pos[:, idx] += yaw_per_joint

        # Debug: Log actions if non-zero
        if torch.any(torch.abs(pitch_rate) > 0.01) or torch.any(torch.abs(yaw_rate) > 0.01):
            logging.debug(f"Actions - pitch: {pitch_rate[0]:.3f}, yaw: {yaw_rate[0]:.3f}, " +
                         f"pitch_per_joint: {pitch_per_joint[0] if len(y_joints) > 0 else 0:.4f}, " +
                         f"yaw_per_joint: {yaw_per_joint[0] if len(z_joints) > 0 else 0:.4f}")
        
        # Clamp to joint limits
        max_angle = math.pi / 4
        new_joint_pos = torch.clamp(new_joint_pos, -max_angle, max_angle)

        # Apply smoothing
        smoothing = 0.3
        smoothed_pos = current_joint_pos + smoothing * (new_joint_pos - current_joint_pos)

        # Create full joint position target array (all joints)
        full_joint_targets = self.robot.data.joint_pos.clone()
        # Update only the active joints
        full_joint_targets[:, self.active_joint_indices] = smoothed_pos

        # Set joint targets for all joints
        self.robot.set_joint_position_target(full_joint_targets)

    def get_observation(self, use_pose_in_obs=False, use_camera=None) -> dict:
        use_camera = use_camera if use_camera is not None else self.use_camera
        obs = {}

        if self._is_initialized:
            active_pos = self.robot.data.joint_pos[:, self.active_joint_indices]
            obs["joint_pos"] = active_pos
        else:
            obs["joint_pos"] = torch.zeros((self.num_envs, len(self.active_joint_indices)), device=self.device)

        if use_camera and hasattr(self, 'egocamera'):
            if hasattr(self.egocamera.data, 'output') and "rgb" in self.egocamera.data.output:
                rgb = self.egocamera.data.output["rgb"]
                obs["rgb"] = rgb.permute(0, 3, 1, 2)

        return obs

    def get_pose(self):
        """Get robot root state. Returns zeros if not yet initialized."""
        if not self._is_initialized:
            return torch.zeros((self.num_envs, 13), device=self.device)
        
        try:
            root_state = self.robot.data.root_state_w
            if torch.isnan(root_state).any():
                logging.error("Robot root state contains NaN values!")
                return torch.zeros((self.num_envs, 13), device=self.device)
            return root_state
        except (AttributeError, RuntimeError) as e:
            logging.warning(f"Robot data not yet available: {e}")
            return torch.zeros((self.num_envs, 13), device=self.device)

    def get_depth(self):
        if not hasattr(self.egocamera.data, 'output') or "depth" not in self.egocamera.data.output:
            raise ValueError("Depth data not available")
        return self.egocamera.data.output["depth"]

    def set_pos(self, init_pos, init_rot=None):
        if isinstance(init_pos, torch.Tensor):
            self.env_init_pos = init_pos.clone()
        else:
            self.init_pos = torch.tensor(init_pos, device=self.device)
            self.env_init_pos = self.init_pos.clone().unsqueeze(0)

        if init_rot is not None:
            self.init_rot = torch.tensor(init_rot, device=self.device)

    def _load_joint_positions_from_csv(self, csv_filepath: str, num_envs: int, env_ids: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Load joint positions and root state from a CSV file saved during teleoperation.

        IMPORTANT: This method loads the FULL configuration from env_0's row in the CSV file,
        including all joint angles and the root pose (position + orientation). For environments
        other than env_0, it applies the env_origins translation to position the robot correctly
        in each environment's coordinate space.

        This ensures all robots start with the same joint configuration and orientation,
        just translated to their respective environment positions.

        Args:
            csv_filepath: Path to the CSV file (e.g., "saved_states/robot_state_20251215_143022.csv")
            num_envs: Number of environments to generate positions for
            env_ids: Optional tensor of environment IDs to reset. If None, resets env_ids 0 to num_envs-1.

        Returns:
            Tuple of (joint_positions, root_state):
                - joint_positions: Joint positions tensor (same for all envs). Shape: (num_envs, num_joints)
                - root_state: Root state tensor with translated positions. Shape: (num_envs, 13)
        """
        import pandas as pd
        import os

        if not os.path.exists(csv_filepath):
            raise FileNotFoundError(f"CSV file not found: {csv_filepath}")

        # Load the CSV
        df = pd.read_csv(csv_filepath)

        # Determine which env_ids to reset
        if env_ids is None:
            env_ids_to_reset = list(range(num_envs))
        else:
            env_ids_to_reset = env_ids.cpu().tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)

        # Extract joint column names
        joint_cols = [col for col in df.columns if col.startswith('joint_') and col.endswith('_pos')]
        joint_cols = sorted(joint_cols, key=lambda x: int(x.split('_')[1]))  # Sort by joint number

        # ALWAYS load from env_0 row - this is the reference configuration
        if 0 not in df['env_id'].values:
            raise ValueError(f"env_id 0 not found in CSV file: {csv_filepath}")

        env0_row = df[df['env_id'] == 0].iloc[0]

        # Extract joint positions from env_0 (same for all environments)
        env0_joint_positions = [env0_row[col] for col in joint_cols]
        env0_joint_pos = torch.tensor(env0_joint_positions, dtype=torch.float32, device=self.device)

        # Extract root state from env_0
        env0_root_pos = torch.tensor([
            env0_row['root_pos_x'],
            env0_row['root_pos_y'],
            env0_row['root_pos_z']
        ], dtype=torch.float32, device=self.device)

        env0_root_quat = torch.tensor([
            env0_row['root_quat_w'],
            env0_row['root_quat_x'],
            env0_row['root_quat_y'],
            env0_row['root_quat_z']
        ], dtype=torch.float32, device=self.device)

        # Initialize output tensors
        joint_pos = torch.zeros((num_envs, len(joint_cols)), dtype=torch.float32, device=self.device)
        root_state = torch.zeros((num_envs, 13), dtype=torch.float32, device=self.device)

        # Calculate environment translations from scene.env_origins
        # env_origins gives the world-space origin of each environment
        env_origins = self.scene.env_origins  # Shape: (total_num_envs, 3)
        env0_origin = env_origins[0]  # Reference origin

        logging.info(f"env_origins[0]: {env0_origin}")
        logging.info(f"env0_root_pos from CSV: {env0_root_pos}")
        logging.info(f"env0_root_quat from CSV: {env0_root_quat}")

        # Apply configuration to each environment with appropriate translation
        for target_idx, env_id in enumerate(env_ids_to_reset):
            # Joint positions are the same for all environments
            joint_pos[target_idx] = env0_joint_pos

            # Calculate translation from env_0 to this environment
            translation = env_origins[env_id] - env0_origin

            logging.info(f"env_id={env_id}: translation={translation}")

            # Position: env_0's position + translation to this env's origin
            root_state[target_idx, :3] = env0_root_pos + translation

            # Orientation: same as env_0 (no rotation change)
            root_state[target_idx, 3:7] = env0_root_quat

            # Velocities: zero (fresh start)
            root_state[target_idx, 7:] = 0.0

            logging.info(f"env_id={env_id}: final root_pos={root_state[target_idx, :3]}")

        # Verify number of joints matches
        if joint_pos.shape[1] != self.robot.num_joints:
            raise ValueError(
                f"Number of joints in CSV ({joint_pos.shape[1]}) doesn't match robot ({self.robot.num_joints})"
            )

        logging.info(f"Loaded configuration from env_0 in {csv_filepath}, applied to {num_envs} environments")
        logging.info(f"env_0 root position: {env0_root_pos}")
        logging.info(f"Translated root positions: {root_state[:, :3]}")
        return joint_pos, root_state

    def generate_random_joint_positions(
        self,
        num_configs: int = None,
        max_angle: float = 0.5,
        smoothness: float = 0.8,
        ) -> torch.Tensor:
        """Generate reasonable random joint configurations for the robot.

        Creates smooth, physically plausible joint configurations by using adjacent
        joint correlation to avoid sharp bends.

        Args:
            num_configs: Number of configurations to generate. If None, generates for all envs.
            max_angle: Maximum joint angle in radians. Default: 0.5 (~28.6 degrees)
            smoothness: Smoothness factor (0-1). Higher values create smoother curves.
                       0 = completely random, 1 = very smooth gradual bending.

        Returns:
            Random joint positions. Shape: (num_configs, num_joints)
        """
        if num_configs is None:
            num_configs = self.num_envs

        num_joints = self.robot.num_joints

        # Generate base random values
        joint_positions = torch.randn(num_configs, num_joints, device=self.device)

        # Apply smoothing by correlating adjacent joints
        if smoothness > 0:
            # Apply exponential moving average across joints for smooth bending
            smoothed = torch.zeros_like(joint_positions)
            alpha = 1.0 - smoothness  # Convert to decay factor

            for env_idx in range(num_configs):
                smoothed[env_idx, 0] = joint_positions[env_idx, 0]
                for joint_idx in range(1, num_joints):
                    # Exponential moving average: smooth blend with previous joint
                    smoothed[env_idx, joint_idx] = (
                        alpha * joint_positions[env_idx, joint_idx] +
                        (1 - alpha) * smoothed[env_idx, joint_idx - 1]
                    )

            joint_positions = smoothed

        # Normalize and scale to desired range
        # Clamp to reasonable values first
        joint_positions = torch.clamp(joint_positions, -3.0, 3.0)

        # Scale to max_angle range
        joint_positions = joint_positions * (max_angle / 3.0)

        return joint_positions

    def reset(self, env_ids: torch.Tensor = None, joint_positions: torch.Tensor = None) -> None:
        """Reset robot to initial pose with specified joint positions.

        Args:
            env_ids: Environment indices to reset. If None, resets all environments.
            joint_positions: Desired joint positions for reset. Shape: (len(env_ids), num_joints).
                           If None, resets to zero joint positions (straight configuration).
                           Can pass 'random' as a string to generate random configurations.
                           Can pass a CSV filepath (str ending with '.csv') to load from saved state.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Check if loading from CSV file (includes both joint and root state)
        load_from_csv = isinstance(joint_positions, str) and joint_positions.endswith('.csv')

        if load_from_csv:
            # Load both joint positions and root state from CSV
            # Pass env_ids so each environment gets its own position from the CSV
            joint_pos, csv_root_state = self._load_joint_positions_from_csv(joint_positions, len(env_ids), env_ids)
            # Use root state from CSV
            root_state = csv_root_state
        else:
            # Get default root state
            root_state = self.robot.data.default_root_state[env_ids].clone()

            # Set position
            if self.env_init_pos is not None and self.env_init_pos.shape[0] == len(env_ids):
                root_state[:, :3] = self.env_init_pos
            else:
                origins = self.scene.env_origins[env_ids]
                root_state[:, :3] = origins + self.init_pos

            # Set orientation and zero velocities
            root_state[:, 3:7] = self.init_rot.repeat(len(env_ids), 1)
            root_state[:, 7:] = 0.0

            # Set joint positions (zero, random, or specified values) with zero velocities
            if joint_positions is None:
                joint_pos = torch.zeros((len(env_ids), self.robot.num_joints), device=self.device)
            elif isinstance(joint_positions, str):
                if joint_positions.lower() == 'random':
                    # Generate random smooth configurations
                    joint_pos = self.generate_random_joint_positions(num_configs=len(env_ids))
                else:
                    raise ValueError(f"Invalid string value for joint_positions: {joint_positions}")
            else:
                # Validate shape
                if joint_positions.shape != (len(env_ids), self.robot.num_joints):
                    raise ValueError(
                        f"joint_positions shape {joint_positions.shape} doesn't match expected shape "
                        f"({len(env_ids)}, {self.robot.num_joints})"
                    )
                joint_pos = joint_positions.to(self.device)

        # Reset the articulation internal state FIRST (this resets to defaults)
        self.robot.reset(env_ids)

        # Then write our custom state to simulation (this overwrites the defaults)
        self.robot.write_root_state_to_sim(root_state, env_ids)

        joint_vel = torch.zeros((len(env_ids), self.robot.num_joints), device=self.device)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Mark as initialized after first reset
        if not self._is_initialized:
            self._is_initialized = True
            logging.info("Robot initialization completed after first reset")
