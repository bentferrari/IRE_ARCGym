import logging
import math
import torch
import pdb
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, UsdShade, PhysxSchema
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors import CameraCfg, TiledCamera, TiledCameraCfg
from isaaclab.sim.spawners.lights import SphereLightCfg
from isaaclab.sensors import ContactSensorCfg, ContactSensor
from isaaclab.utils.math import quat_apply
from arcgym.assets.robots.base_robot import BaseRobot
from isaaclab.actuators import ImplicitActuatorCfg

class RobotEndoscopeChain(BaseRobot):
    def __init__(self, scene, config, isaac_cfg, init_pos, init_rot, device, colon=None):
        self.config = config
        self.isaac_cfg = isaac_cfg
        self.scene = scene
        self.device = device
        self.use_camera = config["env_config"].get("use_camera", False)
        self.num_envs = scene.num_envs
        self.colon = colon  # Reference to ColonModel for stress calculation

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
        # Attach light to passive_0 at the camera position (+X end)
        light_paths = "/World/envs/env_.*/Robot/passive_0/front_light"
        link_height = self.config["robot_config"].get("link_height", 0.05)
        self.egolight = sim_utils.spawn_light(
            prim_path=light_paths,
            cfg=self.isaac_cfg.light_cfg,
            translation=(link_height, 0.0, 0.0)  # Same position as camera at +X end
        )
        logging.info("Light created")

        # Register - The scene will call spawn() on the articulation during its setup
        self.scene.articulations['robot'] = self.robot
        logging.info("Continuum robot registered with scene")

    def _build_robot(self):
        """Build multi-segment continuum robot with 20 passive links and a 6-DoF distal link."""
        robot_config = self.config["robot_config"]
        PASSIVE_COLOR = Gf.Vec3f(0.5, 0.5, 0.5)  # Grey
        BASE_COLOR    = Gf.Vec3f(0.8, 0.2, 0.2)  # Red (unused now but kept)
        ACTIVE_COLOR  = Gf.Vec3f(0.2, 0.8, 0.2)  # Green (unused now but kept)

        # Original config values (kept for compatibility, but overridden)
        num_passive_cfg = robot_config.get("num_passive_segments", robot_config.get("num_passive_links", 20))
        num_active_cfg = robot_config.get("num_active_segments", robot_config.get("num_active_links", 5))

        # === Enforce: 20 links total, all passive, last link on D6 joint ===
        num_links_total = robot_config.get("num_links_total", 20)
        num_passive = num_links_total
        num_active = 0  # no active section anymore

        logging.info(f"Building robot with TOTAL_LINKS={num_links_total} (num_passive={num_passive}, num_active={num_active})")
        
        link_radius = robot_config.get("link_radius", 0.01)
        link_height = robot_config.get("link_height", 0.05)
        passive_stiffness = robot_config.get("passive_stiffness", 1e2)
        passive_damping = robot_config.get("passive_damping", 1e3)

        # 20 links -> 19 joints in the chain. All passive, no "active" joint indices.
        self.num_joints = num_passive - 1
        self.active_joint_indices = []  # all joints passive; no actively-driven section
        
        logging.info(f"Active joint indices: {self.active_joint_indices}, count: {len(self.active_joint_indices)}")

        # Articulation root at passive_0 (the tip with camera)
        prim_paths = f"/World/envs/env_.*/Robot/passive_0"

        # === Define Actuators ===
        actuators = {}

        # Passive joints with PD actuators
        # passive_0 is the root (no joint)
        # passive_1 onwards connect via revolute joints (passive_1_joint, passive_2_joint, ...)
        for i in range(1, num_passive):
            joint_name = f"passive_{i}_joint"
            actuators[joint_name] = ImplicitActuatorCfg(
                joint_names_expr=[f".*{joint_name}"],
                effort_limit=1e1,
                velocity_limit=10.0,
                stiffness=passive_stiffness,
                damping=passive_damping,
            )

        # No base_joint, no active joints anymore
        # (All links passively follow the root via revolute joints with PD actuators)

        # === Build USD ===
        stage = self.scene.stage

        # Build USD for each environment
        for env_id in range(self.num_envs):
            env_path = f"/World/envs/env_{env_id}/Robot"

            # Build chain with passive_0 as the root
            # Structure: passive_0 (root, with camera) → passive_1 → passive_2 → ... → passive_19

            prev_link_path = None

            for i in range(num_passive):
                link_name = f"passive_{i}"
                link_path = f"{env_path}/{link_name}"
                # Generate robot along -X direction: passive_0 at x≈0, passive_1 at x≈-a, passive_2 at x≈-2a, etc.
                pos = Gf.Vec3f(-(i + 0.5) * link_height, 0.0, link_radius)
                self._create_link(stage, link_path, link_radius, link_height, pos, color=PASSIVE_COLOR)

                if i == 0:
                    # passive_0: articulation root (with camera and light at +X end)
                    link_prim = stage.GetPrimAtPath(link_path)
                    UsdPhysics.ArticulationRootAPI.Apply(link_prim)
                else:
                    # passive_1 onwards: revolute joints connecting to previous link
                    # Connect at -X end of previous link to +X end of current link
                    axis = "Z" if i % 2 == 1 else "Y"
                    self._create_revolute_joint(
                        stage=stage,
                        joint_name=f"{link_name}_joint",
                        body0_path=prev_link_path,
                        body1_path=link_path,
                        local_pos0=Gf.Vec3f(-link_height / 2, 0, 0),  # -X end of previous link
                        local_pos1=Gf.Vec3f(link_height / 2, 0, 0),   # +X end of current link
                        axis=axis,
                        stiffness=passive_stiffness,
                        damping=passive_damping,
                        target_pos=0.0
                    )

                prev_link_path = link_path

            # --- No Base Link / Active Links / Tip ---
            # passive_0 is the root, passive_19 is the distal end

            # --- Create a hollow tube around the robot for this env ---
            # tube_path = f"{env_path}/Tube"
            # tube_pos = Gf.Vec3f(0.2882,  0.2539, 0.0)  # tweak so tube encloses the robot
            # inner_r = link_radius + 0.009
            # thickness = 0.003
            # length = (num_passive + 15) * link_height  # span whole robot
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
        self.robot = Articulation(articulation_cfg)

        logging.info(f"Articulation created with {len(actuators)} actuator groups")

    def _create_link(self, stage, path, radius, height, position, color: Gf.Vec3f):
        cyl = UsdGeom.Capsule.Define(stage, path)
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
        #mass_api.CreateDensityAttr().Set(1000.0)
        mass_api.CreateMassAttr().Set(100)

        # Disable gravity for this rigid body
        physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        physx_rigid_body_api.CreateDisableGravityAttr().Set(True)

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

        # Set joint limits to prevent excessive bending
        # Limit to ±30 degrees (±0.524 radians)
        joint.CreateLowerLimitAttr().Set(-1.0)  # degrees
        joint.CreateUpperLimitAttr().Set(1.0)   # degrees

        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr().Set("position")
        drive.CreateTargetPositionAttr().Set(target_pos)
        drive.CreateStiffnessAttr().Set(stiffness)
        drive.CreateDampingAttr().Set(damping)
        drive.CreateMaxForceAttr().Set(1e2)

    def _create_d6_joint(self, stage, joint_name, body0_path, body1_path,
                        local_pos0, local_pos1):
        """
        Create a PhysX D6-like free joint using UsdPhysics.Joint + PhysxJointAPI
        compatible with IsaacLab (no CreateMotionXXXRel helpers).
        Distal link is fully free relative to previous link.
        """

        joint_path = f"{body0_path}/{joint_name}"

        # 1. Generic USD joint prim
        joint = UsdPhysics.Joint.Define(stage, joint_path)
        prim = joint.GetPrim()

        # 2. Attach PhysX joint API
        physx_joint = PhysxSchema.PhysxJointAPI.Apply(prim)

        # 3. Attach bodies
        joint.CreateBody0Rel().SetTargets([body0_path])
        joint.CreateBody1Rel().SetTargets([body1_path])

        # 4. Local frames
        joint.CreateLocalPos0Attr().Set(local_pos0)
        joint.CreateLocalPos1Attr().Set(local_pos1)
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

        # 5. Set all 6 motions to "free"
        # Linear DOF: X Y Z
        prim.CreateAttribute("physics:linear:motionX", Sdf.ValueTypeNames.Token).Set("free")
        prim.CreateAttribute("physics:linear:motionY", Sdf.ValueTypeNames.Token).Set("free")
        prim.CreateAttribute("physics:linear:motionZ", Sdf.ValueTypeNames.Token).Set("free")

        # Angular DOF: X, Y, Z
        prim.CreateAttribute("physics:angular:motionX", Sdf.ValueTypeNames.Token).Set("free")
        prim.CreateAttribute("physics:angular:motionY", Sdf.ValueTypeNames.Token).Set("free")
        prim.CreateAttribute("physics:angular:motionZ", Sdf.ValueTypeNames.Token).Set("free")

        return prim





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
        link_height = robot_config.get("link_height", 0.5)
        # Camera attached to passive_0 (the free-floating tip with D6 joint)
        CAMERA_CFG = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/passive_0/front_cam",
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
                # Put camera directly in front of the capsule tip
                pos=(link_height, 0.0, 0.0),

                # Rotate camera 90° around Y-axis so it looks along +X axis, then 180° around X-axis (upside down)
                rot=(-0.5, 0.5, 0.5, -0.5),

                # Offset is relative to passive_0 frame
                convention="local"
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

    def apply_action(self, actions: torch.Tensor, action_scale: float = 1.0) -> None:
        """
        Apply 6-DoF velocity commands directly to the robot root (passive_0).

        Actions (per env):
            [0] v_left_right   (m/s along local +Y, -Y)
            [1] v_up_down      (m/s along local +Z, -Z)
            [2] v_forward_back (m/s along local +X, -X)
            [3] ω_roll         (rad/s around local X)
            [4] ω_pitch        (rad/s around local Y)
            [5] ω_yaw          (rad/s around local Z)

        passive_0 (with camera) is the articulation root. The rest of the chain passively follows.
        """
        if not self._is_initialized:
            logging.warning("Cannot apply action - robot not initialized yet")
            return

        # Scale actions
        actions = actions * action_scale
        print("actions", actions)

        v_lr   = actions[:, 0]  # left/right
        v_ud   = actions[:, 1]  # up/down
        v_fb   = actions[:, 2]  # forward/back
        w_roll = actions[:, 5]  # roll
        w_pitch= -actions[:, 3]  # pitch
        w_yaw  = actions[:, 4]  # yaw

        # Get current orientation to transform velocities from local to world frame
        current_quat = self.robot.data.root_state_w[:, 3:7]  # (num_envs, 4)

        # ------------------------------------------------------------
        # 1. LINEAR VELOCITY (transform from local to world frame)
        # ------------------------------------------------------------
        # Local velocity vector [forward, left/right, up/down]
        local_linear_vel = torch.stack([v_fb, v_lr, v_ud], dim=1)  # (num_envs, 3)

        # Transform to world frame
        world_linear_vel = quat_apply(current_quat, local_linear_vel)

        # ------------------------------------------------------------
        # 2. ANGULAR VELOCITY (transform from local to world frame)
        # ------------------------------------------------------------
        # Local angular velocity vector [roll, pitch, yaw]
        local_angular_vel = torch.stack([w_roll, w_pitch, w_yaw], dim=1)  # (num_envs, 3)

        # Transform to world frame
        world_angular_vel = quat_apply(current_quat, local_angular_vel)

        # ------------------------------------------------------------
        # 3. WRITE VELOCITIES TO SIMULATION
        # ------------------------------------------------------------
        # Combine into root velocity: [linear_vel (3), angular_vel (3)]
        root_velocity = torch.cat([world_linear_vel, world_angular_vel], dim=1)  # (num_envs, 6)

        # Write directly to simulation
        self.robot.write_root_velocity_to_sim(root_velocity)

        # Get accumulated stress from colon after applying action
        # if self.colon is not None:
        #     try:
        #         stress = self.colon.get_accumulated_stress()
        #         print(f"Colon accumulated stress: {stress}")
        #     except Exception as e:
        #         logging.debug(f"Could not get colon stress: {e}")


    def get_observation(self, use_pose_in_obs=False, use_camera=None) -> dict:
        use_camera = use_camera if use_camera is not None else self.use_camera
        obs = {}

        # Only include joint_pos if there are active joints
        if len(self.active_joint_indices) > 0:
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
        print("initial_joint",joint_positions)

        return joint_positions

    def reset(self, env_ids: torch.Tensor = None, joint_positions: torch.Tensor = None) -> None:
        """Reset robot to initial pose with specified joint positions.

        Args:
            env_ids: Environment indices to reset. If None, resets all environments.
            joint_positions: Desired joint positions for reset. Shape: (len(env_ids), num_joints).
                           If None, resets to zero joint positions (straight configuration).
                           Can pass 'random' as a string to generate random configurations.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

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

        # Write state to simulation
        self.robot.write_root_state_to_sim(root_state, env_ids)

        # Set joint positions (zero, random, or specified values) with zero velocities
        if joint_positions is None:
            joint_pos = torch.zeros((len(env_ids), self.robot.num_joints), device=self.device)
        elif isinstance(joint_positions, str) and joint_positions.lower() == 'random':
            # Generate random smooth configurations
            joint_pos = self.generate_random_joint_positions(num_configs=len(env_ids))
        else:
            # Validate shape
            if joint_positions.shape != (len(env_ids), self.robot.num_joints):
                raise ValueError(
                    f"joint_positions shape {joint_positions.shape} doesn't match expected shape "
                    f"({len(env_ids)}, {self.robot.num_joints})"
                )
            joint_pos = joint_positions.to(self.device)

        joint_vel = torch.zeros((len(env_ids), self.robot.num_joints), device=self.device)

        # Write joint state to simulation first
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Compute and set individual link states based on joint configuration using forward kinematics
        # This directly sets each link's pose in the simulation to match the joint configuration
        self._reset_link_states_directly(env_ids, joint_pos)

        # Mark as initialized after first reset
        if not self._is_initialized:
            self._is_initialized = True
            logging.info("Robot initialization completed after first reset")

    def _reset_link_states_directly(self, env_ids: torch.Tensor, joint_positions: torch.Tensor) -> None:
        """Directly set each link's state in simulation based on joint configuration.

        This method computes each link's position and orientation using forward kinematics,
        then directly writes these poses to the PhysX simulation, bypassing joint constraints.
        This ensures links are immediately positioned correctly without waiting for physics settling.

        Args:
            env_ids: Environment indices to reset.
            joint_positions: Joint positions for the configuration. Shape: (len(env_ids), num_joints).
        """
        from isaaclab.utils import math as math_utils

        link_radius = self.config["robot_config"].get("link_radius", 0.01)
        link_height = self.config["robot_config"].get("link_height", 0.05)

        # Get the root state for calculating relative link positions
        root_state = self.robot.data.root_state_w[env_ids]
        root_pos = root_state[:, :3]  # (len(env_ids), 3)
        root_quat = root_state[:, 3:7]  # (len(env_ids), 4) - (w, x, y, z)

        # Compute all link poses using forward kinematics
        link_poses = self._compute_forward_kinematics(
            root_pos, root_quat, joint_positions, link_height, link_radius
        )

        # Access the PhysX view to directly set body transforms
        physx_view = self.robot.root_physx_view

        # Convert env_ids to the format expected by PhysX
        if isinstance(env_ids, torch.Tensor):
            physx_env_ids = env_ids.cpu().numpy()
        else:
            physx_env_ids = env_ids

        # Try to directly set link transforms in PhysX
        # This bypasses the joint constraints and directly positions each body
        try:
            # Get all body transforms for the environments
            # Shape: (num_envs, num_bodies, 7) where 7 = [pos(3), quat(4) in xyzw]
            num_bodies = self.robot.num_bodies

            # Convert quaternions from wxyz to xyzw for PhysX
            link_poses_xyzw = link_poses.clone()
            link_poses_xyzw[:, :, 3:] = math_utils.convert_quat(link_poses_xyzw[:, :, 3:], to="xyzw")

            # Set body poses for all links at once
            # Reshape to (num_envs * num_bodies, 7) for batch setting
            all_poses = link_poses_xyzw.reshape(-1, 7)
            all_velocities = torch.zeros((len(env_ids) * num_bodies, 6), device=self.device)

            # Create body indices: [0, 1, 2, ..., num_bodies-1] repeated for each env
            body_indices = torch.arange(num_bodies, device=self.device).repeat(len(env_ids))

            # Create env indices: [env_0]*num_bodies + [env_1]*num_bodies + ...
            env_indices_repeated = torch.repeat_interleave(
                torch.tensor(physx_env_ids, device=self.device), num_bodies
            ).cpu().numpy()

            # Try the batch set method
            if hasattr(physx_view, 'set_link_transforms'):
                physx_view.set_link_transforms(
                    all_poses,
                    link_indices=body_indices.cpu().numpy(),
                    indices=env_indices_repeated
                )
                physx_view.set_link_velocities(
                    all_velocities,
                    link_indices=body_indices.cpu().numpy(),
                    indices=env_indices_repeated
                )
                logging.debug(f"Successfully set link transforms for {len(env_ids)} environments")
            else:
                # Fallback: Set each link individually
                logging.debug("Using per-link transform setting (no batch API available)")
                for link_idx in range(num_bodies):
                    link_pose = link_poses_xyzw[:, link_idx, :]
                    link_vel = torch.zeros((len(env_ids), 6), device=self.device)

                    if hasattr(physx_view, 'set_body_transforms'):
                        for i, env_id in enumerate(physx_env_ids):
                            physx_view.set_body_transforms(
                                link_pose[i:i+1],
                                body_indices=[link_idx],
                                indices=[env_id]
                            )
                            physx_view.set_body_velocities(
                                link_vel[i:i+1],
                                body_indices=[link_idx],
                                indices=[env_id]
                            )
                    else:
                        if link_idx == 0:
                            logging.warning(
                                "PhysX view doesn't support direct link transform setting. "
                                "Relying on joint positions only. Links may take time to settle."
                            )
                        break

        except (AttributeError, RuntimeError, TypeError) as e:
            logging.debug(f"Could not directly set link states: {e}. Using joint-based positioning.")
            # Not a critical error - joint positions will still work, just may need settling time

    def _compute_forward_kinematics(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        joint_positions: torch.Tensor,
        link_height: float,
        link_radius: float,
    ) -> torch.Tensor:
        """Compute forward kinematics to get link poses from joint positions.

        This computes the world position and orientation of each link based on the
        kinematic chain and joint angles.

        Args:
            root_pos: Root link positions. Shape: (num_envs, 3)
            root_quat: Root link orientations (w,x,y,z). Shape: (num_envs, 4)
            joint_positions: Joint angles. Shape: (num_envs, num_joints)
            link_height: Length of each link
            link_radius: Radius of each link

        Returns:
            Link poses (position + quaternion) for all links. Shape: (num_envs, num_bodies, 7)
        """
        from isaaclab.utils import math as math_utils

        num_envs = len(root_pos)
        num_bodies = self.robot.num_bodies

        # Initialize output: (num_envs, num_bodies, 7) where 7 = [pos(3), quat(4)]
        link_poses = torch.zeros((num_envs, num_bodies, 7), device=self.device)

        # Link 0 (passive_0) is the root - its pose is the root pose
        link_poses[:, 0, :3] = root_pos
        link_poses[:, 0, 3:7] = root_quat

        # For subsequent links, compute pose based on previous link and joint angle
        # Joint i connects link i-1 to link i
        for link_idx in range(1, num_bodies):
            joint_idx = link_idx - 1  # Joint index (joint_1 connects passive_0 to passive_1)

            # Get previous link's pose
            prev_pos = link_poses[:, link_idx - 1, :3]
            prev_quat = link_poses[:, link_idx - 1, 3:7]

            # Joint rotation axis alternates between Y and Z
            # passive_1 (joint_idx=0) uses Z, passive_2 (joint_idx=1) uses Y, etc.
            if joint_idx % 2 == 0:
                axis = "Z"
            else:
                axis = "Y"

            # Get joint angle for this joint
            joint_angle = joint_positions[:, joint_idx]  # (num_envs,)

            # Create rotation quaternion for the joint angle
            if axis == "Z":
                # Rotation around Z axis
                axis_vec = torch.tensor([0.0, 0.0, 1.0], device=self.device)
            else:  # axis == "Y"
                # Rotation around Y axis
                axis_vec = torch.tensor([0.0, 1.0, 0.0], device=self.device)

            # Create quaternion from axis-angle
            joint_quat = math_utils.quat_from_angle_axis(joint_angle, axis_vec.repeat(num_envs, 1))

            # Local offset from previous link center to joint (at -X end of previous link)
            local_offset_to_joint = torch.zeros((num_envs, 3), device=self.device)
            local_offset_to_joint[:, 0] = -link_height / 2  # Joint at -X end

            # Transform to world frame
            world_offset_to_joint = quat_apply(prev_quat, local_offset_to_joint)
            joint_pos = prev_pos + world_offset_to_joint

            # Current link orientation = previous orientation * joint rotation
            current_quat = math_utils.quat_mul(prev_quat, joint_quat)

            # Local offset from joint to current link center (at +X end of current link in local frame)
            local_offset_to_center = torch.zeros((num_envs, 3), device=self.device)
            local_offset_to_center[:, 0] = link_height / 2  # Link center at +X/2 from joint

            # Transform to world frame using current orientation
            world_offset_to_center = quat_apply(current_quat, local_offset_to_center)
            current_pos = joint_pos + world_offset_to_center

            # Store link pose
            link_poses[:, link_idx, :3] = current_pos
            link_poses[:, link_idx, 3:7] = current_quat

        return link_poses
