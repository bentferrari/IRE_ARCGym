## Autonomous Robotic Colonoscopy Gym Environments (ARCGym)

This is a prototype for ARCGym based on Isaac Sim 4.5 and Isaac Lab. The purpose is to explore the technical feasibility of using Isaac Sim as the physics engine and Isaac Lab as the gym basis.

## Install and usage

We use uv to manage the project. Install python-uv to your system python packages or your base python environment and do.

```console
cd arcgym
uv run scripts/arc_env_rl.py --num_envs 3 --train --headless
```
Alternatively, if already sync with uv (e.g. the virtualenv is installed), then you can do
```console
source .venv/bin/activate
```
followed by
```console
uv run scripts/arc_env_rl.py --num_envs 3 --train --headless
```

An Isaac Sim viewport will be spawned if you do not use `--headless`. Switch between main camera (perspective) and FPV camera by clicking the camera icon in the middle of the top of the viewport. If you do not have `--train`, you can use keyboard (A/D/S/W/Q/E/U/J/I/K/O/L) to move the robot.

## Findings

The reasons for picking-up Isaac Sim/Lab eco-system could be:
* Inherent support for parallel environment population and interfacing to RL libraries. The doc says both rl_games and SB3 are supported. There is a parameter to set the number of environments and it seems all the data will be batched automatically as well.
* It might be easier to connect some photorealistic rendering pipeline as what NVIDIA is selling for other robotics task environments. Will a good ray-tracing be necessary for training policies upon realistic visual input? Note that a preliminary test by turning on RTX trace will significantly slow down the simulation for our current model resolution. Would that be fixed by better hardware or correct software setup?
* They claim interoperability with engines like mujoco and warp in Isaac Lab's roadmap. So in theory there could be hope for our gym environment to exploit more flexibility from warp as it goes mature.
* The officially endorsed [health repo](https://developer.nvidia.com/blog/introducing-nvidia-isaac-for-healthcare-an-ai-powered-medical-robotics-development-platform/) is built upon Isaac Lab as well. There might be opportunities to utilize NVIDIA's communication channels to boost IRE's visibility. 

The cons are mainly:
* The underlying engine PhysX simulates deformables with linear FEM or PBD. Not tried PBD yet but it is unclear how critical a nonlinear FEM would be necessary.
* Isaac Sim is not open-sourced so unless the buggy part is from Issac Lab we have to wait for NVIDIA's fix if there is no workaround for a blocking issue.
* Deformable/vision-guided navigation seem less tested cases for Isaac Sim/Lab. And after navigating through the docs/community a while, I found the documentation and helping are not as good as Mujoco. So there might be more unchartered territories where student helpers may feel lost about "how to do XX". See below for more addressing this point.

### HOW TO ###

Robotics projects usually take a more or less standard routine to build a simulation environment that can be happily used for RL training. This usually goes with loading models (meshes etc.) -> setting parameters (materials, dynamic, inital pose etc.) -> linking control to simulation -> investigate the robustness of env for RL training -> linking all debugging/visualization utils to the environment -> tuning reward to get the desired behaviours. In many cases, we want to programmatically control these steps while it is not often obvious or even possible due to the lack of documentation/support. For new and evolving software like Isaac Sim/Lab, many features could be pending for furture releases or buggy and very often workarounds are needed. This is not different for creating this prototype as well but I think we could more or less navigate it through after many trials and investigating community threads. I hence summarize following findings to address possible "How-to" questions.

__How to load mesh in common formats (.obj/.stl) through python__

The "standard way" is to convert obj/stl to usd format. However, UsdFileCfg does not offer the interface to specify PhysicsMaterial with Young's modulus and Poisson rate to spawn the deformables with these parameters. 
There is a [pull request](https://github.com/isaac-sim/IsaacLab/pull/2379) about this. For now, the prototype basically follows that and creates a utility (utils/isaac_mesh.py) to do so. Note for "smooth shading" we need to specify "subdivisionScheme" as "loop" or the other one. Use "bilinear" if facet mesh is expected for rendering.

__How to specify the simulation/collision geometries for deformables__

It seems that Isaac Sim allows to separate them by setting simulation resolution and collision simplification. For the former, Isaac Sim creates a coarser grid around the deformable to simulate and the DOFs are actually nodal position/velocity. This can be controlled by specifying "simulation_hexahedral_resolution" when spawning a deformable. The default is 10 and seems not fine enough for colon mesh. I am currently using 16 which leads to about 1300 nodes. 

Collision model is simplified by default ("collision_simplification=True"). There is a parameter can tune the level of simplification but I haven't figured out how this is controlled. For now the simplification is turned off so the colon model should be using the original mesh for collision detection and handling.

Note there are probably some [bugs](https://forums.developer.nvidia.com/t/deformable-object-not-connected-to-mesh/330880) about the visualized mesh and the one actually being simulated.

Turning on self-collision for one instance of environment seems fine. It is not as slow as previous experience with Bullet/SOFA. However, when multiple environments are spawned, errors are reported about overflow of some collision cache and asking for more allocation of memory. It is unclear how large vram/ram will be needed if we spawn a few thousands of environment like typical robot learning.

__How to create boundary conditions for colon nodes__

The standard way would be [creating "attachments"](https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/deformable-bodies.html) with some dummy objects. However, this seems not possible yet and might be related to inconsistency bug between Isaac Sim launched from standalone python or Ominiverse Launcher, according to [some discussion](https://forums.developer.nvidia.com/t/discrepancy-between-deformable-and-visual-meshes-in-python-standalone-application/320193).

Our current solution is to kinematically constrain nodal position to a target. We can use python to create such kinematic target but only after the physics data are filled after env.reset(). It is also worth to note that it seems connecting the node of the simulation mesh (resolution=16 for now) but not the original vertices on the mesh. There might be a relation can be exploited to figure out which nodes need to be attached but for now I am not sure about the order of the simulation hexahedras to create such a mapping. For now the solution is a very crude heuristic by picking the nodes closest to some points on the bounding box. This need to be investigated for finer automation of boundary condition creation.

__How to attach an FPV camera to robot tip__

It seems that you just need to spawn the camera with a prim path as the subelement of the robot link. Note that the camera parameters (pinhole model) look a bit weird to me. I tried some focus length and aperture params from google for endoscopy but that does not look very pleasant. So far it is a setup from some tuning and there is no realism alignment yet. Another critical issue is that currently FPV camera can sometimes "see-through" the shell when the robot is touching colon. Probably due to thin occupancy in the space so the camera ray does not detect the occlusion reliably? 

The clipping_range param is critical to have correct rendering behaviors in face of occlusion caused by small features. the default 0.01 will not work here as it may ignore the thickness of colon and see outside. Note if the refresh rate is too low, the rendered image might be blurred.

For parallel environments, it is possible to use [TiledCamera](https://isaac-sim.github.io/IsaacLab/v1.2.0/source/features/tiled_rendering.html) instead of Camera to gain more efficiency, e.g. 5/9 FPS for 32/64 envs on RTX3090 (simulation resolution is 16 and no self-collision turned on). 128 will raise out of CUDA memory error. Note that there are [consistency](https://github.com/isaac-sim/IsaacLab/issues/2631) and [degradation]((https://github.com/isaac-sim/IsaacLab/issues/1031)) issues between TiledCamera and Camera. 

__How about light__

Same as camera. But there are only native support light sources for shapes like sphere/disk/capsule. Not sure if they can cover endoscopy head light.

__How to drive robot__

For capsule endoscopy, the robot is basically modelled as a kinematic rigid body that can be driven by specifying body twist. There seem to be some issues about [attaching articulated bodies and deformables](https://forums.developer.nvidia.com/t/dynamically-create-attachments-at-runtime/322378). This might create complexities when we have a rope-like robot endoscopy. It is also not clear whether it will be stable to have a control interface (a rail translating/rotating the rope?) aligning with the real setup.

__How to sense contact__

For rigid body, it is possible to activate contact sensing and add a prim to read the data. However,
* it only gives normal force applied to the body and no contact location info.
* it seems to only work for [contacts between rigid bodies](https://github.com/isaac-sim/IsaacLab/issues/1599).

It looks like the contact sensing feature will be improved in [future release](https://github.com/isaac-sim/IsaacLab/issues/2446). Possible workarounds for detecting contacts between endoscopy and colon might need to resort to kinematic data. For instance, if the commanded motion is faithfully realised by capsule/deformation of a cable endoscope.


