# Examples/sim

## cartpole - Articulated Robot Simulation from URDF

**What it simulates:**
Multi-instance dynamic simulation of articulated robotic systems (cartpole) loaded from standard URDF files. Includes forward kinematics, joint constraints and semi-implicit integration for numerical stability.

**Warp features used:**
• `wp.sim.ModelBuilder()` - Physics model construction with joints and constraints
• `wp.sim.parse_urdf()` - Built-in parser for standard robotic URDF files
• `wp.sim.SemiImplicitIntegrator()` - Stable numerical integrator for rigid bodies
• `wp.sim.eval_fk()` - Automatic forward kinematics for articulated chains
• `wp.sim.render.SimRenderer()` - Rendering specialised for robotics visualisation
• Multi-environment support - Parallel simulation of multiple instances for RL/testing

**Key capability:**
Warp provides a complete robotics pipeline: industrial URDF loading, accurate physics simulation with constraints, and integrated 3D rendering. It supports massive parallelisation for reinforcement learning training over hundreds of simultaneous environments.

## cloth - Cloth Simulation with FEM and Collisions

**What it simulates:**
Deformable cloth simulation using FEM (Finite Element Method) models with three different numerical integrators. The cloth collides with complex rigid bodies and includes constraints, point masses and elastic forces.

**Warp features used:**
• `wp.sim.ModelBuilder.add_cloth_grid()` - Automatic generation of FEM cloth grids
• Multiple integrators - `SemiImplicitIntegrator`, `XPBDIntegrator`, `VBDIntegrator` for stability
• `wp.sim.collide()` - Optimised soft-body vs rigid-body collision system
• `wp.sim.Mesh()` - USD mesh integration for complex collision geometries
• Configurable physical parameters - elasticity, damping, friction for material realism
• State swapping - Automatic double buffering for temporal performance

**Key capability:**
Warp offers production-ready cloth simulation with multiple numerical integrators for the stability/performance trade-off. The system automatically handles complex topologies, self-collisions and rigid body interactions using GPU-optimised algorithms.

## granular - Particle-Based Granular Material Simulation

**What it simulates:**
Granular material simulation using a particle-based approach with local interactions. Particles behave like sand or grain, with collisions, friction and realistic bulk material properties.

**Warp features used:**
• `wp.sim.ModelBuilder.add_particle_grid()` - Automatic generation of 3D particle grids
• `self.model.particle_grid.build()` - Spatial hash construction for efficient neighbour finding
• `wp.sim.SemiImplicitIntegrator()` - Stable time integration for particle systems
• Configurable physical parameters - `particle_kf`, `soft_contact_kd` for granular behaviour control
• CUDA graph capture - Simulation loop pre-compilation for optimal performance

**Key capability:**
Warp handles interactions between thousands of particles automatically through GPU-optimised spatial hashing. The framework allows real-time simulation of complex granular phenomena (pile formation, flow dynamics) with precise control over the material's physical properties.

## granular collision sdf - Particle-Based Granular Material Simulation

**What it simulates:**
Granular material simulation using a particle-based approach with local interactions. Particles behave like sand or grain, with collisions, friction and realistic bulk material properties.

**Warp features used:**
• `wp.sim.ModelBuilder.add_particle_grid()` - Automatic generation of 3D particle grids
• `self.model.particle_grid.build()` - Spatial hash construction for efficient neighbour finding
• `wp.sim.SemiImplicitIntegrator()` - Stable time integration for particle systems
• Configurable physical parameters - `particle_kf`, `soft_contact_kd` for granular behaviour control
• CUDA graph capture - Simulation loop pre-compilation for optimal performance

**Key capability:**
Warp handles interactions between thousands of particles automatically through GPU-optimised spatial hashing. The framework allows real-time simulation of complex granular phenomena (pile formation, flow dynamics) with precise control over the material's physical properties.

## jacobian ik - Inverse Kinematics with Automatic Differentiation

**What it simulates:**
Solves robotic inverse kinematics problems using the transposed Jacobian method. Computes Jacobian matrices automatically to drive the end-effector of articulated systems towards multiple target positions in parallel.

**Warp features used:**
• `wp.Tape()` - Automatic differentiation system for symbolic gradient computation
• `requires_grad=True` - Gradient tracking enabled on arrays and physics models
• `wp.sim.eval_fk()` - Differentiable forward kinematics for articulated chains
• `tape.backward()` / `tape.gradients` - Automatic backpropagation for partial derivatives
• Multi-environment parallelisation - Simultaneous Jacobian computation over multiple instances
• Gradient-based optimisation loop - Parameter updates via the Jacobian transpose method

**Key capability:**
Warp integrates automatic differentiation natively into physics simulation, allowing real-time robotic optimisation and control. Jacobians are computed automatically on the GPU without numerical approximations, enabling advanced techniques such as reinforcement learning and optimal control.

## quadruped - Multi-Environment Quadruped Robot Simulation

**What it simulates:**
Multi-instance dynamic simulation of articulated quadruped robots with joint position control. Includes floating-base kinematics, built-in PID control and automatic environment distribution for parallel training.

**Warp features used:**
• `wp.sim.FeatherstoneIntegrator()` - Integrator specialised for articulated robot dynamics
• `compute_env_offsets()` - Utility for automatic multi-dimensional environment distribution
• `wp.sim.JOINT_MODE_TARGET_POSITION` - Built-in PID position control for joints
• `floating=True` URDF parsing - Floating base support for robotic locomotion
• Memory pooling + CUDA graphs - Advanced optimisations for massive RL training
• Multi-integrator support - XPBDIntegrator, SemiImplicitIntegrator, FeatherstoneIntegrator

**Key capability:**
Warp offers production-ready robot simulation with numerical integrators specialised for articulations. The framework handles hundreds of parallel environments automatically for reinforcement learning, preserving physical accuracy and optimal real-time performance.

## rigid chain - Multi-Joint Articulated Chains

**What it simulates:**
Simulation of chains of rigid bodies connected by different types of mechanical joint. Demonstrates the dynamic behaviour of articulated systems with revolute, ball, universal, fixed and compound constraints.

**Warp features used:**
• `wp.sim.ModelBuilder.add_articulation()` - Construction of multi-body articulated systems
• Multiple joint types - `JOINT_REVOLUTE`, `JOINT_BALL`, `JOINT_UNIVERSAL`, `JOINT_COMPOUND`, `JOINT_FIXED`
• `wp.sim.JointAxis()` - Rotation axis definition with configurable angular limits
• `wp.sim.FeatherstoneIntegrator()` - Featherstone algorithm for efficient articulated dynamics
• Joint limits/constraints - Automatic enforcement of mechanical limits with stiffness/damping
• Body/shape composition - Creation of complex geometries with inertial properties

**Key capability:**
Warp provides a complete library of mechanical joints for robotic and engineering simulation. The system automatically handles complex constraints, joint limits and multi-body dynamics using state-of-the-art numerical algorithms for real-time performance.

## rigid contact - Multi-Shape Rigid Body Collisions

**What it simulates:**
Free-fall and collision simulation of rigid bodies with different geometries (boxes, spheres, capsules, complex meshes). Includes initial rotations and dynamic interactions with the ground and between objects.

**Warp features used:**
• Multiple shape primitives - `add_shape_box()`, `add_shape_sphere()`, `add_shape_capsule()`, `add_shape_mesh()`
• `wp.sim.collide()` - Automatic rigid-body collision system with broad/narrow phase
• USD mesh integration - Loading of arbitrary geometries through `UsdGeom.Mesh()`
• `wp.sim.SemiImplicitIntegrator()` - Stable time integration for impulsive dynamics
• Configurable physical parameters - stiffness (ke), damping (kd), friction (kf) for material realism

**Key capability:**
Warp unifies simple geometric primitives and complex meshes in a single GPU-optimised collision system. It handles broad-phase detection and contact resolution automatically for multi-object simulations that scale to hundreds of simultaneous bodies.

## rigid force - Applying External Forces to Rigid Bodies

**What it simulates:**
Rigid body simulation with direct application of external forces/torques to produce controlled motion. Demonstrates precise dynamic control through programmatic forces on a multi-body system.

**Warp features used:**
• `wp.sim.XPBDIntegrator()` - XPBD integrator for stable dynamics under impulsive forces
• `state.body_f.assign()` - Direct application of external forces/torques to specific bodies
• `SimRendererOpenGL()` - Interactive real-time rendering for dynamic visualisation
• `wp.sim.collide()` - Collision system integrated with ground and objects
• Force/torque control loop - Continuous force control throughout the simulation

**Key capability:**
Warp allows direct force control on every rigid body, enabling custom actuators, motors and robotic controllers. The system automatically integrates external forces with physical constraints and collisions for realistic behaviour.

## rigid gyroscopic - The Dzhanibekov Effect and Rotational Instability

**What it simulates:**
Demonstrates the Dzhanibekov effect, where rigid bodies with an asymmetric mass distribution develop chaotic tumbling in free space because of unstable rotation axes. Simulates complex gyroscopic dynamics without gravity.

**Warp features used:**
• Multi-shape body composition - Box shapes combined with different densities/positions for asymmetric inertia
• `wp.sim.SemiImplicitIntegrator()` - Stable integration for complex rotational dynamics  
• `builder.body_qd` - Angular velocity initialisation with precise perturbations
• `builder.gravity = 0.0` - Gravity disabled to simulate free space
• `self.model.ground = False` - Ground constraints removed for completely free motion

**Key capability:**
Warp captures subtle physical effects such as gyroscopic instability and precession accurately, demonstrating the numerical precision needed for complex dynamic phenomena. The system handles multi-shape inertia tensors automatically for realistic simulation of advanced rotational mechanics.

## rigid soft contact - Rigid-Soft Body Interaction with FEM

**What it simulates:**
Hybrid collision simulation between rigid bodies (a sphere) and deformable FEM structures (an elastic beam). The sphere falls and interacts with a finite element grid that deforms elastically according to configurable material parameters.

**Warp features used:**
• `builder.add_soft_grid()` - Automatic generation of 3D tetrahedral FEM grids
• Lamé elastic parameters - `k_mu`, `k_lambda` for controlling material deformation
• Hybrid rigid-soft collision - Unified collision system for mixed interactions
• `soft_contact_*` parameters - Configurable soft-body contact stiffness/damping/friction
• `wp.sim.SemiImplicitIntegrator()` - Stable time integration for hybrid systems

**Key capability:**
Warp unifies rigid and soft-body simulation in one coherent framework, handling the complex interactions between rigid and deformable materials automatically. The native FEM system supports realistic physical parameters for accurate engineering simulation.

## soft body - Hyperelastic FEM Materials under Controlled Twist

**What it simulates:**
Simulation of Neo-Hookean hyperelastic materials using tetrahedral FEM under a controlled 180° twist. Includes volume conservation measurement during extreme deformation and boundary condition control for numerical validation.

**Warp features used:**
• `builder.add_soft_grid()` with Lamé parameters - Precise Neo-Hookean hyperelastic modelling
• Custom `twist_points()` kernel - Programmable boundary condition control for specific tests
• `fix_top`/`fix_bottom` constraints - Selective constraints for deformation control
• Volume computation kernel - Volume conservation measured through tetrahedral integration
• `wp.transform` applications - Controlled rotations/translations applied to particle subsets

**Key capability:**
Warp supports advanced constitutive models for hyperelastic materials with precise control over boundary conditions. The framework allows numerical validation through controlled deformation tests while preserving the physical accuracy engineering simulation requires.

## cloth self contact - FEM Cloth with VBD Self-Contact

**What it simulates:**
FEM cloth simulation under controlled twist, demonstrating the VBD integrator's ability to handle complex self-collisions while keeping the cloth entirely intersection-free through extreme deformation.

**Warp features used:**
• `wp.sim.VBDIntegrator(handle_self_contact=True)` - Integrator specialised for advanced self-contact
• `builder.add_cloth_mesh()` - Cloth created from a USD mesh with configurable physical parameters
• Custom rotation kernels - `initialize_rotation()`, `apply_rotation()` for boundary condition control
• `PARTICLE_FLAG_ACTIVE` manipulation - Particle flag control for selective constraints
• `integrator.rebuild_bvh()` - Periodic BVH rebuilding to maintain query efficiency
• Self-contact parameters - `soft_contact_radius`, `soft_contact_margin` for precise detection

**Key capability:**
Warp's VBDIntegrator resolves topologically complex self-contacts automatically without intersection artifacts, which is crucial for cloth under extreme deformation. The system manages BVH rebuilding intelligently to keep real-time performance even with highly deformed geometry.

---
