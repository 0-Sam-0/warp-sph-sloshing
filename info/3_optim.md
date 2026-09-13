
# Examples/optim

## bounce - Differentiable Optimisation of Ballistic Trajectories

**What it simulates:**
Optimises the initial velocity of a particle through gradient descent so that it hits a target after bouncing off obstacles. An inverse control problem that learns the optimal ballistic parameters automatically.

**Warp features used:**
• `wp.Tape()` - Automatic differentiation system for end-to-end gradient computation
• `wp.sim.ModelBuilder()` - Physics scene construction with `requires_grad=True`
• `wp.sim.SemiImplicitIntegrator()` - Differentiable physics integrator 
• `.backward()` method - Automatic backpropagation through the physics simulation
• `wp.sim.collide()` - Differentiable contact system with restitution
• CUDA graph capture - Pre-compilation of the full forward+backward pass

**Key capability:**
Warp implements automatic differentiation through the entire physics simulation pipeline, allowing physical parameters to be optimised directly. The tape/gradient system is fully GPU-native and compatible with standard ML frameworks.

## cloth throw - Differentiable Optimisation of Cloth Dynamics

**What it simulates:**
Optimises the initial velocities of a piece of cloth through gradient descent so that its centre of mass hits a specific target. An inverse control problem for complex deformable dynamics with aerodynamics.

**Warp features used:**
• `wp.sim.ModelBuilder.add_cloth_grid()` - Automatic cloth mesh generation with elastic constraints
• Cloth physical parameters - `tri_ke` (elasticity), `tri_ka` (area), `tri_lift/drag` (aerodynamics)
• `wp.atomic_add()` - Parallel centre-of-mass computation through atomic reduction
• Differentiable integration - Forward/backward pass through the complete cloth dynamics
• `requires_grad=True` states - Simulation states with full gradient support

**Key capability:**
Warp handles differentiation through complex deformable physics automatically, aerodynamic forces and elastic constraints included. The system allows intelligent control of multi-particle soft-body objects with end-to-end optimisation.

## diffray - Differentiable Ray Tracing for Inverse Rendering

**What it simulates:**
A fully differentiable ray tracer that optimises scene parameters (mesh rotation, vertex positions, textures) to match target images. Implements Lambertian shading, texture mapping, anti-aliasing and normal computation for photorealistic rendering.

**Warp features used:**
• `wp.mesh_query_ray()` - Ray-mesh intersection with barycentric coordinates
• Differentiable texture interpolation - Bilinear sampling with automatic gradients
• `wp.atomic_add()` - Parallel per-vertex normal computation through atomic reduction
• `wp.optim.SGD` - Built-in optimiser for rendering parameters
• Anti-aliasing supersampling - Downsampling with a differentiable weighted average
• Multiple directional lights - Lighting system with optimisable intensities/directions

**Key capability:**
Warp implements an entire end-to-end differentiable rendering pipeline, enabling inverse rendering and 3D reconstruction from images. The system supports simultaneous optimisation of geometry, materials and lighting for advanced computer vision applications.

## drone - Model Predictive Control (MPC) for a Quadcopter Drone

**What it simulates:**
Implements Model Predictive Control for a quadcopter drone, optimising trajectories in real time to reach targets while avoiding obstacles. An advanced control system with Gaussian sampling and multi-objective optimisation.

**Warp features used:**
• `@wp.struct` - Custom data structures for propeller and drone parameters
• Parallel Gaussian sampling - Controlled noise generation for exploration
• Multiple cost functions - Weighted penalties for position, velocity, control and collisions
• `wp.optim.SGD` - Built-in optimiser for optimal MPC trajectories
• SDF collision detection - Collision detection against multiple geometric primitives
• Rollout simulations - Multiple parallel simulations for trajectory evaluation
• Control interpolation - Smooth linear interpolation between control waypoints

**Key capability:**
Warp handles complete robotic control systems with real-time differentiable optimisation. The framework integrates physics, optimisation and rendering for rapid development of advanced controllers for dynamic multi-body systems.

## inverse kinematics - Inverse Kinematics for Articulated Chains

**What it simulates:**
Solves inverse kinematics for an articulated robotic arm using gradient descent to place the end-effector on a target. Optimises joint angles to reach goal positions in Cartesian space.

**Warp features used:**
• `wp.sim.eval_fk()` - Differentiable forward kinematics for articulated chains
• `builder.add_joint_revolute()` - Creation of revolute joints with angular limits
• `wp.Tape()` - Automatic differentiation through forward kinematics
• `requires_grad=True` - Joint parameters optimised automatically
• Joint limits enforcement - Automatic constraints on the range of motion

**Key capability:**
Warp implements fully differentiable kinematics, turning complex IK problems into gradient-based optimisation. The system handles joint constraints and Jacobian computation automatically for precise robotic control.

## spring cage - Differentiable Optimisation of Spring-Mass Systems

**What it simulates:**
A particle connected by springs to the fixed points of a spatial "cage". Optimises the rest lengths of the springs to drive the particle towards a specific target position through gradient descent.

**Warp features used:**
• `builder.add_spring()` - Creation of elastic constraints with configurable stiffness/damping
• `requires_grad=True` model - Enables full automatic differentiation of the physics
• `model.spring_rest_length.grad` - Direct access to the gradients of physical parameters
• `wp.SemiImplicitIntegrator()` - Fully differentiable numerical integrator
• Multiple simulation states - Complete state history for backpropagation through time
• CUDA graph optimisation - Pre-compilation of the integrated forward+backward pass

**Key capability:**
Warp turns physical control problems into end-to-end differentiable optimisation. The framework computes gradients through complex dynamic simulations automatically, allowing intelligent tuning of physical parameters towards desired behaviours.

## trajectory - Differentiable Optimisation of Control Trajectories

**What it simulates:**
Optimises the sequence of torques/forces applied to a spherical rigid body so that it follows a circular reference trajectory. An optimal control problem that learns the inputs required for precise tracking automatically.

**Warp features used:**
• `warp.optim.Adam` - Built-in Adam optimiser for advanced differentiable optimisation
• `wp.array2d()` - Two-dimensional arrays storing states and target trajectories
• `wp.spatial_vector` - Spatial vectors applying forces/moments to rigid bodies
• Custom loss functions - L2 kernel computing the error between actual and reference trajectory
• `wp.SemiImplicitIntegrator()` - Fully differentiable physics integrator

**Key capability:**
Warp combines standard ML optimisers with differentiable physics for automatic optimal control. The system computes gradients through complete dynamic simulations, allowing control policies to be learned directly for complex tracking tasks.

## soft body properties - Differentiable Optimisation of FEM Soft-Body Materials

**What it simulates:**
Optimises the material parameters (Lamé parameters μ and λ) of a deformable body discretised with FEM tetrahedra, making it bounce off obstacles and reach a target. Inverse control of elastic properties towards desired dynamic behaviour.

**Warp features used:**
• `builder.add_soft_grid()` - Automatic generation of a tetrahedral FEM grid with material properties
• `wp.array2d()` for `tet_materials` - Differentiable per-tetrahedron material parameters
• `wp.optim.SGD` - Optimiser for physical parameters with constraint enforcement
• Differentiation through FEM - Automatic gradients through finite element dynamics
• Parameter constraints - Bounds enforcement on valid material parameters
• Differentiable collision handling - Soft-body contacts integrated into backpropagation

**Key capability:**
Warp enables inverse material design through fully differentiable FEM simulation, computing gradients across complex deformable dynamics. The system allows intelligent optimisation of material properties towards specific soft-body behaviours.

## fluid checkpoint - Fluid Optimisation with Gradient Checkpointing

**What it simulates:**
A fully differentiable 2D stable-fluids solver that optimises the initial velocity field so that the fluid forms the NVIDIA logo at the end. Implements manual gradient checkpointing to cut memory use dramatically during backpropagation over long simulations.

**Warp features used:**
• Segmented checkpointing - Memory management through selective forward recomputation during the backward pass
• Cyclic boundary conditions - Toroidal domain simulation with `cyclic_index()`
• Jacobi pressure solver - Multiple iterations for a differentiable incompressibility projection
• `wp.optim.Adam` - Advanced optimiser for high-resolution velocity fields (512x512)
• Semi-Lagrangian advection - Backward-Euler transport with differentiable bilinear interpolation
• Multiple CUDA graphs - Separate pre-compilation of forward/backward/zero for optimal performance

**Key capability:**
Warp implements memory-efficient strategies for long differentiable simulations, enabling inverse rendering optimisation over complex fluid dynamics. Checkpointing balances compute against memory for training that scales on the GPU.

---
