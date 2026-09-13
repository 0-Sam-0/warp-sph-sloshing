# Examples/core

## dem - Cohesive Particle Simulation

**What it simulates:**
DEM (Discrete Element Method) simulation of particles interacting through contact, friction and cohesion forces, falling under gravity and sticking to the ground.

**Warp features used:**
• `@wp.kernel`/`@wp.func` - Decorators for parallelised GPU functions
• `wp.HashGrid` - Spatial data structure for fast neighbour search
• `wp.hash_grid_query()` - Efficient spatial query for adjacent particles  
• `wp.array()` - GPU arrays for positions, velocities and forces
• `wp.ScopedCapture`/CUDA graphs - Performance optimisation through pre-compilation
• `wp.render.UsdRenderer` - Direct rendering to USD format

**Key capability:**
Warp handles CPU-GPU data transfer automatically and optimises execution through CUDA graphs. Spatial hashing makes simulations with thousands of particles scalable while keeping real-time performance.

## fluid - 2D Fluid Simulation

**What it simulates:**
Implements a 2D "Stable Fluids" solver for computational fluid dynamics, with semi-Lagrangian advection, pressure projection and gravity forces on a regular grid.

**Warp features used:**
• `wp.array2d()` - Two-dimensional arrays for grid-based simulation
• `wp.constant()` - Compile-time constants for grid dimensions
• `@wp.func` - Helper functions for bilinear interpolation and sampling
• Multiple `@wp.kernel` - Specialised kernels (advection, divergence, pressure_solve)
• `wp.ScopedCapture` - Pressure iteration optimisation through CUDA graphs

**Key capability:**
Warp handles complex 2D grid operations automatically, with high GPU efficiency and smooth interpolation. The iterative loop of the pressure solver is pre-compiled for maximum real-time performance.

## graph capture - Procedural Noise Generation

**What it simulates:**
Generates animated procedural noise using fractional Brownian motion (FBM) across multiple octaves and frequencies. The coordinate grid scrolls over time to produce animated effects.

**Warp features used:**
• `wp.ScopedCapture()` - Captures sequences of kernel launches into CUDA graphs
• `wp.capture_launch()` - Optimised execution of pre-recorded graphs  
• `wp.noise()` - GPU-native procedural noise generation
• `wp.rand_init()` - Random state initialisation for deterministic seeding
• Multiple kernel launches - Loop over 16 octaves with varying frequencies/amplitudes
• `wp.ScopedTimer()` - Automatic performance profiling

**Key capability:**
CUDA graph capture removes the overhead of repeated launches, dramatically speeding up sequences of identical kernels. Warp provides GPU-optimised noise/random primitives, ideal for real-time procedural generation.

## marching cubes - Surface Extraction from SDF Fields

**What it simulates:**
Uses the marching cubes algorithm to extract iso-level surfaces from SDF (Signed Distance Fields) density fields. Generates animated geometric primitives (box, torus) combined through smooth blending operations.

**Warp features used:**
• `wp.MarchingCubes()` - Built-in class for automatic mesh surface extraction
• `wp.array3d()` - Three-dimensional arrays representing volumetric fields  
• SDF `@wp.func` functions - Geometric primitives (box, torus) and spatial transforms
• `wp.quat_*()` operations - Quaternion system for smooth animated rotations
• `.surface()` method - Triangle mesh extraction from an iso-surface threshold

**Key capability:**
Warp natively integrates GPU-optimised marching cubes, converting volumetric fields into renderable meshes automatically. Procedural SDF functions allow complex animated geometries with precise mathematical blending.

## mesh - PBD Simulation with Deforming Mesh Collisions

**What it simulates:**
PBD (Position Based Dynamics) particle simulation colliding against a deforming triangle mesh. Particles fall under gravity and collide with a 3D model (bunny) that deforms dynamically through sinusoidal animation.

**Warp features used:**
• `wp.Mesh()` - Class for triangle meshes with built-in BVH (Bounding Volume Hierarchy)
• `wp.mesh_query_point_sign_normal()` - Optimised point-to-mesh spatial collision queries
• `wp.mesh_eval_position()` - Precise position computation on the triangle surface
• `.refit()` method - Dynamic BVH update after mesh deformation
• USD integration - Asset loading through the Pixar USD library

**Key capability:**
Warp automatically manages BVH spatial data structures that update efficiently during mesh deformation. Collision queries exploit hardware acceleration for real-time performance even with thousands of particles.

## nvdb - Particle Simulation with Volumetric SDF Fields

**What it simulates:**
PBD particle simulation colliding against SDF (Signed Distance Field) fields in NanoVDB format. Particles fall under gravity and collide with complex geometries represented as discretised volumes.

**Warp features used:**
• `wp.Volume.load_from_nvdb()` - Direct loading of NanoVDB files from Houdini/Blender
• `wp.volume_sample_f()` - Trilinear sampling of SDF values from a volumetric grid
• `wp.volume_sample_grad_f()` - Simultaneous sampling of SDF value and gradient
• `wp.volume_world_to_index()` - Automatic world-to-grid coordinate conversion
• Custom `volume_grad()` - Gradient computation by finite differences for surface normals

**Key capability:**
Warp natively supports the industry-standard NanoVDB format, allowing precise collisions with arbitrarily complex geometries. Hardware-accelerated sampling keeps real-time performance even with tens of thousands of particles on high-resolution volumes.

## raycast - Ray Tracer with Triangle Meshes

**What it simulates:**
Implements a basic ray tracer that casts rays from the camera through every pixel to compute intersections with triangle meshes. Renders using surface normals as colour values for 3D visualisation.

**Warp features used:**
• `wp.mesh_query_ray()` - Hardware-accelerated ray-triangle intersection query  
• `wp.Mesh()` - Mesh data structure with automatic BVH for spatial queries
• USD integration - Direct asset loading from the Pixar USD pipeline
• Pixel parallelisation - One kernel thread per pixel with automatic coordinates
• `query.normal` - Direct access to the geometric data (normals, positions) of the hit

**Key capability:**
Warp automatically turns triangle meshes into BVH structures optimised for GPU ray casting. Massive parallelisation allows real-time rendering of complex scenes with no manual thread management or spatial optimisation.

## raymarch - SDF Renderer with Ray Marching

**What it simulates:**
Implements a ray marching renderer that uses Signed Distance Functions (SDF) to define procedural geometries. Builds scenes from primitives (spheres, boxes, planes) combined through boolean operations, with advanced lighting including diffuse, specular, fresnel and soft shadows.

**Warp features used:**
• `@wp.func` SDF primitives - Mathematical functions generating procedural geometry (sphere, box, plane)
• Boolean SDF operations - Union, subtract, intersect for combining complex primitives
• Adaptive ray marching - Loop with dynamic step size based on the SDF distance
• Gradient-based normals - Normal computation by finite differences of the SDF gradient
• Advanced lighting model - Diffuse, specular, fresnel reflection and soft shadow calculation
• Gamma correction - Colour post-processing for realistic output

**Key capability:**
Warp allows complex procedural rendering entirely on the GPU with no explicit geometry. Mathematical ray marching generates infinite detail with physically-based lighting, ideal for procedural scenes and artistic effects.

## sample mesh - Uniform Sampling of Mesh Surfaces

**What it simulates:**
Samples uniformly distributed points on the surface of a triangle mesh using a Cumulative Distribution Function (CDF). Computes triangle areas to build a proportional probability distribution, generating random points that respect the geometric density of the surface.

**Warp features used:**
• `wp.mesh_eval_position()` - Surface position evaluation through barycentric coordinates
• `wp.atomic_add()` - Atomic GPU operations for thread-safe parallel sums
• `wp.lower_bound()` - Optimised binary search for CDF sampling
• `wp.randf()` - GPU random number generation with deterministic per-frame seeding
• Barycentric coordinate sampling - Uniform point generation inside triangles

**Key capability:**
Warp natively implements every primitive needed for advanced geometric sampling, including GPU-optimised binary searches and barycentric coordinates. The system handles massive parallelisation while keeping the statistical distribution mathematically correct for Monte Carlo applications.

## sph - Fluid Simulation with Smoothed Particle Hydrodynamics

**What it simulates:**
SPH fluid simulation using mathematical kernels to compute density, pressure forces and viscosity between particles. Implements a kick-drift scheme for time integration with gravity, bounds collision and damping for realistic fluid behaviour.

**Warp features used:**
• `wp.HashGrid` with `wp.hash_grid_query()` - Spatial neighbour search for SPH interactions
• `wp.hash_grid_point_id()` - Thread ordering by cell for optimised memory access
• Custom SPH kernels `@wp.func` - Mathematical density, pressure and viscous kernels
• Kick-drift integration - Numerical scheme separating velocity and position updates
• Multi-step simulation - Multiple substeps for numerical stability with small timesteps

**Key capability:**
Warp allows the complex mathematical SPH equations to be implemented directly on the GPU while keeping real-time performance. Automatic spatial hashing handles thousands of particle-particle interactions efficiently for convincing fluids.

## torch - Differentiable Optimisation with PyTorch Integration

**What it simulates:**
Optimises the non-convex Rosenbrock function using PyTorch's Adam optimiser over distributed particles. Demonstrates the seamless integration between Warp and PyTorch for automatic differentiation and machine learning.

**Warp features used:**
• `torch.autograd.Function` - Custom class integrating Warp kernels into the PyTorch computational graph
• `wp.from_torch()` / `wp.to_torch()` - Automatic conversion between PyTorch tensors and Warp arrays
• `adjoint=True` - Automatic differentiation backward pass for gradient computation
• `wp.device_to_torch()` - Consistent GPU device handling across frameworks
• Rosenbrock `@wp.func` - Complex mathematical function evaluated in parallel over thousands of points

**Key capability:**
Warp integrates natively with PyTorch while keeping optimal GPU performance and supporting automatic differentiation. It allows custom high-performance computation to be embedded directly into neural network training pipelines without significant overhead.

## wave - 2D Wave Equation Simulation

**What it simulates:**
Solves the 2D wave equation with finite differences on a regular grid, simulating wave propagation with collisions against a moving sphere. Integrates in time using a multiple-substep scheme for numerical stability.

**Warp features used:**
• Finite difference `laplacian()` - Nabla-squared operator for the differential wave equation
• `wave_solve` kernel - Explicit time integrator for wave propagation
• `wave_displace` - Sinusoidal forcing term generating waves from the moving sphere
• 2D grid indexing - Automatic handling of (x,y) coordinates → linear array indexing
• `grid_update` - Height field → mesh vertices conversion for dynamic rendering

**Key capability:**
Warp allows complete PDE (Partial Differential Equations) simulations on the GPU with automatic substeps for an optimal accuracy/performance trade-off. Real-time mesh vertex updates produce fluid visualisations integrated with the physics.

---
