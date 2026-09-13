# Examples/fem

## diffusion 3d - Solving PDEs with Finite Element Methods

**What it simulates:**
Solves the 3D diffusion equation "νΔu = 1" with mixed boundary conditions (Neumann on the horizontal sides, Dirichlet on the others) using finite element methods over several mesh topologies.

**Warp features used:**
• `warp.fem` module - Complete finite element framework with automatic integration
• `@fem.integrand` - Decorators for weak mathematical forms (diffusion, boundary forms)
• Multi-mesh support - Grid3D, Tetmesh, Hexmesh, Nanogrid, Trimesh3D, Quadmesh3D
• `fem.integrate()` - Automatic assembly of matrices and right-hand-side vectors
• Boundary condition handling - Linear system projection and weak enforcement
• `fem_example_utils.bsr_cg()` - Optimised iterative Conjugate Gradient solver

**Key capability:**
Warp.fem provides a complete GPU-native FEM framework that handles complex spatial discretisations and matrix assembly automatically. It supports multiple mesh types and boundary conditions with performance that scales to industrial PDE problems.

## mixed elasticity - Nonlinear Elasticity with Mixed FEM Methods

**What it simulates:**
Solves the nonlinear Neo-Hookean elastic equilibrium equation "Div[d/dF Ψ(F(u))] = 0" using a mixed formulation with separate function spaces for displacement and stress. Applies Newton iterations to handle the nonlinearity of the constitutive model.

**Warp features used:**
• Mixed FEM spaces - Separate function spaces for displacement (Serendipity) and stress tensors
• `@fem.integrand` Neo-Hookean - Bilinear forms for stress/strain energy and Gauss-Newton approximation  
• Newton iterations - Automated loop with incremental matrix assembly for the nonlinearity
• Block diagonal operations - Tau mass matrix inversion and gradient/stress coupling
• Multiple element types - Triangular, quadrilateral with adaptive basis functions
• Area conservation tracking - Integrated monitoring of incompressibility constraints

**Key capability:**
Warp.fem handles advanced mixed formulations with multiple unknown fields and robust nonlinear solvers. Automatic assembly of the Newton-Raphson forms allows industrial hyperelastic simulations with precise control over material properties.

## apic fluid - Fluid Simulation with the Affine Particle-In-Cell Method

**What it simulates:**
APIC (Affine Particle-In-Cell) fluid simulation combining Lagrangian particles with an adaptive Eulerian NanoVDB grid. Includes gravity, incompressibility, SDF collisions and bidirectional particle-grid transfer for angular momentum conservation.

**Warp features used:**
• `wp.Volume.allocate_by_voxels()` - Automatic adaptive grid based on particle distribution
• `fem.PicQuadrature` - Specialised class for particle-in-cell quadrature with measures
• Mixed FEM spaces - Q1 velocity/fraction, P0 pressure for incompressible flow
• `fem.interpolate()` APIC advection - Velocity and gradient transfer grid→particles  
• Incompressibility solver - Divergence-free projection with the Schur complement method
• Collision SDF integration - Boundary conditions through signed distance functions

**Key capability:**
Warp unifies particle methods and FEM seamlessly, over an adaptive grid that resizes itself. The framework handles thousands of particles with accurate fluid physics while keeping real-time performance for production-quality simulations.

## streamlines - 3D Streamline Generation by Velocity Field Tracing

**What it simulates:**
Generates 3D streamlines by tracing through incompressible velocity fields with mixed boundary conditions (inflow, outflow, free-slip). Uses fixed-step forward tracing to visualise fluid flow patterns.

**Warp features used:**
• `fem.lookup()` - Operator for spatial lookup and field value interpolation over arbitrary domains
• `fem.Subdomain` - Definition of element subsets for differentiated boundary conditions
• Raviart-Thomas elements - RT1/P0 function spaces for incompressible velocity/pressure flow
• Streamline generation - `fem.interpolate()` with jittered spawn points and forward integration
• Multiple boundary types - Automatic side classification for inflow/outflow/free-slip conditions

**Key capability:**
Warp.fem integrates PDE solvers with advanced post-processing visualisation seamlessly, allowing efficient field tracing over complex geometries. The lookup operator preserves spatial accuracy even when advection leaves the original domain.

## distortion energy - Optimising the Parametrisation of a 3D Surface

**What it simulates:**
Minimises the (u,v) parametrisation distortion of 3D surfaces using the Symmetric Dirichlet energy "E(F) = ½|F|² + |F⁻¹|²", with Newton iterations for nonlinear optimisation of the deformation gradient F.

**Warp features used:**
• `@fem.integrand` energy forms - Symmetric Dirichlet gradient/Hessian for parametrisation optimisation
• `fem.grad()` - Deformation gradient computation and Gauss-Newton Hessian approximation  
• Newton iterations - Automated loop with implicit line search for energy minimisation
• `make_deformed_geometry()` - Construction of deformed geometries from displacement fields
• Multi-mesh support - Triangular, quad, deformed geometries with 2D/3D parametric spaces
• Checkerboard visualisation - Pattern rendering to validate parametrisation distortion

**Key capability:**
Warp.fem handles advanced energy-based optimisation for geometric parametrisation, essential to texture mapping and mesh processing. The framework automates complex derivatives while preserving the numerical accuracy required by industrial computer graphics.

## navier stokes - 2D Navier-Stokes Equations with Semi-Lagrangian Advection

**What it simulates:**
Solves the incompressible 2D Navier-Stokes equations "Du/dt - νΔ(u) + ∇p = 0, ∇·u = 0" with velocity-Dirichlet boundary conditions and a semi-Lagrangian advection scheme for numerical stability of the convective terms.

**Warp features used:**
• Mixed FEM Q(d)-Q(d-1) - Stable velocity/pressure function spaces for saddle-point systems
• `fem.lookup()` semi-Lagrangian - Backtracking convection with accuracy-preserving interpolation
• `fem.ImplicitField` - Boundary conditions defined through implicit functions
• `SaddleSystem` solver - Automatic assembly and solution of coupled incompressible systems
• Hard boundary projection - Dirichlet condition enforcement through a constraint matrix

**Key capability:**
Warp.fem provides a complete CFD framework with advanced methods for incompressible flows, including semi-Lagrangian advection that removes the CFL restriction. Automatic handling of saddle-point systems keeps fluid simulations stable even at high Reynolds numbers.

## burgers - Inviscid Burgers PDE with Discontinuous Galerkin

**What it simulates:**
Solves the non-conservative inviscid Burgers PDE "∂u/∂t + (u·∇)u = 0" using a Discontinuous Galerkin method with a minmod slope limiter to handle discontinuous solutions and shock formation.

**Warp features used:**
• Discontinuous Galerkin spaces - Discontinuous function spaces for capturing shock waves
• `fem.jump()` / `fem.average()` - Operators for jump/average conditions at element interfaces
• Upwind transport form - Upwind scheme with numerical fluxes for hyperbolic stability  
• Minmod slope limiter - Slope limiter preventing spurious oscillations near discontinuities
• SSPRK3 integration - Strong Stability Preserving Runge-Kutta 3rd order for conservation
• `fem.lookup()` neighbour access - Access to adjacent cells for slope limiting calculations

**Key capability:**
Warp.fem fully supports DG methods for nonlinear hyperbolic equations, automatic shock-capturing limiters included. The framework preserves the essential conservation properties while staying stable even for solutions with extreme gradients and discontinuities.

## magnetostatics - 3D Magnetostatics with Nédélec H(curl) Elements

**What it simulates:**
Solves a 3D magnetostatic problem (a copper coil carrying radial current around a cylindrical iron core) using a curl-curl formulation and H(curl)-conforming function spaces for the static Maxwell equations "1/μ∇×B + j = 0".

**Warp features used:**
• `fem.ElementBasis.NEDELEC_FIRST_KIND` - Nédélec elements for electromagnetic H(curl) conformity
• `fem.ImplicitField` with geometry deformation - Cube→cylinder mapping with analytic gradient
• Multi-material domains - Differentiated permeability fields μ (iron, copper, vacuum) through implicit functions
• `curl_curl_form` - Curl-curl variational formulation for electromagnetic vector fields
• `fem.curl()` operator - Built-in curl operator for post-processing the magnetic field B
• Deformed geometry construction - Complex geometries from implicit field mappings

**Key capability:**
Warp.fem fully supports industrial electromagnetic problems with arbitrary geometries and complex material properties. Nédélec elements guarantee the tangential continuity that vector fields require, which is essential to accuracy in Maxwell simulations.

## adaptive grid - Adaptive Grids for CFD with Complex Colliders

**What it simulates:**
Incompressible flow simulation around complex geometries using adaptive grids that raise resolution automatically near collider boundaries. Handles t-junctions and discontinuities between multiple refinement levels.

**Warp features used:**
• `fem.adaptivity.adaptive_nanogrid_from_field()` - Automatic generation of multi-level grids from refinement fields
• `fem.ImplicitField` refinement function - Refinement zone control through SDF distance-based criteria  
• H(div)-conforming Raviart-Thomas - Elements that conserve mass locally for CFD accuracy
• T-junction handling - Automatic handling of discontinuities at boundaries between resolution levels
• `side_divergence_form` - Flux corrections for non-conforming resolution transitions
• NanoVDB collider integration - Complex collision geometries from industrial pipelines

**Key capability:**
Warp.fem fully automates adaptive mesh refinement while preserving conservation properties and numerical accuracy. Adaptive grids make CFD simulations scalable at an optimal computational cost, concentrating resolution only where it matters.

## nonconforming contact - Elastic Contact Between Non-Conforming Bodies

**What it simulates:**
Solves a no-slip contact problem between two elastic bodies discretised separately, with "Div[E:D(u)] = g". Uses a staggered iterative scheme in which the bodies influence each other through displacement Dirichlet BCs and applied boundary stress.

**Warp features used:**
• `fem.field.NonconformingField` - Field coupling between independently discretised geometries
• `fem.SymmetricTensorMapper(wp.mat22)` - Optimised storage of symmetric tensors (3 DOF instead of a full 4)
• Mixed displacement-stress formulation - Serendipity S_k spaces for displacement, Q_{k-1}d for stress
• Staggered iterative coupling - Bodies solved alternately, coupled through boundary conditions
• Damped stress updates - Numerical stability control through relaxation parameters

**Key capability:**
Warp.fem natively handles multi-body contact problems with non-conforming discretisations, essential to industrial mechanical simulation. The framework automates information transfer between separate geometries while preserving the physical accuracy of the contact interface.

## darcy level-set optimization - Topology Optimisation of 2D Darcy Flow

**What it simulates:**
Level-set based shape optimisation maximising Darcy flow through a square 2D domain. Evolves an implicit shape representation through the adjoint gradient to optimise the permeability of material regions under a constant volume constraint.

**Warp features used:**
• `wp.Tape()` automatic differentiation - Adjoint gradient computation through complex PDE solves
• Level set method with sigmoid smoothing - Differentiable implicit representation of material interfaces
• `fem.ImplicitField` for material properties - Permeability fields as a smooth function of the level set
• Semi-Lagrangian/DG advection - Multiple schemes for level set evolution (continuous/discontinuous)
• Implicit Function Theorem - Differentiation through iterative linear system solves
• Volume constraint penalty - Constraint enforcement through a weighted loss combination
• Forward/backward optimisation loop - Full management of the topology optimisation cycle

**Key capability:**
Warp.fem unifies PDE solving with automatic differentiation seamlessly for industrial topology optimisation problems. The framework handles adjoint computation through implicit solvers automatically, allowing shape optimisation that scales to realistic engineering applications.

## elastic shape optimization - Shape Optimisation of a 2D Elastic Beam

**What it simulates:**
Shape optimisation of a 2D cantilever beam minimising the squared norm of the stress field (compliance) through finite element analysis and gradient-based optimisation. The beam is fixed on the left with a constant load on the right, under volume constraints and boundary conditions.

**Warp features used:**
• `wp.Tape()` + Implicit Function Theorem - Automatic differentiation through elastic linear system solves
• `warp.optim.Adam` - Built-in gradient-based optimiser for nodal position updates
• Hooke elasticity `@fem.integrand` - Stress-strain formulation with Lamé parameters for realistic materials
• Deformed geometry construction - Support for triangular/quad/grid meshes with vertex position optimisation
• Quality regularisation terms - Penalties on degenerate/inverted elements for mesh stability without remeshing
• Fixed vertex projectors - Boundary constraint enforcement during shape optimisation

**Key capability:**
Warp.fem fully automates structural shape optimisation with automatic differentiation through complex PDE solvers. Native integration with scalable optimisers allows industrial design optimisation while preserving physics accuracy and mesh quality.
