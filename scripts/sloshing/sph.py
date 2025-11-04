# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Smoothed Particle Hydrodynamics
#
# Shows how to implement a SPH fluid simulation.
#
# Neighbors are found using the wp.HashGrid class, and
# wp.hash_grid_query(), wp.hash_grid_query_next() kernel methods.
#
# Reference Publication
# Matthias Müller, David Charypar, and Markus H. Gross.
# "Particle-based fluid simulation for interactive applications."
# Symposium on Computer animation. Vol. 2. 2003.
#
###########################################################################
#
# !! Branched on 8 October
# !! Revised version by Samuel Seligardi
# 
###########################################################################

import numpy as np
import warp as wp
import warp.render

# @wp.func decorator indicates a "device" function that will run on GPU instead of "host" (CPU)
@wp.func
def square(x: float):
    return x * x

@wp.func
def cube(x: float):
    return x * x * x

# This function is never used
@wp.func                        #
def fifth(x: float):            #
    return x * x * x * x * x    #

@wp.func
def density_kernel(xyz: wp.vec3, h: float):
    """
    Compute the density contribution kernel for SPH (Smoothed Particle Hydrodynamics).

    This kernel calculates a smoothed density value based on the distance from a point
    to a particle. The kernel returns zero beyond the smoothing length and follows a
    cubic polynomial within the support radius.

    Parameters
    ----------
    xyz : wp.vec3
        The relative position (difference) vector between two particles.
    h : float
        The smoothing length (support radius) of the kernel.
        Particles beyond this distance do not contribute to the density calculation.

    Returns
    -------
    float
        The kernel value. Returns a non-negative value proportional to the cube of
        (h² - distance_squared) if the distance is within the smoothing length,
        otherwise returns 0.0.
    """
    # calculate distance
    distance_squared = wp.dot(xyz, xyz)

    return wp.max(cube(square(h) - distance_squared), 0.0)

@wp.func
def diff_pressure_kernel(
    xyz: wp.vec3, pressure: float, neighbor_pressure: float, neighbor_rho: float, h: float
):
    """
    Calculate the pressure gradient contribution from a neighboring particle using SPH kernel.

    This function computes the differential pressure kernel term used in Smoothed Particle
    Hydrodynamics (SPH) simulations for calculating pressure forces between particles.

    Parameters
    ----------
    xyz : wp.vec3
        Position vector from current particle to neighbor particle.
    pressure : float
        Pressure value at the current particle.
    neighbor_pressure : float
        Pressure value at the neighboring particle.
    neighbor_rho : float
        Density value at the neighboring particle.
    h : float
        Kernel smoothing length (radius of influence).

    Returns
    -------
    wp.vec3
        Pressure gradient contribution vector. Returns zero vector if neighbor is outside
        the smoothing length, otherwise returns the weighted pressure gradient term.
    """
    
    # calculate distance
    distance = wp.sqrt(wp.dot(xyz, xyz))

    if distance < h:
        # calculate terms of kernel
        term_1 = -xyz / distance
        term_2 = (neighbor_pressure + pressure) / (2.0 * neighbor_rho)
        term_3 = square(h - distance)
        return term_1 * term_2 * term_3
    else:
        return wp.vec3()

@wp.func
def diff_viscous_kernel(xyz: wp.vec3, v: wp.vec3, neighbor_v: wp.vec3, neighbor_rho: float, h: float):
    """
    Calculate the viscous force contribution from a neighboring particle using SPH kernel.

    This function computes the differential viscous kernel term used in Smoothed Particle
    Hydrodynamics (SPH) simulations for calculating viscous forces between particles.

    Parameters
    ----------
    xyz : wp.vec3
        Position vector from current particle to neighbor particle.
    v : wp.vec3
        Velocity vector of the current particle.
    neighbor_v : wp.vec3
        Velocity vector of the neighboring particle.
    neighbor_rho : float
        Density value at the neighboring particle.
    smoothing_length : float
        Kernel smoothing length (radius of influence).

    Returns
    -------
    wp.vec3
        Viscous force contribution vector. Returns zero vector if neighbor is outside
        the smoothing length, otherwise returns the weighted velocity difference term
        scaled by the distance factor (smoothing_length - distance).
    """
    
    # calculate distance
    distance = wp.sqrt(wp.dot(xyz, xyz))

    # calculate terms of kernel
    if distance < h:
        term_1 = (neighbor_v - v) / neighbor_rho
        term_2 = h - distance
        return term_1 * term_2
    else:
        return wp.vec3()

@wp.kernel
def compute_density(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    density_normalization: float,
    h: float,
):
    """
    Compute the density at each particle position using SPH density summation.

    This kernel function calculates the density field for each particle by summing
    contributions from neighboring particles within the smoothing length using a
    density kernel. The computation is accelerated using a spatial hash grid for
    efficient neighbor search.

    Parameters
    ----------
    grid : wp.uint64
        Hash grid structure used for efficient spatial neighbor queries.
    particle_x : wp.array(dtype=wp.vec3)
        Array of particle positions in 3D space.
    particle_rho : wp.array(dtype=float)
        Output array where computed density values are stored for each particle.
    density_normalization : float
        Normalization factor applied to the final density computation.
    h : float
        Kernel smoothing length (radius of influence) defining the neighborhood
        size for density computation.
    """

    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # get local particle variables
    x = particle_x[i]

    # store density
    rho = float(0.0)

    # particle contact
    neighbors = wp.hash_grid_query(grid, x, h)

    # loop through neighbors to compute density
    for index in neighbors:
        # compute distance
        distance = x - particle_x[index]

        # compute kernel derivative
        rho += density_kernel(distance, h)

    # add external potential
    particle_rho[i] = density_normalization * rho

@wp.kernel
def get_acceleration(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    particle_a: wp.array(dtype=wp.vec3),
    isotropic_exp: float,
    base_density: float,
    gravity: float,
    pressure_normalization: float,
    viscous_normalization: float,
    h: float,
):
    """
    Calculate the acceleration for each SPH particle due to pressure, viscosity, and external forces.

    This kernel computes the acceleration of each particle in a Smoothed Particle Hydrodynamics (SPH)
    simulation by iterating over neighboring particles within the smoothing length and accumulating
    pressure and viscous forces. The acceleration is then computed by dividing the total force by
    the particle's density and adding gravitational acceleration.

    Parameters
    ----------
    grid : wp.uint64
        Hash grid structure used for efficient spatial neighbor queries.
    particle_x : wp.array(dtype=wp.vec3)
        Array of particle positions.
    particle_v : wp.array(dtype=wp.vec3)
        Array of particle velocities.
    particle_rho : wp.array(dtype=float)
        Array of particle densities.
    particle_a : wp.array(dtype=wp.vec3)
        Output array where computed particle accelerations will be stored.
    isotropic_exp : float
        Isotropic exponent (stiffness parameter) for the equation of state relating pressure to density.
    base_density : float
        Rest density (reference density) of the fluid.
    gravity : float
        Gravitational acceleration magnitude in the y-direction.
    pressure_normalization : float
        Normalization factor applied to the pressure force contribution.
    viscous_normalization : float
        Normalization factor applied to the viscous force contribution.
    h : float
        Kernel smoothing length defining the radius of influence for particle interactions.
    """

    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # get local particle variables
    x = particle_x[i]
    v = particle_v[i]
    rho = particle_rho[i]
    pressure = isotropic_exp * (rho - base_density)

    # store forces
    pressure_force = wp.vec3()
    viscous_force = wp.vec3()

    # particle contact
    neighbors = wp.hash_grid_query(grid, x, h)

    # loop through neighbors to compute acceleration
    for index in neighbors:
        if index != i:
            # get neighbor velocity
            neighbor_v = particle_v[index]

            # get neighbor density and pressures
            neighbor_rho = particle_rho[index]
            neighbor_pressure = isotropic_exp * (neighbor_rho - base_density)

            # compute relative position
            relative_position = particle_x[index] - x

            # calculate pressure force
            pressure_force += diff_pressure_kernel(
                relative_position, pressure, neighbor_pressure, neighbor_rho, h
            )

            # compute kernel derivative
            viscous_force += diff_viscous_kernel(relative_position, v, neighbor_v, neighbor_rho, h)

    # sum all forces
    force = pressure_normalization * pressure_force + viscous_normalization * viscous_force

    # add external potential
    particle_a[i] = force / rho + wp.vec3(0.0, gravity, 0.0)

@wp.kernel
def apply_bounds(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    damping_coef: float,
    xl: float,
    xr: float,
    xs: float,
    yb: float,
    zl: float,
    zr: float,
    zs: float
):
    """
    Apply boundary conditions to particles with collision detection and response.

    This kernel enforces spatial constraints on particles within a container with slanted walls.
    The container has a flat bottom (yb) and four slanted walls whose inclination depends on
    the slope parameters xs and zs. When particles collide with boundaries, their positions 
    are clamped and velocities are adjusted using reflection with damping.

    The container boundaries are defined as:
    - Bottom: y >= yb (horizontal plane)
    - X-axis walls: x ∈ [-y/xs + xl, y/xs + xr] (slanted walls)
    - Z-axis walls: z ∈ [-y/zs + zl, y/zs + zr] (slanted walls)

    With positive xs and zs values, the container widens as y increases (inverted pyramid).
    With negative xs and zs values, the container narrows as y increases (pyramid).

    Parameters
    ----------
    particle_x : wp.array(dtype=wp.vec3)
        Array of particle positions to be constrained.
    particle_v : wp.array(dtype=wp.vec3)
        Array of particle velocities to be adjusted upon collision.
    damping_coef : float
        Coefficient for velocity damping on collision (typically negative, e.g., -0.95).
        The normal component of velocity is multiplied by this factor.
    xl : float
        Left boundary offset on x-axis at y=0.
    xr : float
        Right boundary offset on x-axis at y=0.
    xs : float
        Slope factor for x-axis walls. Positive values widen the container with height,
        negative values narrow it.
    yb : float
        Bottom boundary on y-axis (minimum y coordinate).
    zl : float
        Left boundary offset on z-axis at y=0.
    zr : float
        Right boundary offset on z-axis at y=0.
    zs : float
        Slope factor for z-axis walls. Positive values widen the container with height,
        negative values narrow it.

    Notes
    -----
    The velocity reflection formula used is:
    v_new = v - (v · n) · n · (1 - damping_coef)
    where n is the outward normal vector of the colliding wall.

    v_new = v_tangential + v_normal_new
          = v_tangential + (damping_coef · v_normal)
          = v_tangential + damping_coef · v_normal
          =(v - v_normal) + damping_coef · v_normal
          = v - v_normal + damping_coef · v_normal
          = v - v_normal · (1 - damping_coef)
          = v - (v · n) · n · (1 - damping_coef)
    """
    tid = wp.tid()

    # get pos and velocity
    p = particle_x[tid]
    v = particle_v[tid]

    # clamp y bottom
    colliding_y = wp.bool(p[1] < yb)   
    p = wp.vec3(p[0],
                max(yb, p[1]),
                p[2])
    
    v = wp.vec3(v[0],
                wp.where(colliding_y, v[1] * damping_coef, v[1]),
                v[2])
    
    
    # clamp x. x must be in [-p[1]/xs + xl; p[1]/xs + xr]
    offset_x = wp.float(p[1]/xs)

    # collision xl (left on x axis)
    colliding_xl = wp.bool(p[0] < -offset_x + xl)

    p = wp.vec3(wp.max(-offset_x + xl, p[0]),
                p[1],
                p[2])

    nxl = wp.normalize(wp.vec3(xs, 1.0, 0.0))
    v = wp.where(colliding_xl, v - wp.dot(v, nxl) * nxl * (1.0 - damping_coef), v)

    # collision xr (right on x axis)
    colliding_xr = wp.bool(p[0] > offset_x + xr)

    p = wp.vec3(wp.min(p[0], offset_x + xr),
                p[1],
                p[2])
    
    nxr = wp.normalize(wp.vec3(-xs, 1.0, 0.0))
    v = wp.where(colliding_xr, v - wp.dot(v, nxr) * nxr * (1.0 - damping_coef), v)

    # clamp z. z must be in [-p[1]/zs + zl; p[1]/zs + zr]
    offset_z = wp.float(p[1]/zs)

    # collision zl (left on z axis)
    colliding_zl = wp.bool(p[2] < -offset_z + zl)

    p = wp.vec3(p[0],
                p[1],
                wp.max(-offset_z + zl, p[2]))

    nzl = wp.normalize(wp.vec3(0.0, 1.0, zs))
    v = wp.where(colliding_zl, v - wp.dot(v, nzl) * nzl * (1.0 - damping_coef), v)

    # collision zr (right on z axis)
    colliding_zr = wp.bool(p[2] > offset_z + zr)

    p = wp.vec3(p[0],
                p[1],
                wp.min(p[2], offset_z + zr))
    
    nzr = wp.normalize(wp.vec3(0.0, 1.0, -zs))
    v = wp.where(colliding_zr, v - wp.dot(v, nzr) * nzr * (1.0 - damping_coef), v)

    # apply clamps
    particle_x[tid] = p
    particle_v[tid] = v

@wp.kernel
def kick(particle_v: wp.array(dtype=wp.vec3), particle_a: wp.array(dtype=wp.vec3), dt: float):
    """
        Update particle velocities using the kick step in a leapfrog integration scheme.

        This function performs a velocity update (kick) by adding the acceleration contribution
        over a time step. It is typically used as part of a symplectic integration scheme for
        particle dynamics in SPH simulations.

        Parameters
        ----------
        particle_v : wp.array(dtype=wp.vec3)
            Array of particle velocities to be updated in-place.
        particle_a : wp.array(dtype=wp.vec3)
            Array of particle accelerations used for the velocity update.
        dt : float
            Time step size for the integration.
    """
    
    tid = wp.tid()
    v = particle_v[tid]
    particle_v[tid] = v + particle_a[tid] * dt

@wp.kernel
def drift(particle_x: wp.array(dtype=wp.vec3), particle_v: wp.array(dtype=wp.vec3), dt: float):
    """
    Update particle positions based on their velocities using explicit Euler integration.

    This function performs a simple drift step in a particle simulation by advancing
    each particle's position according to its current velocity over a given time step.

    Parameters
    ----------
    particle_x : wp.array(dtype=wp.vec3)
        Array of particle positions to be updated in-place.
    particle_v : wp.array(dtype=wp.vec3)
        Array of particle velocities used for position updates.
    dt : float
        Time step size for the integration.
    """

    tid = wp.tid()
    x = particle_x[tid]
    particle_x[tid] = x + particle_v[tid] * dt

@wp.kernel
def initialize_particles(
    particle_x: wp.array(dtype=wp.vec3), dp: float,
    width: float, height: float, length: float,
    x0: float, y0: float, z0: float
):
    """
    Initialize particle positions in a 3D grid with random jitter for SPH simulation.

    This kernel function sets up initial particle positions arranged in a regular 3D grid
    pattern within a rectangular domain. Each particle position is perturbed by a small
    random offset to prevent artificial regularity in the initial configuration.

    Parameters
    ----------
    particle_x : wp.array(dtype=wp.vec3)
        Output array to store particle positions. Must be pre-allocated with sufficient
        size to hold all particles in the grid.
    dp : float
        Inter-particle distance (spacing) in the simulation.
    width : float
        Width of the fluid domain in the x-direction.
    height : float
        Height of the fluid domain in the y-direction.
    length : float
        Length of the fluid domain in the z-direction.
    x0 : float, optional
        Starting x-coordinate offset for the grid (default is 0.0).
    y0 : float, optional
        Starting y-coordinate offset for the grid (default is 0.0).
    z0 : float, optional
        Starting z-coordinate offset for the grid (default is 0.0).
    """
    
    tid = wp.tid()

    # grid size
    nr_x = wp.int32(width  / dp)
    nr_y = wp.int32(height / dp)
    nr_z = wp.int32(length / dp)

    # calculate particle position
    z = wp.float(tid % nr_z)
    y = wp.float((tid // nr_z) % nr_y)
    x = wp.float((tid // (nr_z * nr_y)) % nr_x)
    pos = dp * wp.vec3(x, y, z) + wp.vec3(x0, y0, z0)

    # add small jitter
    state = wp.rand_init(123, tid)
    pos = pos + 0.001 * dp * wp.vec3(wp.randn(state), wp.randn(state), wp.randn(state))

    # set position
    particle_x[tid] = pos

class SPH_Simulation:
    def __init__(self, stage_path="example_sph.usd", verbose=False):
        """
        Initialize the SPH (Smoothed Particle Hydrodynamics) simulation.

        This constructor sets up all parameters, arrays, and data structures needed for an SPH
        fluid simulation, including particle properties, physical constants, and rendering components.

        Parameters
        ----------
        stage_path : str, optional
            Path to the USD file for rendering output. If provided, initializes a USD renderer.
            Default is "example_sph.usd".
        verbose : bool, optional
            Flag to enable verbose output during simulation. Default is False.
        """
        # Verbose output flag.
        self.verbose = verbose        
        ###########################################################################
        # FLUID SIMULATION PARAMS (SPATIAL)
        # Inter-particle distance (initial spacing between adjacent particles).
        # Sets the resolution of the simulation: lower values = more particles = higher accuracy.
        # ? Particle count: 1/(dp)³ per unit volume
        # ? Each particle typically occupies a volume of dp³
        # ? Default: 0.1 cm. Smaller values raise the computational cost sharply
        # ? Typical values: 0.5 cm (fast), 0.25 cm (medium), 0.1 cm (slow), 0.05 cm (very slow)
        self.dp = 0.1  # [cm] - inter-particle distance
        # Coefficient used to derive the smoothing length from the particle parameters.
        # Controls the width of the SPH kernel: h = coefh × √(3 × dp²)
        # ? Typical values for 3D: 1.2-1.3
        # ? Larger coefh → wider kernels → more smoothing → more neighbours to evaluate
        self.coefh = 1.3  # []
        # SPH kernel smoothing length (radius of influence for particle interactions).
        # ? Derived automatically from dp and coefh with the formula: h = coefh × √(3 × dp²)
        # ? With coefh=1.3 this gives h ≈ 2.25 × dp
        # ? Particles interact only if they are closer than h to each other.
        self.h = self.coefh * (3.0 * self.dp**2)**0.5  # [cm]
        # Initial position offsets for particle placement.
        self.x0 = 0.0   # [cm]
        self.y0 = 0.0   # [cm]
        self.z0 = 0.0   # [cm]
        # Fluid block dimensions.
        self.width = 12.0   # [cm] - x direction
        self.height = 2.0   # [cm] - y direction
        self.length = 6.1   # [cm] - z direction
        # Total number of particles in the simulation.
        self.n = int(self.height * self.width * self.length / (self.dp**3))
        # Boundaries parameters
        self.xl = 0.0   # [cm] - left boundary x at y=0
        self.xr = 12.0  # [cm] - right boundary x at y=0
        self.xs = 1e4   # []   - x-axis wall slope
        self.yb = 0.0   # [cm] - bottom boundary y
        self.zl = 0.0   # [cm] - left boundary z at y=0
        self.zr = 6.1   # [cm] - right boundary z at y=0
        self.zs = 1e4   # []   - z-axis wall slope
        # ? with slope 1e4 the walls are almost vertical
        # ? at height y0, the error in x due to slope is y0/1e4
        ###########################################################################
        # FLUID SIMULATION PARAMS (PHYSICAL)
        # Reference density of the fluid.
        self.base_density = 1.0 # [g/cm³] - water at standard conditions
        # Exponent for isotropic pressure calculations.
        self.isotropic_exp = 100 # [cm²/s²] - stiffness (~100-500 for water)
        # Mass of each particle.
        # ? Mass = density × volume = base_density × dp³
        self.particle_mass = self.base_density * self.dp**3 # [g]
        # Dynamic viscosity coefficient.
        self.dynamic_visc = 0.01   # [g/(cm·s)] = [Poise] - internal fluid friction (0.01 for water)
        # Damping coefficient for boundary collisions.
        self.damping_coef = -0.95   # [] - negative for damping
        # Gravitational acceleration (negative for downward).
        self.gravity = -981.0 # [cm/s²]
        ###########################################################################
        # SIM/RENDER TIME PARAMS
        # Time step for rendering frames (1/fps).
        self.fps = 60   # [Hz] = [1/s]
        # Total number of frames to simulate, set by simulate() method
        self.tot_frames = 0
        # Time interval between rendered frames.
        self.frame_dt = 1.0 / self.fps # [s] - real time between frames
        # Current simulation time, initialized to 0.0.
        self.sim_time = 0.0
        # Current frame index in the simulation sequence
        self.current_frame = 0
        # Number of simulation steps per rendered frame.
        self.substeps = 420 # [] - number of sim steps per frame
        # Time between each sub-step of the physical simulation calculation.
        self.step_dt = self.frame_dt / self.substeps  # [s] - time per sim step
        ###########################################################################
        # CONSTANTS
        # Normalization constant for density kernel integration.
        self.density_normalization = (315.0 * self.particle_mass) / (
            64.0 * np.pi * self.h**9
        )
        # Normalization constant for pressure gradient kernel.
        self.pressure_normalization = -(45.0 * self.particle_mass) / (np.pi * self.h**6)
        # Normalization constant for viscosity kernel.
        self.viscous_normalization = (45.0 * self.dynamic_visc * self.particle_mass) / (
            np.pi * self.h**6
        )
        ###########################################################################
        # ALLOCATE ARRAYS
        # Particle positions.
        self.x = wp.empty(self.n, dtype=wp.vec3)
        # Particle velocities.
        self.v = wp.zeros(self.n, dtype=wp.vec3)
        # Particle densities.
        self.rho = wp.zeros(self.n, dtype=float)
        # Particle accelerations.
        self.a = wp.zeros(self.n, dtype=wp.vec3)
        ###########################################################################
        # INITIALIZE PARTICLES
        wp.launch(
            kernel=initialize_particles,
            dim=self.n,
            inputs=[self.x, self.dp, self.width, self.height, self.length, self.x0, self.y0, self.z0],
        )
        # Store maximum heights for analysis/visualization
        self.max_heights = []
        ###########################################################################
        # HASH GRID CONFIGURATION
        # 
        # The HashGrid is a spatial acceleration structure that maps 3D space into
        # a finite hash table for efficient neighbor queries.
        # 
        # Two independent parameters control its behavior:
        # 
        # 1. CELL SIZE (radius parameter in build()):
        #    - Set to h (smoothing length)
        #    - Defines the physical size of spatial cells: each cell is a cube of size h×h×h
        #    - Particles interact only within distance h, so cells of size h are optimal
        # 
        # 2. HASH TABLE SIZE (dim_x, dim_y, dim_z in constructor):
        #    - Number of buckets in the hash table: dim_x × dim_y × dim_z
        #    - Multiple cells map to the same bucket (hash collisions)
        #    - Trade-off: more buckets = less collisions but more memory
        # 
        # Grid Size Selection Strategy:
        # - We divide space into "macro-cells" of size (factor×h) to determine bucket count
        # - Factor of 4 means ~64 physical cells map to 1 bucket (acceptable collision rate)
        # - Larger factor (5-6) = fewer buckets, more collisions, less memory
        # - Smaller factor (2-3) = more buckets, fewer collisions, more memory
        # 
        # Domain Coverage:
        # - Based on CONTAINER GEOMETRY at maximum fluid height (with sloshing)
        # - Container has slanted walls that widen with height:
        #   * x-width at height y: (xr - xl) + 2×(y/xs)
        #   * z-depth at height y: (zr - zl) + 2×(y/zs)
        # - We estimate max fluid height considering sloshing and use container dimensions there
        # 
        # Note: The hash grid has NO spatial boundaries. It maps any coordinate in
        # infinite 3D space into the finite hash table. Grid dimensions should match
        # the EXTENT of the fluid distribution based on container shape.

        # Hash grid cell size factor (controls memory vs collision trade-off)
        # Typical values: 3.0 (more memory), 4.0 (balanced), 5.0 (less memory)
        self.hash_grid_factor = 4.0

        # Sloshing factor: fluid can rise higher than initial fill level
        # During violent sloshing, fluid height can increase by 2-3×
        self.sloshing_factor = 3.0

        # Calculate maximum fluid height considering sloshing
        max_fluid_height = self.height * self.sloshing_factor  # [cm]

        # Calculate container dimensions at maximum fluid height
        # Container widens due to slanted walls: width(y) = base_width + 2×(y/slope)
        container_width_at_max_height = (self.xr - self.xl) # + 2.0 * (max_fluid_height / self.xs)
        container_depth_at_max_height = (self.zr - self.zl) # + 2.0 * (max_fluid_height / self.zs)

        # Use container dimensions at max height as fluid extent
        # (fluid conforms to container shape)
        domain_extent_x = container_width_at_max_height   # [cm]
        domain_extent_y = max_fluid_height                # [cm]
        domain_extent_z = container_depth_at_max_height   # [cm]

        # Calculate anisotropic grid dimensions based on fluid extent in container
        # Each dimension is sized to cover the extent with macro-cells of size (factor × h)
        grid_dim_x = int(domain_extent_x / (self.hash_grid_factor * self.h))
        grid_dim_y = int(domain_extent_y / (self.hash_grid_factor * self.h))
        grid_dim_z = int(domain_extent_z / (self.hash_grid_factor * self.h))

        # Ensure minimum grid dimensions
        grid_dim_x = max(grid_dim_x, 1)
        grid_dim_y = max(grid_dim_y, 1)
        grid_dim_z = max(grid_dim_z, 1)

        # Total hash table buckets
        total_buckets = grid_dim_x * grid_dim_y * grid_dim_z

        if self.verbose:
            print(f"\n=== Hash Grid Configuration ===")
            print(f"Smoothing length h: {self.h:.4f} cm")
            print(f"Hash grid factor: {self.hash_grid_factor}")
            print(f"Sloshing factor: {self.sloshing_factor}")
            print(f"Macro-cell size: {self.hash_grid_factor * self.h:.4f} cm")
            print(f"\nContainer geometry (slanted walls):")
            print(f"  At y=0 (bottom):  width={self.xr - self.xl:.1f} cm, depth={self.zr - self.zl:.1f} cm")
            print(f"  At y={max_fluid_height:.1f} (max): width={container_width_at_max_height:.1f} cm, depth={container_depth_at_max_height:.1f} cm")
            print(f"\nFluid extent approximation (parallelepiped):")
            print(f"  X={domain_extent_x:.1f} cm, Y={domain_extent_y:.1f} cm, Z={domain_extent_z:.1f} cm")
            print(f"\nHash grid dimensions: {grid_dim_x} × {grid_dim_y} × {grid_dim_z}")
            print(f"Total hash buckets: {total_buckets:,}")
            
            # Estimate average particles per physical cell
            volume_per_cell = self.h ** 3
            initial_particle_density = self.n / (self.width * self.height * self.length)
            particles_per_physical_cell = volume_per_cell * initial_particle_density
            print(f"\nParticle distribution:")
            print(f"  Particles per physical cell (h³): {particles_per_physical_cell:.1f}")
            
            # Estimate theoretical cells in fluid domain
            theoretical_cells_x = int(domain_extent_x / self.h)
            theoretical_cells_y = int(domain_extent_y / self.h)
            theoretical_cells_z = int(domain_extent_z / self.h)
            total_theoretical_cells = theoretical_cells_x * theoretical_cells_y * theoretical_cells_z
            collision_factor = total_theoretical_cells / total_buckets if total_buckets > 0 else 0
            print(f"  Theoretical physical cells in domain: {total_theoretical_cells:,}")
            print(f"  Average cells per bucket (collision factor): {collision_factor:.1f}")
            print(f"\nNote: Container movement does NOT affect grid sizing.")
            print(f"      Grid covers fluid extent based on container shape, not position.")
            print("=" * 60)

        # Create spatial hash grid for efficient neighbor search
        # This maps infinite 3D space into a finite hash table
        self.grid = wp.HashGrid(grid_dim_x, grid_dim_y, grid_dim_z)
        ###########################################################################
        # RENDERER
        # USD renderer for visualization, or None if no stage_path provided.
        self.renderer = None
        if stage_path:
            self.renderer = wp.render.UsdRenderer(stage_path)
        ###########################################################################

    def move_container(self, dx: float, dz: float, dy: float = 0.0):
        """
        Translate the container boundaries in 3D space.

        This method shifts all container boundaries by the specified offsets,
        effectively moving the entire simulation domain.

        Parameters
        ----------
        dx : float
            Displacement along the x-axis.
        dz : float
            Displacement along the z-axis.
        dy : float, optional
            Displacement along the y-axis. Default is 0.0.
        """
        self.xl += dx
        self.xr += dx
        self.yb += dy
        self.zl += dz
        self.zr += dz
    
    def get_delta_x(self):
        """
        Calculate x-axis displacement for current frame based on acceleration profile.
        
        Motion profile (heavy sloshing, single sequence):
        - Phase 0 (0.0 - 0.75s): Settling time (stationary)
        - Phase 1 (0.75 - 1.5s): +160 cm/s² acceleration
        - Phase 2 (1.5 - 1.75s): 0 cm/s² (constant velocity)
        - Phase 3 (1.75 - 2.25s): -240 cm/s² deceleration
        - Phase 4 (2.25s - end): 0 cm/s² (stationary)
        
        Returns
        -------
        float
            Container displacement in [cm] for this frame
        """
        # Phase boundaries (absolute time)
        t_settling = 0.75 # [s] - end of settling phase
        t1 = 1.5          # [s] - end of acceleration phase
        t2 = 1.75         # [s] - end of constant velocity phase
        t3 = 2.25         # [s] - end of deceleration phase

        # Acceleration values
        a1 = 160.0   # [cm/s²] - positive acceleration
        a3 = -240.0  # [cm/s²] - negative acceleration
        
        t = self.sim_time
        
        # Phase 0: Settling (stationary)
        if t < t_settling:
            velocity = 0.0
            
        # Phase 1: Positive acceleration
        elif t < t1:
            # v(t) = a1 * (t - t_settling)
            # Starting from rest at t = t_settling
            velocity = a1 * (t - t_settling)
            
        # Phase 2: Constant velocity
        elif t < t2:
            # v = v_peak (reached at end of phase 1)
            v_peak = a1 * (t1 - t_settling)  # = 160 * 0.75 = 120 cm/s
            velocity = v_peak
            
        # Phase 3: Negative acceleration (braking)
        elif t < t3:
            # v(t) = v_peak + a3 * (t - t2)
            v_peak = a1 * (t1 - t_settling)  # = 120 cm/s
            velocity = v_peak + a3 * (t - t2)
            
        # Phase 4: Rest (stationary)
        else:
            velocity = 0.0
        
        # Displacement for this frame: Δx = v * Δt
        displacement = velocity * self.frame_dt  # [cm]
        
        return displacement
    
    def get_delta_z(self):
        """
        Calculate the z-axis displacement for the current frame.

        This method computes how much the container should move along the z-axis
        based on the current simulation time.

        Returns
        -------
        float
            The z-axis displacement for this frame.

        Notes
        -----
        Currently returns 0.0. This should be implemented with time-dependent
        logic if motion along z-axis is required.
        """
        # self.sim_time
        # self.frame_dt
        return 0.0

    def get_delta_y(self):
        pass

    def get_max_fluid_height(self):
        """
        Calculate the maximum y-coordinate (height) of all fluid particles.
        
        Returns
        -------
        float
            Maximum height reached by any particle in [cm]
        """
        # Copy particle positions to CPU and find max y coordinate
        positions = self.x.numpy()  # Shape: (n, 3)
        max_y = np.max(positions[:, 1])  # Get max of y-coordinates (index 1)
        return max_y

    def simulate(self, num_frames=600):
        """
        Run the SPH simulation for a specified number of frames.

        This method iteratively advances the simulation state and renders each frame.
        It calls the `step()` method to perform the physics calculations and the `render()`
        method to visualize the current state.

        Parameters
        ----------
        num_frames : int, optional
            Total number of frames to simulate. Default is 600.
        """
        self.tot_frames = num_frames
        self.current_frame = 0

        for frame in range(num_frames):
            self.current_frame = frame
            print(f"Frame {frame + 1}/{num_frames}")
            self.step()
            self.render()
            self.move_container(self.get_delta_x(), self.get_delta_z())


        if self.renderer:
            self.renderer.save()
        
        import csv
        with open('max_heights.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Time [s]', 'Max Height [cm]'])
            for i, h in enumerate(self.max_heights):
                writer.writerow([i, i * self.frame_dt, h])
        print(f"Max heights saved to max_heights.csv")
        
    def step(self):
        """
        Advance the SPH simulation by one frame.

        This method performs multiple substeps of the SPH algorithm for each rendered frame:
        1. Builds the spatial hash grid for efficient neighbor queries
        2. Computes particle densities using SPH kernel summation
        3. Calculates particle accelerations from pressure and viscosity forces
        4. Applies boundary conditions and collision responses
        5. Updates velocities (kick step) and positions (drift step)
        6. Moves the container according to time-dependent motion

        The method uses a leapfrog integration scheme for temporal discretization.

        Notes
        -----
        The number of substeps per frame is determined by the `substeps` attribute
        to ensure numerical stability regardless of the rendering frame rate.
        After all substeps, the simulation time is advanced by `frame_dt`.
        """
        with wp.ScopedTimer("step"):
            for _ in range(self.substeps):
                with wp.ScopedTimer("grid build", active=self.verbose):
                    # build grid
                    self.grid.build(self.x, self.h)

                with wp.ScopedTimer("forces", active=self.verbose):
                    # compute density of points
                    wp.launch(
                        kernel=compute_density,
                        dim=self.n,
                        inputs=[self.grid.id, self.x, self.rho, self.density_normalization, self.h],
                    )

                    # get new acceleration
                    wp.launch(
                        kernel=get_acceleration,
                        dim=self.n,
                        inputs=[
                            self.grid.id,
                            self.x,
                            self.v,
                            self.rho,
                            self.a,
                            self.isotropic_exp,
                            self.base_density,
                            self.gravity,
                            self.pressure_normalization,
                            self.viscous_normalization,
                            self.h,
                        ],
                    )

                    # apply bounds
                    wp.launch(
                        kernel=apply_bounds,
                        dim=self.n,
                        inputs=[self.x, self.v, self.damping_coef, self.xl, self.xr, self.xs, self.yb, self.zl, self.zr, self.zs],
                    )

                    # kick
                    wp.launch(kernel=kick, dim=self.n, inputs=[self.v, self.a, self.step_dt])

                    # drift
                    wp.launch(kernel=drift, dim=self.n, inputs=[self.x, self.v, self.step_dt])

            max_height = self.get_max_fluid_height()
            self.max_heights.append(max_height)

            self.sim_time += self.frame_dt

    def render(self):
        """
        Render the current state of the SPH simulation.

        This function visualizes the particle system by rendering particle positions using
        the configured renderer. If no renderer is available, the function returns without
        performing any rendering operations. The rendering is wrapped in a timer for
        performance monitoring.
        """

        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_points(
                points=self.x.numpy(), radius=self.h, name="points", colors=(0.0, 0.4, 0.8)
            )
            self.renderer.end_frame()

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="sph_sim.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=240, help="Total number of frames.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")

    args = parser.parse_known_args()[0]
    
    starting_time = time.time()

    with wp.ScopedDevice(args.device):
        sim = SPH_Simulation(stage_path=args.stage_path, verbose=args.verbose)
        sim.simulate(args.num_frames)

    elapsed_time = time.time() - starting_time
    print(f"Simulation completed in {elapsed_time:.2f} seconds.")