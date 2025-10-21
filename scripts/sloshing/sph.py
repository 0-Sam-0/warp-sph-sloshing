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
def density_kernel(xyz: wp.vec3, smoothing_length: float):
    """
    Compute the density contribution kernel for SPH (Smoothed Particle Hydrodynamics).

    This kernel calculates a smoothed density value based on the distance from a point
    to a particle. The kernel returns zero beyond the smoothing length and follows a
    cubic polynomial within the support radius.

    Parameters
    ----------
    xyz : wp.vec3
        The relative position (difference) vector between two particles.
    smoothing_length : float
        The smoothing length (support radius) of the kernel.
        Particles beyond this distance do not contribute to the density calculation.

    Returns
    -------
    float
        The kernel value. Returns a non-negative value proportional to the cube of
        (smoothing_length² - distance_squared) if the distance is within the smoothing
        length, otherwise returns 0.0.
    """
    # calculate distance
    distance_squared = wp.dot(xyz, xyz)

    return wp.max(cube(square(smoothing_length) - distance_squared), 0.0)

@wp.func
def diff_pressure_kernel(
    xyz: wp.vec3, pressure: float, neighbor_pressure: float, neighbor_rho: float, smoothing_length: float
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
    smoothing_length : float
        Kernel smoothing length (radius of influence).
    
    Returns
    -------
    wp.vec3
        Pressure gradient contribution vector. Returns zero vector if neighbor is outside
        the smoothing length, otherwise returns the weighted pressure gradient term.
    """
    
    # calculate distance
    distance = wp.sqrt(wp.dot(xyz, xyz))

    if distance < smoothing_length:
        # calculate terms of kernel
        term_1 = -xyz / distance
        term_2 = (neighbor_pressure + pressure) / (2.0 * neighbor_rho)
        term_3 = square(smoothing_length - distance)
        return term_1 * term_2 * term_3
    else:
        return wp.vec3()

@wp.func
def diff_viscous_kernel(xyz: wp.vec3, v: wp.vec3, neighbor_v: wp.vec3, neighbor_rho: float, smoothing_length: float):
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
    if distance < smoothing_length:
        term_1 = (neighbor_v - v) / neighbor_rho
        term_2 = smoothing_length - distance
        return term_1 * term_2
    else:
        return wp.vec3()

@wp.kernel
def compute_density(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    density_normalization: float,
    smoothing_length: float,
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
    smoothing_length : float
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
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)

    # loop through neighbors to compute density
    for index in neighbors:
        # compute distance
        distance = x - particle_x[index]

        # compute kernel derivative
        rho += density_kernel(distance, smoothing_length)

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
    smoothing_length: float,
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
    smoothing_length : float
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
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)

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
                relative_position, pressure, neighbor_pressure, neighbor_rho, smoothing_length
            )

            # compute kernel derivative
            viscous_force += diff_viscous_kernel(relative_position, v, neighbor_v, neighbor_rho, smoothing_length)

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
    particle_x: wp.array(dtype=wp.vec3), smoothing_length: float,
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
    smoothing_length : float
        SPH kernel smoothing length, used as the base spacing between particles in the grid.
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
    nr_x = wp.int32(width  / smoothing_length)
    nr_y = wp.int32(height / smoothing_length)
    nr_z = wp.int32(length / smoothing_length)

    # calculate particle position
    z = wp.float(tid % nr_z)
    y = wp.float((tid // nr_z) % nr_y)
    x = wp.float((tid // (nr_z * nr_y)) % nr_x)
    pos = smoothing_length * wp.vec3(x, y, z) + wp.vec3(x0, y0, z0)

    # add small jitter
    state = wp.rand_init(123, tid)
    pos = pos + 0.001 * smoothing_length * wp.vec3(wp.randn(state), wp.randn(state), wp.randn(state))

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
        # SPH kernel smoothing length (radius of influence for particle interactions).
        # ? Particle count 1/(smoothing_length)³ per cube-unit.
        # ? In other terms, each particle typically occupies a volume of smoothing_length³.
        # ? Default: 0.8. Lower => more precision but much slower
        # ? 1/(num_particles_for_each_cm_cubed**(1/3))
        self.smoothing_length = 1/(27**(1/3)) # [cm]
        # Initial position offsets for particle placement.
        self.x0 = 0.0   # [cm]
        self.y0 = 0.0   # [cm]
        self.z0 = 0.0   # [cm]
        # Fluid block dimensions.
        self.width = 40.0   # [cm] - x direction
        self.height = 20.0  # [cm] - y direction
        self.length = 20.0  # [cm] - z direction
        # Total number of particles in the simulation.
        self.n = int(self.height * self.width * self.length / (self.smoothing_length**3))
        # Boundaries parameters
        self.xl = 0.0   # [cm] - left boundary x at y=0
        self.xr = 40.0  # [cm] - right boundary x at y=0
        self.xs = 2.0   # []   - x-axis wall slope
        self.yb = 0.0   # [cm] - bottom boundary y
        self.zl = 0.0   # [cm] - left boundary z at y=0
        self.zr = 20.0  # [cm] - right boundary z at y=0
        self.zs = 4.0   # []   - z-axis wall slope
        ###########################################################################
        # FLUID SIMULATION PARAMS (PHYSICAL)
        # Reference density of the fluid.
        self.base_density = 1.0 # [g/cm³] - water at standard conditions
        # Exponent for isotropic pressure calculations.
        self.isotropic_exp = 100 # [cm²/s²] - stiffness (~100-500 for water)
        # Mass of each particle, scaled by smoothing length cubed.
        #? Mass proportional to smoothing length cubed
        self.particle_mass = self.base_density * self.smoothing_length**3   # ! [g] - (originally: 0.01 * self.smoothing_length³)
        # Dynamic viscosity coefficient.
        self.dynamic_visc = 0.01   # [g/(cm·s)] = [Poise] - internal fluid friction
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
        self.frame_dt = 1.0 / self.fps # [s]
        # Current simulation time, initialized to 0.0.
        self.sim_time = 0.0
        # Current frame index in the simulation sequence
        self.current_frame = 0 
        # Number of simulation steps per rendered frame. Note: 32 is a random choice.
        self.substeps = int(32 / self.smoothing_length) # !
        # Time between each sub-step of the physical simulation calculation.
        self.dt = self.frame_dt / self.substeps # ! [s] (originally: 0.01 * self.smoothing_length)
        ###########################################################################
        # CONSTANTS
        # Normalization constant for density kernel integration.
        self.density_normalization = (315.0 * self.particle_mass) / (
            64.0 * np.pi * self.smoothing_length**9
        )
        # Normalization constant for pressure gradient kernel.
        self.pressure_normalization = -(45.0 * self.particle_mass) / (np.pi * self.smoothing_length**6)
        # Normalization constant for viscosity kernel.
        self.viscous_normalization = (45.0 * self.dynamic_visc * self.particle_mass) / (
            np.pi * self.smoothing_length**6
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
            inputs=[self.x, self.smoothing_length, self.width, self.height, self.length, self.x0, self.y0, self.z0],
        )
        ###########################################################################
        # HASH GRID
        # create hash array
        grid_size = int(self.height / (4.0 * self.smoothing_length))
        # Spatial hash grid for efficient neighbor search.
        self.grid = wp.HashGrid(grid_size, grid_size, grid_size)
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
        Calculate the x-axis displacement for the current frame.

        This method computes how much the container should move along the x-axis
        based on the current simulation time.

        Returns
        -------
        float
            The x-axis displacement for this frame.

        Notes
        -----
        Currently the container remains stationary during the initial phase (first 20% of frames)
        to allow the fluid to settle, then begins moving at constant velocity.
        """
        # self.sim_time
        # self.frame_dt
        
        if self.current_frame <= self.tot_frames * 0.2:
            return 0.0
        elif self.current_frame <= self.tot_frames * 0.5:
            return 0.025
        elif self.current_frame <= self.tot_frames * 0.8:
            return 0.0125
        elif self.current_frame <= self.tot_frames * 1.0:
            return 0.0
    
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
            self.render()
            self.step()

        if self.renderer:
            self.renderer.save()
        
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
                    self.grid.build(self.x, self.smoothing_length)

                with wp.ScopedTimer("forces", active=self.verbose):
                    # compute density of points
                    wp.launch(
                        kernel=compute_density,
                        dim=self.n,
                        inputs=[self.grid.id, self.x, self.rho, self.density_normalization, self.smoothing_length],
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
                            self.smoothing_length,
                        ],
                    )

                    # apply bounds
                    wp.launch(
                        kernel=apply_bounds,
                        dim=self.n,
                        inputs=[self.x, self.v, self.damping_coef, self.xl, self.xr, self.xs, self.yb, self.zl, self.zr, self.zs],
                    )

                    # kick
                    wp.launch(kernel=kick, dim=self.n, inputs=[self.v, self.a, self.dt])

                    # drift
                    wp.launch(kernel=drift, dim=self.n, inputs=[self.x, self.v, self.dt])

                    self.move_container(self.get_delta_x(), self.get_delta_z())

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
                points=self.x.numpy(), radius=self.smoothing_length, name="points", colors=(0.0, 0.4, 0.8)
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
    parser.add_argument("--num_frames", type=int, default=600, help="Total number of frames.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")

    args = parser.parse_known_args()[0]
    
    starting_time = time.time()

    with wp.ScopedDevice(args.device):
        sim = SPH_Simulation(stage_path=args.stage_path, verbose=args.verbose)
        sim.simulate(args.num_frames)

    elapsed_time = time.time() - starting_time
    print(f"Simulation completed in {elapsed_time:.2f} seconds.")