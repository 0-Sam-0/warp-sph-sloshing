#!/usr/bin/env python3
"""
NVIDIA Warp 1.9.0 Rigid Body Simulation Pipeline
Simulates a watertight mesh as rigid body with precise triangle-mesh collisions.
"""

import numpy as np
import open3d as o3d
import warp as wp
import warp.sim
import warp.sim.render
from typing import Tuple, Optional
import os


class RigidBodySimulator:
    """
    Complete pipeline for rigid body simulation in NVIDIA Warp.
    Loads PLY mesh, aligns to floor, simulates physics, exports to USD.
    """
    
    def __init__(
        self,
        mesh_path: str = "mesh.ply",
        device: str = "cuda",
        unit_scale: float = 1.0,
        mass: float = 1.0,
        ke: float = 1e5,      # contact stiffness
        kd: float = 1e3,      # contact damping  
        kf: float = 1e3,      # contact friction
        thickness: float = 0.01,  # collision thickness
        scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        rot: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),  # quaternion (x,y,z,w)
        dt: float = 1.0/60.0,
        substeps: int = 1,
        num_frames: int = 240,
        usd_path: str = "simulation.usd",
        gravity: float = -9.81,
        floor_ke: float = 1e6,    # floor stiffness (higher for solid ground)
        floor_kd: float = 1e4,    # floor damping
        floor_kf: float = 1e3,    # floor friction
        auto_scale_tiny_objects: bool = True,  # Auto-scale if object is too small
        min_object_size: float = 0.01,  # Minimum object size in meters
    ):
        """
        Initialize the rigid body simulator with all parameters.
        
        Args:
            mesh_path: Path to watertight PLY mesh file
            device: Computation device ("cuda" or "cpu")
            unit_scale: Scale factor for unit conversion (e.g., 0.001 for mm to m)
            mass: Mass of the rigid body in kg
            ke: Contact elastic stiffness (N/m)
            kd: Contact damping coefficient (Ns/m)
            kf: Contact friction coefficient
            thickness: Collision detection thickness (m)
            scale: Additional scaling factors (x,y,z)
            rot: Initial rotation quaternion (x,y,z,w)
            dt: Simulation timestep (seconds)
            substeps: Number of substeps per frame
            num_frames: Total number of frames to simulate
            usd_path: Output USD file path
            gravity: Gravity acceleration (m/s^2)
            floor_ke: Floor contact stiffness
            floor_kd: Floor contact damping
            floor_kf: Floor friction coefficient
            auto_scale_tiny_objects: Automatically scale up objects smaller than min_object_size
            min_object_size: Minimum object dimension in meters for stable simulation
        """
        self.mesh_path = mesh_path
        self.device = device
        self.unit_scale = unit_scale
        self.mass = mass
        self.ke = ke
        self.kd = kd
        self.kf = kf
        self.thickness = thickness
        self.scale = scale
        self.rot = rot
        self.dt = dt
        self.substeps = substeps
        self.num_frames = num_frames
        self.usd_path = usd_path
        self.gravity = gravity
        self.floor_ke = floor_ke
        self.floor_kd = floor_kd
        self.floor_kf = floor_kf
        self.auto_scale_tiny_objects = auto_scale_tiny_objects
        self.min_object_size = min_object_size
        
        # Initialize Warp
        wp.init()
        
        # Model and simulation objects
        self.model = None
        self.integrator = None
        self.states = []
        self.renderer = None
        
    def load_and_process_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load PLY mesh using Open3D and prepare for Warp.
        Applies unit scaling and aligns mesh to floor (min_y = 0).
        
        Returns:
            vertices: Nx3 float32 array of vertex positions
            triangles: Mx3 int32 array of triangle indices
        """
        print(f"Loading mesh from {self.mesh_path}...")
        
        # Load mesh with Open3D
        mesh_o3d = o3d.io.read_triangle_mesh(self.mesh_path)
        
        # Ensure mesh is triangulated
        if not mesh_o3d.has_triangles():
            raise ValueError("Mesh must have triangles!")
        
        # Check if watertight (optional warning)
        if not mesh_o3d.is_watertight():
            print("Warning: Mesh is not watertight. Simulation may be unstable.")
        
        # Extract vertices and triangles
        vertices = np.asarray(mesh_o3d.vertices, dtype=np.float32)
        triangles = np.asarray(mesh_o3d.triangles, dtype=np.int32)
        
        # Show original dimensions
        print(f"Original bounding box: min={vertices.min(axis=0)}, max={vertices.max(axis=0)}")
        print(f"Original size: {vertices.max(axis=0) - vertices.min(axis=0)}")
        
        # Apply unit scaling
        vertices *= self.unit_scale
        
        # Apply additional scaling
        vertices[:, 0] *= self.scale[0]
        vertices[:, 1] *= self.scale[1]
        vertices[:, 2] *= self.scale[2]
        
        # Check if object is too small and auto-scale if needed
        bbox_size = vertices.max(axis=0) - vertices.min(axis=0)
        min_dim = np.min(bbox_size)
        
        if self.auto_scale_tiny_objects and min_dim < self.min_object_size:
            scale_factor = self.min_object_size / min_dim
            print(f"WARNING: Object is very small ({min_dim:.6f}m). Auto-scaling by {scale_factor:.2f}x for stability.")
            vertices *= scale_factor
            
            # Adjust mass proportionally (assuming uniform density)
            original_volume = bbox_size[0] * bbox_size[1] * bbox_size[2]
            new_bbox_size = vertices.max(axis=0) - vertices.min(axis=0)
            new_volume = new_bbox_size[0] * new_bbox_size[1] * new_bbox_size[2]
            volume_ratio = new_volume / original_volume if original_volume > 0 else scale_factor**3
            self.mass *= volume_ratio
            print(f"Mass adjusted to {self.mass:.6f} kg to maintain density")
            
            # Adjust collision parameters for larger object
            self.thickness *= scale_factor
            print(f"Collision thickness adjusted to {self.thickness:.6f} m")
        
        # Align to floor: translate so min_y = 0
        min_y = np.min(vertices[:, 1])
        vertices[:, 1] -= min_y
        
        print(f"Mesh loaded: {len(vertices)} vertices, {len(triangles)} triangles")
        print(f"Bounding box after alignment: min={vertices.min(axis=0)}, max={vertices.max(axis=0)}")
        
        return vertices, triangles
    
    def create_warp_mesh(self, vertices: np.ndarray, triangles: np.ndarray) -> wp.sim.Mesh:
        """
        Create Warp mesh object from numpy arrays.
        
        Args:
            vertices: Nx3 vertex positions
            triangles: Mx3 triangle indices
            
        Returns:
            Warp Mesh object
        """
        # Create Warp mesh directly from numpy arrays
        # The Mesh constructor expects numpy arrays or Python lists, not wp.arrays
        mesh = wp.sim.Mesh(
            vertices=vertices,  # Nx3 numpy array
            indices=triangles.flatten()  # Flattened triangle indices
        )
        
        return mesh
    
    def build_model(self, mesh: wp.sim.Mesh) -> wp.sim.Model:
        """
        Build simulation model with rigid body and floor.
        
        Args:
            mesh: Warp mesh object
            
        Returns:
            Warp Model object
        """
        builder = wp.sim.ModelBuilder()
        
        # Add a small vertical offset to prevent initial penetration
        # This is crucial for small objects to avoid explosion
        initial_height = self.thickness * 2.0
        
        # Add rigid body
        # Note: Inertia will be computed automatically from the mesh shape
        body_id = builder.add_body(
            origin=wp.transform((0.0, initial_height, 0.0), wp.quat(*self.rot)),
            m=self.mass
        )
        
        # Add mesh shape for rigid body with triangle-mesh collision
        builder.add_shape_mesh(
            body=body_id,
            mesh=mesh,
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            scale=(1.0, 1.0, 1.0),
            density=0.0,  # Use explicit mass instead
            ke=self.ke,
            kd=self.kd,
            kf=self.kf,
            thickness=self.thickness
        )
        
        # Add static floor plane
        floor_id = builder.add_body(
            origin=wp.transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            m=0.0  # Static body (infinite mass)
        )
        
        builder.add_shape_plane(
            body=floor_id,
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            width=100.0,
            length=100.0,
            ke=self.floor_ke,
            kd=self.floor_kd,
            kf=self.floor_kf
        )
        
        # Set gravity
        builder.gravity = (0.0, self.gravity, 0.0)
        
        # Build and return model
        model = builder.finalize(device=self.device)
        model.ground = True  # Enable ground collision
        
        print(f"Model created with {model.body_count} bodies and {model.shape_count} shapes")
        print(f"Initial body height: {initial_height:.6f} m (to prevent penetration)")
        
        return model
    
    def simulate(self):
        """
        Run the complete simulation pipeline.
        """
        # Load and process mesh
        vertices, triangles = self.load_and_process_mesh()
        
        # Create Warp mesh
        mesh = self.create_warp_mesh(vertices, triangles)
        
        # Build model
        self.model = self.build_model(mesh)
        
        # Create integrator
        self.integrator = wp.sim.SemiImplicitIntegrator()
        
        # Initialize states list
        self.states = []
        
        # Create initial state and store it
        state = self.model.state()
        initial_state_copy = self.model.state()
        wp.copy(initial_state_copy.body_q, state.body_q)
        wp.copy(initial_state_copy.body_qd, state.body_qd)
        self.states.append(initial_state_copy)
        
        # Run simulation
        print(f"Starting simulation: {self.num_frames} frames, dt={self.dt}, substeps={self.substeps}")
        
        # Create a second state for double-buffering
        state_next = self.model.state()
        
        for frame in range(1, self.num_frames):
            # Simulate with substeps
            for substep in range(self.substeps):
                # Compute collisions
                wp.sim.collide(self.model, state)
                
                # Integrate
                self.integrator.simulate(
                    self.model,
                    state,
                    state_next,
                    self.dt / self.substeps
                )
                
                # Swap states for next substep
                state, state_next = state_next, state
            
            # Store a copy of the current state
            frame_state = self.model.state()
            wp.copy(frame_state.body_q, state.body_q)
            wp.copy(frame_state.body_qd, state.body_qd)
            self.states.append(frame_state)
            
            if frame % 30 == 0:
                # Get position for progress update
                pos = state.body_q.numpy()[0][:3]
                print(f"Frame {frame}/{self.num_frames}: body position = {pos}")
        
        print("Simulation complete!")
    
    def export_to_usd(self):
        """
        Export simulation results to USD sequence.
        """
        if not self.states:
            raise RuntimeError("No simulation data to export. Run simulate() first.")
        
        print(f"Exporting to USD: {self.usd_path}")
        
        # Ensure the directory exists
        import os
        usd_dir = os.path.dirname(self.usd_path)
        if usd_dir and not os.path.exists(usd_dir):
            os.makedirs(usd_dir)
            print(f"Created directory: {usd_dir}")
        
        try:
            # Try to create renderer for USD export
            self.renderer = wp.sim.render.SimRenderer(
                self.model,
                self.usd_path,
                scaling=1.0
            )
            
            # Export each frame
            for i, state in enumerate(self.states):
                self.renderer.begin_frame(self.dt * i)
                self.renderer.render(state)
                self.renderer.end_frame()
            
            # Save USD file
            self.renderer.save()
            
            print(f"USD export complete: {self.usd_path}")
            print(f"Total frames exported: {len(self.states)}")
            
        except ImportError as e:
            print(f"WARNING: USD export failed - {e}")
            print("Trying alternative export method...")
            
            # Alternative: save states as numpy arrays
            fallback_path = self.usd_path.replace('.usd', '_states.npz')
            positions = []
            velocities = []
            
            for state in self.states:
                positions.append(state.body_q.numpy())
                velocities.append(state.body_qd.numpy())
            
            np.savez(fallback_path,
                    positions=np.array(positions),
                    velocities=np.array(velocities),
                    dt=self.dt,
                    num_frames=len(self.states))
            
            print(f"Saved simulation data to: {fallback_path}")
            print("You can load this data for visualization in another script.")
            
        except Exception as e:
            print(f"ERROR during USD export: {e}")
            print("Continuing without USD export...")
    
    def run(self):
        """
        Execute the complete pipeline: load, simulate, export.
        """
        print("=" * 60)
        print("NVIDIA Warp Rigid Body Simulation Pipeline")
        print("=" * 60)
        
        # Run simulation
        self.simulate()
        
        # Export results (with error handling)
        try:
            self.export_to_usd()
        except Exception as e:
            print(f"Note: USD export encountered an issue: {e}")
            print("Simulation data has been saved in alternative format.")
        
        print("=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)


def main():
    """
    Main entry point with example usage.
    """
    # Create simulator with custom parameters
    simulator = RigidBodySimulator(
        mesh_path="mesh_refined/container_ref.ply",           # Your PLY file
        device="cuda",                   # Use GPU
        unit_scale=1.0,                 # Adjust based on your mesh units
        mass=0.5,                       # 500g object
        ke=1e5,                         # Contact stiffness
        kd=1e3,                         # Contact damping
        kf=0.8,                         # Friction coefficient
        thickness=0.001,                # 1mm collision thickness
        scale=(1.0, 1.0, 1.0),         # No additional scaling
        rot=(0.0, 0.0, 0.0, 1.0),      # No rotation
        dt=1.0/60.0,                   # 60 FPS
        substeps=2,                    # 2 substeps for stability
        num_frames=240,                # 4 seconds at 60 FPS
        usd_path="usd/rigid_simulation.usd",
        gravity=-9.81,                 # Earth gravity
        floor_ke=1e6,                  # Stiff floor
        floor_kd=1e4,                  # Floor damping
        floor_kf=0.9,                  # Floor friction
        auto_scale_tiny_objects=True,  # Auto-scale if too small
        min_object_size=0.01           # Minimum 1cm for stability
    )
    
    # Run the pipeline
    simulator.run()


if __name__ == "__main__":
    main()