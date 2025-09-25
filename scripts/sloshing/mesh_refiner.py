"""
Mesh Refiner Utility Class

A focused utility class for loading, refining, and processing mesh data using Open3D.
Designed for mesh preprocessing workflows before physics simulation or other applications.
"""

import numpy as np
import open3d as o3d
from typing import Optional, Dict, Any
from pathlib import Path


class MeshRefinerError(Exception):
    """Custom exception for MeshRefiner errors"""
    pass


class MeshRefiner:
    """
    A utility class for mesh refinement and processing.
    
    This class provides functionality to:
    - Load mesh data from various file formats (PLY, OBJ, STL, etc.)
    - Apply subdivision and smoothing refinements
    - Preprocess meshes (cleanup operations)
    - Save processed meshes
    - Extract mesh statistics and properties
    """
    
    def __init__(self, mesh_path: Optional[str] = None):
        """
        Initialize the mesh refiner.
        
        Args:
            mesh_path: Optional path to mesh file to load immediately
        """
        self._mesh_path: Optional[Path] = Path(mesh_path) if mesh_path else None
        self._original_mesh: Optional[o3d.geometry.TriangleMesh] = None
        self._refined_mesh: Optional[o3d.geometry.TriangleMesh] = None
        
        # Auto-load if path provided
        if self._mesh_path:
            self.load_mesh()
    
    @property
    def original_mesh(self) -> Optional[o3d.geometry.TriangleMesh]:
        """Get the original mesh"""
        return self._original_mesh
    
    @property
    def refined_mesh(self) -> Optional[o3d.geometry.TriangleMesh]:
        """Get the refined mesh"""
        return self._refined_mesh
    
    @property
    def has_original(self) -> bool:
        """Check if original mesh is loaded"""
        return self._original_mesh is not None
    
    @property
    def has_refined(self) -> bool:
        """Check if refined mesh is available"""
        return self._refined_mesh is not None
    
    def load_mesh(self, mesh_path: Optional[str] = None) -> o3d.geometry.TriangleMesh:
        """
        Load mesh from file using Open3D.
        
        Supports common formats: PLY, OBJ, STL, OFF, GLTF
        
        Args:
            mesh_path: Path to mesh file. If None, uses the path from initialization.
            
        Returns:
            The loaded mesh
            
        Raises:
            MeshRefinerError: If mesh loading fails or mesh is invalid
        """
        if mesh_path:
            self._mesh_path = Path(mesh_path)
        
        if not self._mesh_path:
            raise MeshRefinerError("No mesh path specified")
        
        if not self._mesh_path.exists():
            raise MeshRefinerError(f"Mesh file not found: {self._mesh_path}")
        
        print(f"Loading mesh from: {self._mesh_path}")
        
        self._original_mesh = o3d.io.read_triangle_mesh(str(self._mesh_path))
        
        if not self._original_mesh.has_vertices():
            raise MeshRefinerError("Failed to load mesh or mesh has no vertices")
        
        # Basic mesh validation
        vertices_count = len(self._original_mesh.vertices)
        triangles_count = len(self._original_mesh.triangles)
        
        if vertices_count == 0:
            raise MeshRefinerError("Mesh has no vertices")
        
        print(f"Loaded mesh - Vertices: {vertices_count}, Triangles: {triangles_count}")
        
        return self._original_mesh
    
    def preprocess_mesh(self, 
                       remove_duplicates: bool = True,
                       remove_degenerate: bool = True,
                       compute_normals: bool = True) -> o3d.geometry.TriangleMesh:
        """
        Apply basic preprocessing to the mesh (cleanup operations).
        
        Args:
            remove_duplicates: Remove duplicate vertices and triangles
            remove_degenerate: Remove degenerate triangles
            compute_normals: Compute vertex normals if missing
            
        Returns:
            The preprocessed mesh
            
        Raises:
            MeshRefinerError: If no mesh is loaded
        """
        if not self._original_mesh:
            raise MeshRefinerError("No mesh loaded. Call load_mesh() first.")
        
        # Work on a copy
        mesh = o3d.geometry.TriangleMesh(self._original_mesh)
        
        print("Preprocessing mesh...")
        
        if remove_duplicates:
            print("  Removing duplicated vertices and triangles")
            mesh.remove_duplicated_vertices()
            mesh.remove_duplicated_triangles()
        
        if remove_degenerate:
            print("  Removing degenerate triangles")
            mesh.remove_degenerate_triangles()
        
        if compute_normals and not mesh.has_vertex_normals():
            print("  Computing vertex normals")
            mesh.compute_vertex_normals()
        
        return mesh
    
    def refine_mesh(self, 
                   subdivision_levels: int = 1, 
                   smooth_iterations: int = 5,
                   subdivision_method: str = "midpoint",
                   smooth_method: str = "simple") -> o3d.geometry.TriangleMesh:
        """
        Refine the mesh using subdivision and smoothing.
        
        Args:
            subdivision_levels: Number of subdivision iterations
            smooth_iterations: Number of smoothing iterations  
            subdivision_method: Type of subdivision ("midpoint" or "loop")
            smooth_method: Type of smoothing ("simple" or "laplacian")
            
        Returns:
            The refined mesh
            
        Raises:
            MeshRefinerError: If no mesh is loaded or parameters are invalid
        """
        if not self._original_mesh:
            raise MeshRefinerError("No mesh loaded. Call load_mesh() first.")
        
        if subdivision_levels < 0:
            raise MeshRefinerError("Subdivision levels must be non-negative")
        
        if smooth_iterations < 0:
            raise MeshRefinerError("Smooth iterations must be non-negative")
        
        valid_subdivision = ["midpoint", "loop"]
        if subdivision_method not in valid_subdivision:
            raise MeshRefinerError(f"Invalid subdivision method. Must be one of: {valid_subdivision}")
        
        valid_smoothing = ["simple", "laplacian"]
        if smooth_method not in valid_smoothing:
            raise MeshRefinerError(f"Invalid smooth method. Must be one of: {valid_smoothing}")
        
        # Start with preprocessed mesh
        self._refined_mesh = self.preprocess_mesh()
        
        # Apply subdivision refinement
        if subdivision_levels > 0:
            print(f"Applying {subdivision_levels} subdivision levels using {subdivision_method} method")
            for i in range(subdivision_levels):
                print(f"  Subdivision level {i+1}/{subdivision_levels}")
                if subdivision_method == "midpoint":
                    self._refined_mesh = self._refined_mesh.subdivide_midpoint(number_of_iterations=1)
                elif subdivision_method == "loop":
                    self._refined_mesh = self._refined_mesh.subdivide_loop(number_of_iterations=1)
        
        # Apply smoothing
        if smooth_iterations > 0:
            print(f"Applying {smooth_iterations} smoothing iterations using {smooth_method} method")
            if smooth_method == "simple":
                self._refined_mesh = self._refined_mesh.filter_smooth_simple(
                    number_of_iterations=smooth_iterations
                )
            elif smooth_method == "laplacian":
                self._refined_mesh = self._refined_mesh.filter_smooth_laplacian(
                    number_of_iterations=smooth_iterations
                )
        
        # Recompute normals after refinement
        self._refined_mesh.compute_vertex_normals()
        
        vertices_count = len(self._refined_mesh.vertices)
        triangles_count = len(self._refined_mesh.triangles)
        
        print(f"Refined mesh - Vertices: {vertices_count}, Triangles: {triangles_count}")
        
        return self._refined_mesh
    
    def get_vertices_array(self, use_refined: bool = True) -> np.ndarray:
        """
        Get vertices as numpy array.
        
        Args:
            use_refined: Use refined mesh if available, otherwise original
            
        Returns:
            Vertices as numpy array (N, 3)
            
        Raises:
            MeshRefinerError: If no mesh is available
        """
        mesh = self._get_mesh_to_use(use_refined)
        return np.asarray(mesh.vertices, dtype=np.float32)
    
    def get_triangles_array(self, use_refined: bool = True) -> np.ndarray:
        """
        Get triangles as numpy array.
        
        Args:
            use_refined: Use refined mesh if available, otherwise original
            
        Returns:
            Triangles as numpy array (M, 3)
            
        Raises:
            MeshRefinerError: If no mesh is available
        """
        mesh = self._get_mesh_to_use(use_refined)
        return np.asarray(mesh.triangles, dtype=np.int32)
    
    def get_normals_array(self, use_refined: bool = True) -> Optional[np.ndarray]:
        """
        Get vertex normals as numpy array.
        
        Args:
            use_refined: Use refined mesh if available, otherwise original
            
        Returns:
            Vertex normals as numpy array (N, 3) or None if not available
            
        Raises:
            MeshRefinerError: If no mesh is available
        """
        mesh = self._get_mesh_to_use(use_refined)
        if mesh.has_vertex_normals():
            return np.asarray(mesh.vertex_normals, dtype=np.float32)
        return None
    
    def _get_mesh_to_use(self, use_refined: bool) -> o3d.geometry.TriangleMesh:
        """Helper method to determine which mesh to use"""
        if use_refined and self._refined_mesh is not None:
            return self._refined_mesh
        elif self._original_mesh is not None:
            return self._original_mesh
        else:
            raise MeshRefinerError("No mesh available.")
    
    def save_mesh(self, output_path: str, use_refined: bool = True) -> bool:
        """
        Save mesh to file.
        
        Args:
            output_path: Path where to save the mesh
            use_refined: Save refined mesh if available, otherwise original
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            MeshRefinerError: If no mesh is available to save
        """
        mesh_to_save = self._get_mesh_to_use(use_refined)
        mesh_type = "refined" if (use_refined and self._refined_mesh is not None) else "original"
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        success = o3d.io.write_triangle_mesh(str(output_path), mesh_to_save)
        
        if success:
            print(f"Saved {mesh_type} mesh to: {output_path}")
        else:
            print(f"Failed to save mesh to: {output_path}")
        
        return success
    
    def get_mesh_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive statistics about the meshes.
        
        Returns:
            Dictionary with statistics for available meshes
        """
        stats = {}
        
        if self._original_mesh is not None:
            stats['original'] = self._compute_mesh_stats(self._original_mesh)
        
        if self._refined_mesh is not None:
            stats['refined'] = self._compute_mesh_stats(self._refined_mesh)
            
            # Calculate refinement ratio
            if self._original_mesh is not None:
                orig_vertices = len(self._original_mesh.vertices)
                refined_vertices = len(self._refined_mesh.vertices)
                stats['refinement_ratio'] = {
                    'vertices': refined_vertices / orig_vertices,
                    'triangles': len(self._refined_mesh.triangles) / len(self._original_mesh.triangles)
                }
        
        return stats
    
    def _compute_mesh_stats(self, mesh: o3d.geometry.TriangleMesh) -> Dict[str, Any]:
        """Compute statistics for a single mesh"""
        return {
            'vertices': len(mesh.vertices),
            'triangles': len(mesh.triangles),
            'has_normals': mesh.has_vertex_normals(),
            'has_colors': mesh.has_vertex_colors(),
            'is_watertight': mesh.is_watertight(),
            'is_orientable': mesh.is_orientable(),
            'surface_area': mesh.get_surface_area(),
            'volume': mesh.get_volume() if mesh.is_watertight() else None,
            'bounding_box': {
                'min': np.asarray(mesh.get_min_bound()).tolist(),
                'max': np.asarray(mesh.get_max_bound()).tolist(),
                'center': np.asarray(mesh.get_center()).tolist()
            }
        }
    
    def visualize_mesh(self, use_refined: bool = True, show_wireframe: bool = False):
        """
        Visualize the mesh using Open3D viewer.
        
        Args:
            use_refined: Show refined mesh if available, otherwise original
            show_wireframe: Show mesh as wireframe
        """
        mesh_to_show = self._get_mesh_to_use(use_refined)
        mesh_type = "refined" if (use_refined and self._refined_mesh is not None) else "original"
        
        print(f"Visualizing {mesh_type} mesh...")
        
        # Create visualization
        vis_mesh = o3d.geometry.TriangleMesh(mesh_to_show)
        
        if show_wireframe:
            # Convert to wireframe
            wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(vis_mesh)
            o3d.visualization.draw_geometries([wireframe])
        else:
            # Ensure mesh has colors for better visualization
            if not vis_mesh.has_vertex_colors():
                vis_mesh.paint_uniform_color([0.7, 0.7, 0.7])
            o3d.visualization.draw_geometries([vis_mesh])
    
    def reset(self):
        """Reset the refiner, clearing all loaded meshes."""
        self._original_mesh = None
        self._refined_mesh = None
        print("MeshRefiner reset - all meshes cleared")
    
    def __str__(self) -> str:
        """String representation of the MeshRefiner."""
        if not self._original_mesh:
            return "MeshRefiner(no mesh loaded)"
        
        orig_v = len(self._original_mesh.vertices)
        orig_t = len(self._original_mesh.triangles)
        
        if self._refined_mesh:
            ref_v = len(self._refined_mesh.vertices)
            ref_t = len(self._refined_mesh.triangles)
            return f"MeshRefiner(original: {orig_v}v/{orig_t}t, refined: {ref_v}v/{ref_t}t)"
        else:
            return f"MeshRefiner(original: {orig_v}v/{orig_t}t)"
        


"""
From Blender:
- edit mode -> select all faces -> ctrl+T (to triangulate)
- select all
- file -> export -> ply:
  𐄂 ASCII
  ✓ Selectin Only
  Scale 1.000
  Forward Axis -Z
  Up Axis Y

  𐄂 UV Coordinates
  ✓ Vertex Normals
  ✓ Vertex Attributes
  Vertex Colors sRGB
  ✓ Triangulated Mesh
  𐄂 Apply Modifiers

Use:
from mesh_refiner import MeshRefiner
refiner = MeshRefiner("path/to/model.ply")
refined = refiner.refine_mesh(subdivision_levels=7, smooth_iterations=10)
refiner.save_mesh("output/refined_model.ply")
refiner.visualize_mesh(use_refined=True, show_wireframe=False)
"""