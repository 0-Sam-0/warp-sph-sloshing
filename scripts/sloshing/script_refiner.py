from mesh_refiner import MeshRefiner

refiner = MeshRefiner("assets/container.ply")
refiner.visualize_mesh(use_refined=True, show_wireframe=True)
refined = refiner.refine_mesh(subdivision_levels=4, smooth_iterations=0)
refiner.save_mesh("mesh_refined/container_ref.ply")
refiner.visualize_mesh(use_refined=True, show_wireframe=True)