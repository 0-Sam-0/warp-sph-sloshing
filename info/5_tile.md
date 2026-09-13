# Examples/tile

## mlp - Coordinate-Based Neural Network for Image Representation

**What it simulates:**
Trains a coordinate-based multilayer perceptron to predict RGB colours at given input positions, using positional encoding to improve the representation of high-frequency content. Compares the Warp and PyTorch implementations.

**Warp features used:**
• `wp.Tape()` automatic differentiation - Native GPU backpropagation through the network layers
• `warp.optim.Adam` - Built-in Adam optimiser for GPU-optimised parameter updates
• `wp.tile_*` operations - Use of tensor cores for high-performance matrix operations
• CUDA graph capture - Pre-compilation of training epochs to eliminate CPU overhead
• `wp.float16` precision - Improved memory efficiency and throughput on modern hardware
• Block-level parallelisation - Automatic thread block handling for batch processing

**Key capability:**
Warp competes directly with PyTorch on modern ML workloads while integrating seamlessly with physics simulation. Tile operations exploit specialised hardware, reaching performance comparable to dedicated ML frameworks.

## nbody - Gravitational N-Body Simulation with Tile Primitives

**What it simulates:**
Simulates a gravitational N-Body problem using an all-pairs approach built on Warp's tile primitives. Particles start distributed over a sphere with initial velocities that induce rotation, simulating the formation of gravitational structures.

**Warp features used:**
• `wp.tile_load()` - Efficient loading of data into tiles to exploit GPU shared memory
• `wp.constant()` - Compile-time constants for physical parameters (DT, SOFTENING_SQ, TILE_SIZE)
• Block-level parallelisation with `block_dim=TILE_SIZE` - Memory access optimisation through tiling
• `body_body_interaction()` - Function computing the gravitational force between particle pairs with softening
• All-pairs interaction pattern - Every particle interacts with every other one for an accurate simulation
• Array swapping - Double buffering for position updates without race conditions

**Key capability:**
Warp handles tile operations automatically to exploit modern GPU architectures with shared memory. The tiled approach scales N-Body simulations to tens of thousands of particles while keeping physical accuracy and real-time performance for astrophysical applications.

## walker - Training Quadruped Soft-Body Locomotion with Neural Networks

**What it simulates:**
Trains a tetrahedral mesh quadruped to walk through a fully-connected neural network that converts 8 sinusoidal time phases into tetrahedral activations. The soft-body model is simulated forward and evaluated on the momentum of its centre of mass.

**Warp features used:**
• `wp.sim` soft-body physics - Deformable tetrahedra simulation with contact handling and ground interaction
• `wp.launch_tiled()` with tile operations - Matrix multiplication optimised via tensor cores for the neural network
• `warp.optim.Adam` - Built-in optimiser for training the network parameters  
• `wp.Tape()` + CUDA graph capture - Automatic differentiation through the physics simulation with pre-compilation
• Tetrahedron activation control - Neural outputs read as muscle activations for soft-body locomotion

**Key capability:**
Warp unifies machine learning with complex physics simulation seamlessly, allowing end-to-end training of soft robotic systems through automatic differentiation. The framework propagates gradients through physics engines automatically, for advanced neuro-mechanical applications.

---
