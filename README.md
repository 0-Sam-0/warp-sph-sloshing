# Fluid sloshing with SPH in NVIDIA Warp

Samuel Seligardi · exploratory work, September–November 2025

Smoothed particle hydrodynamics on the GPU, worked from the SPH example shipped with NVIDIA Warp
towards a concrete question: how a liquid moves inside a rigid container that is accelerated,
carried at constant speed, and then braked. Getting there meant rewriting the example's physics in
real units — the original runs on dimensionless parameters tuned to look right, which is fine for a
demo and useless the moment the motion of the container is prescribed in cm/s².

This repository holds that work as it was left: a CGS-unit SPH solver with a movable container, a
small Open3D pipeline for preparing the container mesh, an abandoned rigid-body attempt, and the
reading notes taken while learning the framework. It is exploratory code, not a finished product —
the section on `rbsimv1.py` says where it stops.

---

## Layout

| Directory | Contents |
|---|---|
| **`scripts/sloshing/`** | the solver, the mesh pipeline, the rigid-body attempt |
| **`assets/`** | the container mesh, as a triangulated PLY export |
| **`info/`** | five reading notes on the official Warp examples |

| File | What it does |
|---|---|
| `sph.py` | the SPH solver — a CGS-unit revision of Warp's SPH example, with a slanted-wall container driven along a prescribed motion profile |
| `mesh_refiner.py` | Open3D utility class: load, clean, subdivide, smooth, measure and export a triangle mesh |
| `script_refiner.py` | the six-line driver that refines `assets/container.ply` for the rigid-body pipeline |
| `rbsimv1.py` | `warp.sim` rigid-body pipeline — mesh as a rigid body falling on a floor. Never worked properly; kept as it was |

---

## `sph.py`

The solver is Müller, Charypar and Gross (2003) as Warp implements it — poly6 kernel for density,
spiky gradient for pressure, viscosity Laplacian, neighbours found through `wp.HashGrid` — with the
numerics rebuilt underneath. Seven changes separate it from the example it came from:

- **Units.** Everything is CGS: centimetres, grams, seconds. Every physical parameter in the
  constructor carries its unit in a comment, because in SPH a dimensionally wrong constant does not
  crash, it quietly produces a plausible-looking fluid that is not water.
- **Particle mass is derived, not chosen.** `m = ρ · dp³` ties mass to the reference density and the
  particle spacing, so changing resolution no longer silently changes the fluid.
- **Timestep from frame rate and substeps.** `step_dt = 1/(fps · substeps)` rather than a free
  number, which makes the substep count the one stability knob.
- **A container with slanted walls**, parameterised by slope: at slope `1e4` the walls are vertical
  to within `y/1e4`, and lowering it gives a tapered tank without touching the collision kernel.
- **A translatable container** — boundaries move as a rigid frame, which is what makes sloshing
  possible at all.
- **A prescribed motion profile** driving the tank: settle, accelerate, coast, brake, rest.
- **Instrumentation**: wall-clock timing and per-frame maximum fluid height.

### Parameters

| Quantity | Value | Unit |
|---|---|---|
| inter-particle distance `dp` | 0.1 | cm |
| smoothing length `h = 1.3·√(3·dp²)` | ≈ 0.225 | cm |
| fluid block (x × y × z) | 12 × 2 × 6.1 | cm |
| particle count | ≈ 146 400 | — |
| reference density | 1.0 | g/cm³ |
| dynamic viscosity | 0.01 | P |
| pressure stiffness `isotropic_exp` | 100 | cm²/s² |
| boundary damping | −0.95 | — |
| gravity | −981 | cm/s² |
| frame rate · substeps | 60 · 420 | Hz · — |
| resulting timestep | ≈ 3.97 × 10⁻⁵ | s |

The hash grid is built with cell size `h`: particles interact only within `h`, so cells of that size
are the natural choice. The bucket count is a separate decision, and the grid has no spatial bounds —
it maps all of ℝ³ into a finite table — so what matters is covering the *extent* the fluid can reach,
not a fixed domain. That extent is the container footprint by a height of three times the initial
fill level, because violent sloshing throws the fluid well above where it starts. Divided into
macro-cells of `4h` it gives 13 × 6 × 6 ≈ 470 buckets, about 64 physical cells each. `sph.py`
documents the trade-off in place.

### Motion profile

| Phase | Interval | Acceleration |
|---|---|---|
| settling | 0.0 – 0.75 s | 0 |
| acceleration | 0.75 – 1.5 s | +160 cm/s² |
| constant velocity | 1.5 – 1.75 s | 0 — peak 120 cm/s |
| braking | 1.75 – 2.25 s | −240 cm/s² |
| rest | > 2.25 s | 0 |

The braking phase is sized to bring the tank exactly to rest — 120 cm/s shed over 0.5 s is
−240 cm/s² — so that past 2.25 s the container is still and whatever the fluid does afterwards is
purely its own inertia.

### Running it

```bash
python scripts/sloshing/sph.py --num_frames 240 --stage_path sph_sim.usd --verbose
```

`--device` overrides the Warp device; `--stage_path None` disables USD output and runs the solver
alone. At the parameters above a 240-frame run is four seconds of simulated time over roughly
100 000 integration steps. Nothing in the code is CUDA-only, so it runs wherever Warp runs — but
146 400 particles at that step count is a GPU proposition.

Requirements: `warp-lang`, `numpy`, and `usd-core` for the renderer.

---

## `assets/` and the mesh pipeline

`container.ply` is the container as the code receives it: a triangulated export of a scaled box,
32 vertices and 60 faces. `script_refiner.py` turns it into the denser mesh the rigid-body pipeline
expects — four midpoint subdivisions, 3586 vertices. That refined mesh is not tracked here, because
`script_refiner.py` regenerates it. The script in full:

```python
from mesh_refiner import MeshRefiner

refiner = MeshRefiner("assets/container.ply")
refiner.visualize_mesh(use_refined=True, show_wireframe=True)
refined = refiner.refine_mesh(subdivision_levels=4, smooth_iterations=0)
refiner.save_mesh("mesh_refined/container_ref.ply")
refiner.visualize_mesh(use_refined=True, show_wireframe=True)
```

The two `visualize_mesh` calls open a blocking Open3D window, so this is an interactive script and
not a headless one. The first of them also runs before `refine_mesh`, so despite `use_refined=True`
there is nothing refined yet and it falls back to showing the original mesh.

`MeshRefiner` wraps the Open3D operations that matter before a mesh reaches a solver — duplicate and
degenerate triangle removal, midpoint or Loop subdivision, simple or Laplacian smoothing, normal
recomputation — and reports what came out: vertex and triangle counts, watertightness, orientability,
surface area, volume, bounding box. Watertightness is the one to watch: a rigid-body or collision
pipeline needs it, and a mesh exported carelessly from Blender usually does not have it. The export
settings that produce a usable PLY are recorded at the bottom of `mesh_refiner.py`. This half of the
repository needs `open3d`, which the solver does not.

The SPH solver does not use these meshes — its container is analytic, defined by the boundary
parameters above. They were prepared for `rbsimv1.py`.

## `rbsimv1.py`

A full `warp.sim` pipeline: load the PLY, align it to the floor, build a rigid body with
triangle-mesh collisions, integrate with `SemiImplicitIntegrator`, export USD. It is parameterised
down to contact stiffness, damping and friction for both body and floor, and it scales up anything
whose smallest dimension falls under a centimetre, below which the default contact settings stop
behaving. Note that it works in SI — metres, kilograms, −9.81 m/s² — unlike the solver next to it.

It never worked properly: the commit that introduced it says so, and nothing in the repository
records why. What is recorded is what happened next — the SPH route with an analytic container
turned out to be the better answer to the original question, and this file was left where it stopped.
It is here as part of that record, not as a working example.

## `info/`

Five notes written while reading the official Warp examples, one per module family — `core`, `fem`,
`optim`, `sim`, `tile`. Each example gets the phenomenon it simulates, the API surface it exercises,
and a short observation. They are what the first weeks of this work produced: a map of what the
framework offers before committing to one corner of it. They were written in Italian at the time and
translated for publication.

---

## Provenance and licence

`scripts/sloshing/sph.py` is a modified version of the SPH example distributed with
[NVIDIA Warp](https://github.com/NVIDIA/warp), Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES,
licensed under the Apache License 2.0. The original copyright header is preserved in the file, and
the point at which it was branched is marked there.

Everything else — `mesh_refiner.py`, `script_refiner.py`, `rbsimv1.py`, the meshes, the notes — is
original work. The whole repository is released under the [Apache License 2.0](LICENSE), the licence
of the code it builds on; see [NOTICE](NOTICE) for the attribution details and the list of
modifications.
