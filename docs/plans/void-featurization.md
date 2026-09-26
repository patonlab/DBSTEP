# Featurizing voids: pockets and active sites as vectors and tensors

Status: design notes, no code yet. Companion to `2.0-proteins-and-trajectories.md`, which
delivered the machinery this builds on (PDB parsing, residue selection, radial crop, occupancy
tensors, per-residue decomposition, frames, Boltzmann ensembles).

## 1. The idea

DBSTEP measures the space **occupied** around a reference atom. A pocket is the space **left
empty** around a point. Everything downstream of the occupancy grid works the same on the
complement of that grid, so a void featurizer is mostly a question of three things a
substituent gets for free and a void does not:

| A substituent has | A void needs | Proposed source |
|---|---|---|
| a reference atom (atom1) | a centre | ligand centroid, residue-set centroid, explicit xyz, refined by the distance transform |
| its own atoms bound the measurement | a boundary that stops at the mouth | sphere plus flood fill, or morphological closing |
| an axis (atom1 to atom2) | an orientation | centre-to-mouth direction from ray casting, principal axis for the roll |

With those in place the existing grid Sterimol, buried-volume, scan, decomposition and frame
loops can be run on void voxels instead of occupied ones, and the scalar descriptors fall out.
The more interesting outputs, and the focus of this note, are **vectors and tensors** that
describe the shape of the void as a whole.

## 2. Centre, boundary, orientation

### Centre

- **Ligand:** centroid of a HETATM residue, run with the ligand excluded (`--exclude-self`
  semantics). Bound ligands mark real pockets and provide a validation target (section 6).
- **Residue set:** centroid of chosen atoms, e.g. the side chains of a catalytic triad
  (`--residue A:57,A:102,A:195`).
- **Explicit:** `--center x,y,z` for docking boxes and literature sites.
- **Refinement:** the Euclidean distance transform of the occupancy grid
  (`scipy.ndimage.distance_transform_edt`) gives every empty voxel its distance to the nearest
  atom surface, i.e. the radius of the largest sphere that fits there. The local maximum near
  the seed is the deepest point of the pocket; its value is the inscribed radius. This makes the
  centre robust to where the user clicked, and it is one call on a grid we already build.

### Boundary

- **Sphere-bounded:** empty voxels within radius R that are connected to the centre by a flood
  fill through empty space (`scipy.ndimage.label`). Disconnected gaps behind residues are
  excluded, so this is the *connected-pocket* volume. It is smaller than or equal to the plain
  free volume, the complement of what `--exclude-self --vbur` reports today, which counts every
  empty voxel in the sphere including gaps the centre cannot reach. Report both: their difference
  is itself informative (buried voids near the site).
- **Morphologically closed (fpocket-like):** dilate the occupancy with a large probe (about
  8 Å), erode back, which seals the mouth; then flood-fill from the centre inside the sealed
  envelope. Pocket volume becomes independent of R.
- **Probe radius:** eroding the void by 1.4 Å turns the bare void into a water-accessible void.
  Sterimol-style comparisons want the bare surface, binding questions want the accessible one;
  make it an option and default to bare so results line up with the rest of DBSTEP.

### Orientation

Cast a few hundred rays from the centre (Fibonacci sphere) through the occupancy grid and record
the distance to the first wall along each. Rays that escape within a cut-off define the mouth;
their mean direction is the axis. The fraction of blocked rays is the buriedness. With the axis
fixed, the roll is set by the principal axis of the void's cross-section, or left free for the
rotation-invariant representations below, which do not need it.

## 3. Vector and tensor representations

Three families, distinguished by what they are invariant to and what they can be compared
against.

### 3.1 Aligned tensors (equivariant, frame-dependent)

`--tensor` already returns a binary occupancy grid in a frame set by atom1, atom2 and atom3.
For a void, the extension is the complement within the sphere, aligned by the mouth axis and the
principal cross-section axis. With channels it becomes a pocket image:

- channel 0: free space
- channels 1..k: walls split by residue class (hydrophobic, polar, charged, H-bond donor,
  acceptor) or by element

Shape `C × N × N × N`; at 1 Å spacing roughly `5 × 15³`. This is the input 3D convolutional
models expect. Weaknesses: everything depends on the alignment, two pockets with poorly defined
mouth axes will not overlay, and voxel grids are large and not directly comparable by distance.

### 3.2 Rotation-invariant expansions (fixed length, alignment-free)

These are the ones to build first, because they can be compared across unrelated proteins
without any alignment and can be averaged over frames or Boltzmann ensembles.

- **Radial profiles.** Free volume per shell as a function of R is the void's Vol2Vec; the
  shell-occupancy scan is the wall profile. Both exist in spirit today.
- **Angular depth map.** The ray-cast distances form a function on the sphere. Expanded in
  spherical harmonics to degree 6 to 8 and reduced to the power spectrum, it gives about ten
  numbers that are exactly rotation invariant: degree 0 is mean depth, degree 2 elongation,
  higher degrees lobes and sub-pockets. The bispectrum keeps more shape at a few dozen numbers.
- **Radial × angular power spectrum.** Expand the free-space field in a radial basis times
  spherical harmonics and keep the power spectrum. This is the SOAP construction applied to an
  occupancy field instead of atom densities: smooth, invariant, and it captures how the shape
  changes with depth. A few hundred numbers, tunable through the number of radial functions and
  the maximum degree.
- **3D Zernike moments** of the void voxels: the shape-retrieval standard, rotation invariant,
  scale normalisable, reconstructable. Order 20 gives about 120 invariants. Best choice for
  pocket similarity search.

### 3.3 Interpretable matrices along the mouth axis

Slice the void along the axis and record per slice: free area, the narrowest and widest
cross-sections from the inverse grid Sterimol (B1 and B5 of the void), and the offset of the
slice centroid from the axis (tortuosity). A `slices × 4` matrix, the void's Sterimol2Vec. It
tells a funnel from a tunnel from a pit in a way a chemist can read. It depends on the axis but
not on the roll, so it is far more robust than a voxel grid.

### 3.4 Summary

| Layer | Output | Invariance | Best for |
|---|---|---|---|
| Free-space tensor with channels | `C × N × N × N` | none (aligned frame) | 3D CNNs, visualisation |
| Void Sterimol2Vec | `slices × 4` | axis only | interpretation, funnel vs tunnel vs pit |
| Depth-map power spectrum | ~10 numbers | full rotation | quick comparisons, regression |
| Radial × angular power spectrum | a few hundred numbers | full rotation | general ML features |
| Zernike invariants | ~120 numbers | rotation and scale | similarity search |

The scalar descriptors are projections of these layers: buriedness is the fraction of the depth
map beyond a threshold, inscribed radius is its minimum, depth and widths are extremes of the
slice matrix, lining residues come from the existing `--decompose` applied to the walls. The
vector layer subsumes the scalar layer.

## 4. Design points

- **Comparability needs a shared normalisation.** Fix R, grid spacing and probe radius per
  study; normalise free volume by the sphere volume; for Zernike normalise the scale explicitly
  or pocket size is counted twice.
- **Grid resolution sets the frequency ceiling.** Degree 8 harmonics at 3.5 Å resolve features
  of about 1.4 Å, which matches a 0.5 Å grid. Finer grids add nothing to a low-degree expansion,
  so the void layer can run coarser than the volume layer.
- **Trajectories multiply naturally.** Any layer per frame gives a time series of vectors; the
  invariant layers can be averaged over frames or over a Boltzmann ensemble without alignment,
  which the aligned tensor cannot.
- **Centre sensitivity.** Report features at the refined centre and, optionally, their spread
  over a small jitter, so a user knows whether the pocket is well defined.
- **Hydrogens.** Crystal structures lack them and a pocket looks roomier without them. Protonate
  first, or use `--sambvca` consistently and say so.
- **Waters and ions** fill the pocket being measured. `--nowater` and `--nohet` exist; a void run
  should default to both and say so.
- **Cost.** Everything runs on the cropped grid: same order as a residue run. Ray casting at
  500 directions and a distance transform on a 300³ box are each well under a second.

## 5. Where it sits in the code

Existing: occupancy mask (`sterics.occupied_direct`, `return_mask=True`), radial crop, residue
metadata and filters, `--tensor` frame handling, `--decompose`, the frame loop, Boltzmann
weighting.

New, all numpy/scipy with no new dependencies:

| Piece | Module | Notes |
|---|---|---|
| centre options and distance-transform refinement | `selection.py` | `--center`, ligand/residue centroid, `refine=True` |
| void extraction (flood fill, closing, probe erosion) | new `void.py` | returns a boolean void mask on the crop lattice |
| ray casting: depth map, mouth axis, buriedness | `void.py` | Fibonacci directions, march along the grid |
| spherical-harmonic utilities and power spectra | new `shapes.py` | `scipy.special.sph_harm`; radial basis (Gaussians or polynomials) |
| inverse grid Sterimol and slice matrix | `sterics.py` | reuse `get_cube_sterimol` on void voxels |
| Zernike invariants (optional, later) | `shapes.py` | standard recurrence; validate against a sphere and a cylinder |
| outputs | `writer.py` | `.npy` for tensors and vectors, one CSV row of invariants per run for `--csv` |

Suggested CLI surface: `--void` switches to void mode around the chosen centre; `--void-features
tensor,slices,depth,radial,zernike` picks layers; `--probe 1.4`; `--close 8.0` for the
morphological boundary; `--save` writes arrays next to the input as today for tensors.

## 6. Validation

**Synthetic cavities with exact answers**

- Hollow spherical shell of carbon atoms: known inscribed radius and free volume; depth map has
  only degree 0; all higher power-spectrum terms vanish to grid tolerance.
- Cylinder of atoms with one open end: known depth and width; slice matrix constant along the
  axis; depth map has only even degrees.
- Cone: slice widths decrease linearly; distinguishes funnel from tunnel.
- Two overlapping spherical cavities: the flood fill must return both when connected and one
  when a wall is inserted.

**Invariance**

- Rotate a structure by a random matrix and recompute: invariant vectors match to grid
  tolerance; aligned outputs match after applying the same rotation.
- Translate: everything identical (already tested for %V_bur on the trajectory fixture).

**Real structures**

- The bound ligand must fit: its vdW volume is smaller than the closed pocket volume, and its
  own L, B1 and B5 along the mouth axis are smaller than the void's. Any violation means the
  boundary or the axis is wrong.
- Compare pocket volumes and buriedness against fpocket or POVME on a handful of enzymes; expect
  agreement in ranking rather than in absolute numbers, since the definitions differ.
- Ensemble sanity: on the penta-alanine trajectory fixture the void around residue 3 shrinks
  monotonically as the water moves in.

## 7. Suggested order of work

1. Centre options and the distance-transform refinement. Small, and it makes `--exclude-self`
   immediately useful for pockets.
2. Void extraction with the sphere boundary and flood fill; free-volume profile as the first
   vector; ray casting for buriedness and the mouth axis.
3. Depth-map and radial × angular power spectra with the synthetic-cavity tests.
4. Inverse grid Sterimol and the slice matrix.
5. Free-space tensor with residue-class channels, reusing the `--tensor` frame code.
6. Morphological closing and probe erosion as options.
7. Zernike invariants, if similarity search becomes a use case.

Each step is testable on the synthetic cavities before touching a real enzyme, and each adds a
column or an array to the existing `results`/`--csv`/`--save` outputs rather than a new output
path.
