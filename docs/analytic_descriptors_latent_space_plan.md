# **State Representation Plan: Task-Aligned Latent Vector ($z$)**

This document outlines the structure of a task-aligned, low-dimensional latent state representation vector $z$. It is designed for Model Predictive Control (MPC) and transition learning in a fenced workspace.  
By applying an optional **affine transformation** (translating and rotating the scene relative to the tool's starting pose), we reduce the action space to a single scalar (push length). The plan below describes how each component of the vector behaves both with and without this transformation.

## **1\. Optional Scene Pre-Processing: Tool-Relative Affine Warp**

Before computing the descriptors, we can optionally warp the global Eulerian density field $\\rho\_{\\text{global}}$ into a tool-relative frame $\\rho\_{\\text{tool\\\_relative}}$.

* **Translation:** Shifts the coordinate origin $(0,0)$ to the tool's starting contact point.  
* **Rotation:** Rotates the grid so that the tool's push vector points strictly along the positive $X$-axis.

## **2\. Structured Latent Vector ($z$) Layout**

The embedding vector $z$ is a concatenated 1D tensor of size **104** (or more, depending on polar/projection resolution).

| Section | Feature Name | Dim | Description | Frame Dependence |
| :---- | :---- | :---- | :---- | :---- |
| **A** | Mass & Global Statistics | 5 | Total volume, center of mass, and spatial dispersion. | Coordinate-dependent |
| **B** | Ellipsoidal Geometry | 3 | Principal axes of the distribution and orientation. | Coordinate-dependent |
| **C** | Invariant Shape Signatures | 9 | Scale/rotation invariant descriptors (Hu moments, compactness). | **Invariant** |
| **D** | Directional Projections (Radon) | 32 | Low-dimensional mass "shadows" along key angles. | Coordinate-dependent |
| **E** | Polar Density Profiles | 32 | Concentric ring and wedge mass distributions. | Coordinate-dependent |
| **F** | Environmental & Wall Constraints | 23 | Proximity and interaction with the physical boundary walls. | Coordinate-dependent |

## **3\. Detailed Component Description**

### **Section A: Mass & Global Statistics (5 Dimensions)**

This section captures the bulk presence of the material and its coarse location.

* **Total Mass ($1$ Dim):** Sum of all pixel values in the density grid.  
  * *Behavior:* Globally invariant. Acts as a vital anchor for volume conservation.  
* **Center of Mass (CoM) ($2$ Dim):** The density-weighted centroid $(x\_{\\text{com}}, y\_{\\text{com}})$.  
  * *Standard Frame:* Absolute coordinates in the workspace.  
  * *Tool-Relative Frame:* Relative $(x', y')$ offset from the tool's face. $y' \\approx 0$ means the pile is perfectly centered in front of the tool.  
* **Spatial Variance / Spread ($2$ Dim):** The standard deviation of the density distribution along both axes.  
  * *Standard Frame:* Spread along global axes.  
  * *Tool-Relative Frame:* Spread parallel and perpendicular to the face of the tool.

### **Section B: Ellipsoidal Geometry (3 Dimensions)**

Derived from the eigenvalues and eigenvectors of the spatial covariance matrix of the pile.

* **Principal Eigenvalues ($\\lambda\_1, \\lambda\_2$) ($2$ Dim):** The length of the major and minor axes of the best-fit ellipse representing the pile.  
  * *Behavior:* Invariant. Captures whether the pile is a circular blob or a sheared, elongated line.  
* **Relative Orientation ($\\theta\_{\\text{rel}}$) ($1$ Dim):** The angle of the major principal axis.  
  * *Standard Frame:* Absolute angle in the workspace.  
  * *Tool-Relative Frame:* The orientation relative to the tool face. Directly predicts whether a push will result in a symmetric plow or cause the pile to slip off to the left or right.

### **Section C: Invariant Shape Signatures (9 Dimensions)**

These features analyze the topology and structure of the pile. Because they are completely invariant to scale, translation, and rotation, they remain identical regardless of whether the affine warp is applied.

* **Hu Moments ($7$ Dim):** Orthogonal moment combinations that provide a scale-and-rotation-free signature of the shape. Excellent for classifying target letters or distinct shapes.  
* **Solidity ($1$ Dim):** Ratio of the pile's area to its convex hull area. Captures whether the pile is a single dense cluster or has sparse gaps/pockets.  
* **Compactness ($1$ Dim):** Calculated as $4\\pi \\cdot \\text{Area} / \\text{Perimeter}^2$. Quantifies how closely the pile resembles a perfect circle.

### **Section D: Directional Projections (32 Dimensions)**

Acts as a 1D "shadow projection" of the pile, which translates non-linear 2D deformations into linear, 1D translations.

* **Projections ($32$ Dim):** Integrated density along 4 discrete angles ($0^\\circ$, $45^\\circ$, $90^\\circ$, $135^\\circ$), binned into 8 values per projection.  
  * *Standard Frame:* Projections relative to the room walls.  
  * *Tool-Relative Frame:*  
    * The $0^\\circ$ projection aligns with the push axis, mapping how much material is "piled up" directly in front of the shovel.  
    * The $90^\\circ$ projection aligns perpendicular to the push, mapping how far the material is spilling out to the sides.

### **Section E: Polar Density Profiles (32 Dimensions)**

Splits the distribution into polar coordinates to capture how the pile is distributed radially and angularly.

* **Radial Profile ($16$ Dim):** Average density calculated across 16 concentric rings.  
  * *Origin:* \* *Standard Frame:* Centered at the pile's CoM.  
    * *Tool-Relative Frame:* Centered at the **tool's starting contact point**, directly measuring how far away the mass sits from the tool.  
* **Angular Profile ($16$ Dim):** Average density calculated across 16 wedge-shaped sectors.  
  * *Origin:* \* *Standard Frame:* Centered at the pile's CoM.  
    * *Tool-Relative Frame:* Centered at the **tool's starting contact point**, capturing the angular spread of the material radiating away from the shovel's path.

### **Section F: Environmental & Wall Constraints (23 Dimensions)**

Essential for modeling the boundaries in your fenced workspace, where physical dynamics change abruptly due to collisions.

* **Boundary Distance Projections ($16$ Dim):** The density field element-wise multiplied by a static 2D Signed Distance Field (SDF) of the fence, binned into a 16-dimensional spatial profile.  
  * *Behavior:* Measures how much of the pile's mass is actively pressed against which walls.  
* **Min Distance to Walls ($4$ Dim):** A 4-element vector representing the distance from the pile's closest edge to the North, South, East, and West walls.  
* **Topological Euler Characteristic ($1$ Dim):** A single integer representing (Connected Components $-$ Holes).  
  * *Behavior:* Invariant. Instantly flags discrete events like a single pile splitting into two, or a pile being squeezed flat against a corner and losing its shape.  
* **Shannon Entropy ($2$ Dim):** A metric tracking the global dispersion of the pile's mass. One dimension tracks the global entropy; the second tracks the entropy specifically inside the local neighborhood of the tool's swept path.