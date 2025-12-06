Observer-independent implicit neural representations using projective geometric algebra -> coordinate free neural fields.

The neural SDF is a continuous implicit function that represents the input shape.

By embedding geometric priors directly into the network architecture, this library enables the learning of neural fields that are mathematically guaranteed to be invariant to rigid body transformations. Instead of learning a function of absolute coordinates $f(x,y,z)$, it learns a function of relative geometric configurations using PGA Motors.

- Coordinate independence: Networks do not see "points" $(x,y,z)$; they see "relationships" transformed by the observer's frame.
- PGA motors: All rigid body motions (translations + rotations) are handled uniformly using the even subalgebra of $\mathbb{R}_{3,0,1}$.
- Geometric regularization: Loss functions enforce physical laws (Eikonal constraint, Normal alignment) in the local frame, ensuring valid geometry regardless of orientation.
- Spacetime Dynamics: Extends static geometry to 4D with temporal motors and kinematic chains for articulated motion.

## neural animation

In example 10, a neural SDF is trained on the canonical T-pose mesh, then animated using skeleton-driven deformation. The comparison shows the original mesh (top) vs the neural reconstruction (bottom) from front, side, and top views.

![Articulated Character Animation](output/10_comparison_multiview.gif)

## dynamic scene Composition

Example 9 demonstrates how multiple trained primitive shapes (Sphere, Box, Cylinder) can be dynamically composed and animated at runtime. Each object moves according to its own trajectory (orbit, rotation, translation), and their SDFs are blended using a "smooth union" operation, showcasing the library's ability to handle interactive scenes without retraining.

![Dynamic Scene Composition](output/09_dynamic_scene.gif)
