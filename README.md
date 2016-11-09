# Stokes Preconditioning

Comparison of two preconditioners for Stokes equations in Firedrake and PETSc.
  + Inverse Mass Diagonal
  + BFBT Preconditioner. Notes:
    + Using the BFBT preconditioner for Dirichlet boundary conditions is tricky. Some authors suggest different types of scaling around the boundaries. We notes that natural boundary conditions seem to result in mesh independence in the simple 2d case. this needs to be investigated further.
    + BFBT type preconditioner only work on quadrilateral meshes!

Florian Wechsung <wechsung@maths.ox.ac.uk>
