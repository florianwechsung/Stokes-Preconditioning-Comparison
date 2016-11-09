"""
Comparison of two preconditioners for Stokes equations.
    + Inverse Mass Diagonal
    + BFBT Preconditioner. Notes:
        + Using the BFBT preconditioner for Dirichlet boundary conditions is tricky. Some authors suggest different types of scaling around the boundaries. We notes that natural boundary conditions seem to result in mesh independence in the simple 2d case. this needs to be investigated further.
        + BFBT type preconditioner only work on quadrilateral meshes!
"""

from firedrake import *
from firedrake.petsc import PETSc
import numpy as np
from numpy.linalg import inv, cond, eig
parameters["pyop2_options"]["block_sparsity"] = False

n=20
mesh = UnitSquareMesh(n, n, quadrilateral=True)
if n<31:
    doDenseAnalysis = True
else
    doDenseAnalysis = False
    
V = VectorFunctionSpace(mesh, "CG", 2, name="velocity")
W = FunctionSpace(mesh, "CG", 1, name="pressure")
Z = V*W

(u, p) = TrialFunctions(Z)
(v, q) = TestFunctions(Z)

zerovector = Constant((0.0, 0.0))

F = (
      inner(grad(u), grad(v)) * dx
    - div(v) * p * dx
    - q * div(u) * dx
    + inner(zerovector, v) * dx
    )

bcin = DirichletBC(Z.sub(0), Expression(("x[1]*(1-x[1])", "0.0")), 1)
bcnoslip = DirichletBC(Z.sub(0), zerovector, (3, 4))

bcs = [
        bcin,
        bcnoslip,
      ]

def PETScToNumpy(mat):
    return mat.convert(PETSc.Mat.Type.DENSE).getDenseArray()

def NumpyToPETSc(mat):
    res = PETSc.Mat()
    res.createDense(mat.shape, array=mat)
    return res

def GetSchurComplement():
    Fass = assemble(lhs(F))
    for bc in bcs:
        bc.apply(Fass)
    A = Fass.M[0,0].handle
    B = Fass.M[1,0].handle
    BT = Fass.M[0,1].handle
    npA = PETScToNumpy(A)
    npB = PETScToNumpy(B)
    npBT = PETScToNumpy(BT)
    return -1 * npB.dot(inv(npA).dot(npBT))

def GetBFBTApproximateToSchurInverse(method):
    Fass = assemble(lhs(F))
    # for bc in bcs:
        # bc.apply(Fass)
    A = Fass.M[0,0].handle
    B = Fass.M[1,0].handle
    BT = Fass.M[0,1].handle
    npA = PETScToNumpy(A)
    npB = PETScToNumpy(B)
    npBT = PETScToNumpy(BT)
    if method == "lumpedVelocityMass":
        npvmass = PETScToNumpy(assemble(inner(u,v) * dx).M[0, 0].handle)
        npCD = np.diag(np.sum(npvmass, axis=0))
    else:
        npCD = np.diag(np.diag(npA))
    npCDInv = inv(npCD)
    npBEBT = npB.dot(npCDInv.dot(npBT))
    npBEBT_inv = inv(npBEBT)
    npMiddle = npB.dot(npCDInv.dot(npA.dot(npCDInv.dot(npBT))))
    npSInvApprox = npBEBT_inv.dot(npMiddle.dot(npBEBT_inv))
    return (-1) * npBEBT_inv.dot(npMiddle.dot(npBEBT_inv))

def GetInverseMassApproximateToSchurInverse():
    Q = assemble(p * q * dx).M[1, 1].handle
    npQ = PETScToNumpy(Q)
    return inv(npQ) * (-1)

def Kappa(mat, name):
    print "kappa(%s)=%f" % (name, cond(mat))

def solve(schur_inverse_approx):
    solution = Function(Z)
    problem = LinearVariationalProblem(lhs(F), rhs(F), solution, bcs = bcs)
    params = {'ksp_monitor': True,
              'ksp_type': 'minres',
              'ksp_rtol': 1e-9,
              'ksp_atol': 1e-10,
              'ksp_stol': 1e-16,
              'ksp_max_it': 10,
              'pc_type': 'fieldsplit',
              'pc_fieldsplit_type': 'schur',
              'pc_fieldsplit_schur_factorization_type': 'diag',
              'pc_fieldsplit_schur_precondition': 'user',
              'fieldsplit_velocity_ksp_type': 'richardson',
              'fieldsplit_velocity_pc_type': 'hypre',
              'fieldsplit_velocity_ksp_max_it': 1,
              # 'fieldsplit_velocity_ksp_monitor': None,
              'fieldsplit_pressure_ksp_type': 'richardson',
              'fieldsplit_pressure_pc_type': 'mat',
              'fieldsplit_pressure_rtol': 1e-10,
              'fieldsplit_pressure_atol': 1e-10,
              'fieldsplit_pressure_ksp_max_it': 1,
              # 'fieldsplit_pressure_ksp_monitor': None,
              # 'fieldsplit_pressure_ksp_monitor_true_residual': None
              }
    solver = LinearVariationalSolver(problem, solver_parameters=params)
    schuruser = PETSc.PC.SchurPreType.USER
    pc = solver.snes.ksp.pc
    pc.setFieldSplitSchurPreType(schuruser, schur_inverse_approx)
    try:
        solver.solve()
    except:
        pass


def GetPETScInverseMassApproximateToSchurInverse():
    mass = assemble( p*q*dx).M[1,1].handle
    mass_diag = mass.createVecLeft()
    mass.getDiagonal(mass_diag)
    mass_diag.reciprocal()
    mass_diag.scale(-1)

    class InvMassSchurInvApprox(object):
        def mult(self, mat, x, y):
            y.pointwiseMult(mass_diag, x)

    schur = PETSc.Mat()
    schur.createPython(mass.getSizes(), InvMassSchurInvApprox())
    schur.setUp()
    return schur

def GetPETScBFBTApproximateToSchurInverse():

    Fass = assemble(lhs(F))
    Gass = assemble(inner(u, v) * dx)
    # bcin.apply(Fass)
    # bcnoslip.apply(Fass)
    # bcin.apply(Gass)
    # bcnoslip.apply(Gass)
    A = Fass.M[0,0].handle
    BT = Fass.M[0,1].handle
    B = Fass.M[1,0].handle
    D = Fass.M[1,1].handle
    VM = Gass.M[0, 0].handle
    # Adiag = A.getDiagonal()
    Adiag = VM.getRowSum()
    invAdiag = Adiag.duplicate()
    Adiag.copy(invAdiag)
    invAdiag.reciprocal()

    Einv = A.duplicate()
    Einv.setDiagonal(invAdiag)
    # Einv.convert(PETSc.Mat.Type.SEQAIJ)
    BEBT = B.matMult(Einv.matMult(BT))

    BEBT_ksp = PETSc.KSP().create()
    BEBT_ksp.setOperators(BEBT)
    opts = PETSc.Options()
    BEBT_ksp.setOptionsPrefix("bebt_")
    opts['bebt_ksp_type'] = 'richardson'
    opts['bebt_pc_type'] = 'hypre'
    opts['bebt_ksp_max_it'] = 1
    opts['bebt_ksp_atol'] = 1.0e-9
    opts['bebt_ksp_rtol'] = 1.0e-9
    BEBT_ksp.setUp()
    BEBT_ksp.setFromOptions()

    MIDDLE = B.matMult(Einv.matMult(A.matMult(Einv.matMult(BT)))) 
    MIDDLE.scale(-1)
    class SchurInvApprox(object):
        def mult(self, mat, x, y):
            y1 = y.duplicate()
            BEBT_ksp.solve(x, y1)
            y2 = y.duplicate()
            MIDDLE.mult(y1, y2)
            BEBT_ksp.solve(y2, y)

    schur = PETSc.Mat()
    schur.createPython(D.getSizes(), SchurInvApprox())
    schur.setUp()
    return schur

if doDenseAnalysis:
    npS = GetSchurComplement()
    npSinv = inv(npS)
    npSinv_BFBT_1 = GetBFBTApproximateToSchurInverse("lumpedVelocityMass")
    npSinv_BFBT_2 = GetBFBTApproximateToSchurInverse("diagonalVelocityLaplace")
    npSinv_mass = GetInverseMassApproximateToSchurInverse()
    Kappa(npS, "npS")
    Kappa(npSinv, "npSinv")
    Kappa(npS.dot(npSinv), "npS * npSinv")
    Kappa(npS.dot(npSinv_BFBT_1), "npS * npSinv_BFBT_1")
    Kappa(npS.dot(npSinv_BFBT_2), "npS * npSinv_BFBT_2")
    Kappa(npS.dot(npSinv_mass), "npS * npSinv_mass")

    S = NumpyToPETSc(npS)
    Sinv = NumpyToPETSc(npSinv)
    Sinv_BFBT_1 = NumpyToPETSc(npSinv_BFBT_1)
    Sinv_BFBT_2 = NumpyToPETSc(npSinv_BFBT_2)
    Sinv_mass = NumpyToPETSc(npSinv_mass)
    print "Sinv"
    solve(Sinv)
    print "Sinv_mass"
    solve(Sinv_mass)
    print "Sinv_BFBT_1"
    solve(Sinv_BFBT_1)
    print "Sinv_BFBT_2"
    solve(Sinv_BFBT_2)

print "GetPETScInverseMassApproximateToSchurInverse()"
solve(GetPETScInverseMassApproximateToSchurInverse())
print "GetPETScBFBTApproximateToSchurInverse()"
solve(GetPETScBFBTApproximateToSchurInverse())
