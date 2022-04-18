#!/usr/bin/env python
r"""
In this example we solve a FOCUS like Stage II coil optimisation problem: the
goal is to find coils that generate a specific target normal field on a given
surface.  In this particular case we consider a vacuum field, so the target is
just zero.

The objective is given by

    J = (1/2) ∫ |B_{BiotSavart}·n - B_{External}·n|^2 ds
        + LENGTH_PENALTY * Σ ½(CurveLength - L0)^2


The target equilibrium is the QA configuration of arXiv:2108.03711.
"""

import os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from simsopt.mhd.vmec import Vmec
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.objectives.utilities import QuadraticPenalty
from simsopt.geo.curve import curves_to_vtk, create_equally_spaced_curves
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Current, coils_via_symmetries, ScaledCurrent
from simsopt.geo.curveobjectives import CurveLength
from simsopt.mhd.virtual_casing import VirtualCasing


# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ncoils = 5

# Major radius for the initial circular coils:
R0 = 5.5

# Minor radius for the initial circular coils:
R1 = 1.25

# Number of Fourier modes describing each Cartesian component of each coil:
order = 6

# Weight on the curve length penalties in the objective function:
LENGTH_PEN = 1e0

# Number of iterations to perform:
ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']
MAXITER = 50 if ci else 400

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = 'wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc'
vmec_file = TEST_DIR / filename

# Only the phi resolution needs to be specified. The theta resolution
# is computed automatically to minimize anisotropy of the grid.

# vc = VirtualCasing.from_vmec(vmec_file, nphi=128)
# vc.save(TEST_DIR / ("vcasing_" + filename))

vc = VirtualCasing.load(TEST_DIR / ("vcasing_" + filename))
vc.plot()

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

#######################################################
# End of input parameters.
#######################################################

# Initialize the boundary magnetic surface:
nphi = 64
ntheta = 64

s = SurfaceRZFourier.from_wout(vmec_file, range="half period", nphi=nphi, ntheta=ntheta)
total_current = Vmec(vmec_file).external_current() / (2 * s.nfp)

vc = vc.resample(surf=s)
Btarget = vc.B_external_normal
Btotal = vc.B_total

# Create the initial coils:
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=128)
# Since we know the total sum of currents, we only optimize for ncoils-1
# currents, and then pick the last one so that they all add up to the correct
# value.
base_currents = [ScaledCurrent(Current(total_current/ncoils*1e-5), 1e5) for _ in range(ncoils-1)]
total_current = Current(total_current)
total_current.fix_all()
base_currents += [total_current - sum(base_currents)]

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))
curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

# Define the objective function:
Jf = SquaredFlux(s, bs, target=Btarget)
Jls = [CurveLength(c) for c in base_curves]

# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:
JF = Jf \
    + LENGTH_PEN * sum(QuadraticPenalty(Jls[i], Jls[i].J()) for i in range(len(base_curves)))

# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize


def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    BdotN = np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)-Btarget)/np.linalg.norm(Btotal, axis=2)
    BdotN_mean = np.mean(BdotN)
    BdotN_max = np.max(BdotN)
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨|B·n|⟩={BdotN_mean:.1e}, max(|B·n|)={BdotN_max:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    return 1e-4*J, 1e-4*grad


print("""
################################################################################
### Perform a Taylor test ######################################################
################################################################################
""")
f = fun
dofs = JF.x
np.random.seed(1)
h = np.random.uniform(size=dofs.shape)
J0, dJ0 = f(dofs)
dJh = sum(dJ0 * h)
for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    J1, _ = f(dofs + eps*h)
    J2, _ = f(dofs - eps*h)
    print("err", (J1-J2)/(2*eps) - dJh)

print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300, 'ftol': 1e-20, 'gtol': 1e-20}, tol=1e-20)
print("res", res)
curves_to_vtk(curves, OUT_DIR + f"curves_opt")
BdotN = np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)-Btarget)/np.linalg.norm(Btotal, axis=2)
pointData = {"B_N": BdotN[:, :, None]}
s.to_vtk(OUT_DIR + f"surf_opt", extra_data=pointData)