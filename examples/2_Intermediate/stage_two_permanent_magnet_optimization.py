#!/usr/bin/env python
r"""
In this example we solve a FOCUS like Stage II coil optimisation problem: the
goal is to find coils that generate a specific target normal field on a given
surface.  In this particular case we consider a vacuum field, so the target is
just zero.

The objective is given by

    J = (1/2) \int |B dot n|^2 ds
        + LENGTH_WEIGHT * (sum CurveLength)
        + DISTANCE_WEIGHT * MininumDistancePenalty(DISTANCE_THRESHOLD)
        + CURVATURE_WEIGHT * CurvaturePenalty(CURVATURE_THRESHOLD)
        + MSC_WEIGHT * MeanSquaredCurvaturePenalty(MSC_THRESHOLD)

if any of the weights are increased, or the thresholds are tightened, the coils
are more regular and better separated, but the target normal field may not be
achieved as well.

The target equilibrium is the QA configuration of arXiv:2108.03711.
"""

import os
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.objectives.utilities import QuadraticPenalty
from simsopt.geo.surfaceobjectives import QfmResidual, ToroidalFlux, Area, Volume
from simsopt.geo.curve import curves_to_vtk, create_equally_spaced_curves
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.magneticfieldclasses import InterpolatedField, UniformInterpolationRule, DipoleField
from simsopt.field.coil import Current, coils_via_symmetries
from simsopt.geo.curveobjectives import CurveLength, MinimumDistance, \
    MeanSquaredCurvature, LpCurveCurvature
from simsopt.field.tracing import SurfaceClassifier, \
    particles_to_vtk, compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data, \
    IterationStoppingCriterion
from simsopt.geo.plot import plot
from simsopt.util.permanent_magnet_optimizer import PermanentMagnetOptimizer
import time

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None

# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ncoils = 2

# Major radius for the initial circular coils:
R0 = 1.0

# Minor radius for the initial circular coils:
R1 = 0.5

# Number of Fourier modes describing each Cartesian component of each coil:
order = 5

# Number of iterations to perform:
ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']
ci = True
nfieldlines = 30 if ci else 30
tmax_fl = 30000 if ci else 40000
degree = 2 if ci else 4

MAXITER = 50 if ci else 400

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'  # _lowres

# Directory for output
OUT_DIR = "./output_pm/"
os.makedirs(OUT_DIR, exist_ok=True)

#######################################################
# End of input parameters.
#######################################################

# Initialize the boundary magnetic surface:
nphi = 16
ntheta = 32
s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)

stellsym = True
# Create the initial coils:
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=stellsym, R0=R0, R1=R1, order=order)
base_currents = [Current(6.5e5) for i in range(ncoils)]
# Since the target field is zero, one possible solution is just to set all
# currents to 0. To avoid the minimizer finding that solution, we fix one
# of the currents:
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
base_currents[0].fix_all()

# Uncomment if want to keep the coils circular
for i in range(ncoils):
    base_curves[i].fix_all()

bs = BiotSavart(coils)
bspoints = np.zeros((nphi, 3))
for i in range(nphi):
    # R = 1.0, Z = 0.0 
    bspoints[i] = np.array([1.0 * np.cos(s.quadpoints_phi[i]), 1.0 * np.sin(s.quadpoints_phi[i]), 0.0]) 
bs.set_points(bspoints)
print("Bmag at R = 1, Z = 0: ", np.linalg.norm(bs.B(), axis=-1))
print("toroidally averaged Bmag at R = 1, Z = 0: ", np.mean(np.linalg.norm(bs.B(), axis=-1)))
bs.set_points(s.gamma().reshape((-1, 3)))
# b_target_pm = -np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

# Define the regular coil optimization objective function:
Jf = SquaredFlux(s, bs)

# Form the total objective function. In this case we only optimize
# the currents and not the shapes, so only the squared flux is needed.
JF = Jf 

# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize


def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    return J, grad


print("dofs: ", JF.dof_names)
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
for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    J1, _ = f(dofs + eps*h)
    J2, _ = f(dofs - eps*h)
    print("err", (J1-J2)/(2*eps) - dJh)

print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
# Plot the optimized results
curves_to_vtk(curves, OUT_DIR + f"curves_opt")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_opt", extra_data=pointData)

# Basic TF coil currents now optimized, turning to 
# permanent magnet optimization now. 
pm_opt = PermanentMagnetOptimizer(
    s, coil_offset=0.1, dr=0.04, plasma_offset=0.1,
    B_plasma_surface=bs.B().reshape((nphi, ntheta, 3))
)
max_iter_MwPGP = 1000
print('Done initializing the permanent magnet object')
MwPGP_history, RS_history, m_history, dipoles = pm_opt._optimize(
    max_iter_MwPGP=max_iter_MwPGP, 
    max_iter_RS=10, reg_l2=0, reg_l0=0,
)

# recompute normal error using the dipole field and bs field
# to check nothing got mistranslated
b_dipole = DipoleField(pm_opt.dipole_grid, dipoles, pm_opt, stellsym=stellsym, nfp=s.nfp)
b_dipole.set_points(s.gamma().reshape((-1, 3)))
b_dipole._toVTK("Dipole_Fields")
pm_opt._plot_final_dipoles()

# b_dipole._toVTK("Dipole_Fields_surf", dim=())

dphi = (pm_opt.phi[1] - pm_opt.phi[0]) * 2 * np.pi
dtheta = (pm_opt.theta[1] - pm_opt.theta[0]) * 2 * np.pi
print("Average Bn without the PMs = ", 
      np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal() * np.sqrt(dphi * dtheta), axis=2))))
print("Average Bn with the PMs = ", 
      np.mean(np.abs(np.sum((bs.B() + b_dipole.B()).reshape((nphi, ntheta, 3)) * s.unitnormal() * np.sqrt(dphi * dtheta), axis=2))))

dipole_grid = pm_opt.dipole_grid
plt.figure()
ax = plt.axes(projection="3d")
colors = []
dipoles = dipoles.reshape(pm_opt.ndipoles, 3)
for i in range(pm_opt.ndipoles):
    colors.append(np.sqrt(dipoles[i, 0] ** 2 + dipoles[i, 1] ** 2 + dipoles[i, 2] ** 2))
sax = ax.scatter(dipole_grid[:, 0], dipole_grid[:, 1], dipole_grid[:, 2], c=colors)
plt.colorbar(sax)
plt.axis('off')
plt.grid(None)
plt.savefig('PMs_optimized.png')

dipoles = np.ravel(dipoles)
print(dipoles, pm_opt.m_maxima)

# Create full torus QA surface, plot Bn on that surface, 

# Does b_dipole satisfy the nfp and stellarator symmetry?
# Could try evaluating on bunch of random points and try again
# on points rotated/flipped/etc. 

print('Dipole field setup done')

make_plots = True 
if make_plots and (comm is None or comm.rank == 0):
    # Make plot of ATA element values
    plt.figure()
    plt.hist(np.ravel(np.abs(pm_opt.ATA)), bins=np.logspace(-20, -2, 100), log=True)
    plt.xscale('log')
    plt.grid(True)
    plt.savefig('histogram_ATA_values.png')

    # Make plot of the relax-and-split convergence
    plt.figure()
    plt.semilogy(MwPGP_history)
    plt.grid(True)
    plt.savefig('objective_history.png')

    # make histogram of the dipoles, normalized by their maximum values
    plt.figure()
    plt.hist(abs(np.ravel(m_history[:, :, -1])) / np.ravel(np.outer(pm_opt.m_maxima, np.ones(3))), bins=np.linspace(0, 1, 30), log=True)
    plt.savefig('m_histogram.png')
    print('Done optimizing the permanent magnets')

# Get full surface and get level sets for the Poincare plots below
#mpol = 5
#ntor = 5
#stellsym = True
#nfp = s.nfp
#phis = np.linspace(0, 1, nfp*2*ntor+1, endpoint=False)
#thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
#s = SurfaceRZFourier.from_vmec_input(filename, range="full torus", quadpoints_phi=phis, quadpoints_theta=thetas)

s = SurfaceRZFourier.from_vmec_input(filename, range="full torus", nphi=nphi, ntheta=ntheta)
#sc_fieldline = SurfaceClassifier(s, h=0.1, p=2)
#sc_fieldline.to_vtk(OUT_DIR + 'levelset', h=0.02)


def trace_fieldlines(bfield, label): 
    t1 = time.time()
    R0 = np.linspace(0.8, 1.3, nfieldlines)
    Z0 = np.zeros(nfieldlines)
    phis = [(i / 4) * (2 * np.pi / s.nfp) for i in range(4)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=1e-12, comm=comm,
        #phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
        phis=phis, stopping_criteria=[IterationStoppingCriterion(200000)])
    t2 = time.time()
    # print(fieldlines_phi_hits, np.shape(fieldlines_phi_hits))
    print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
    particles_to_vtk(fieldlines_tys, OUT_DIR + f'fieldlines_{label}')
    plot_poincare_data(fieldlines_phi_hits, phis, OUT_DIR + f'poincare_fieldline_{label}.png', dpi=300)


n = 16
rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
zs = s.gamma()[:, :, 2]
rrange = (np.min(rs), np.max(rs), n)
phirange = (0, 2 * np.pi / s.nfp, n * 2)
zrange = (0, np.max(zs), n // 2)
bsh = InterpolatedField(
    bs, degree, rrange, phirange, zrange, True, nfp=s.nfp, stellsym=stellsym
)
bsh.to_vtk('biot_savart_fields')
#trace_fieldlines(bsh, 'bsh_without_PMs')
print('Done with Poincare plots without the permanent magnets')
#t1 = time.time()
#bsh = InterpolatedField(
#    b_dipole, degree, rrange, phirange, zrange, True, nfp=s.nfp, stellsym=True
#)
#bsh.to_vtk('only_dipole_fields')
#t2 = time.time()
#trace_fieldlines(bsh, 'bsh_only_PMs')
#print('Done with Poincare plots with the permanent magnets')
t1 = time.time()
bsh = InterpolatedField(
    bs + b_dipole, degree, rrange, phirange, zrange, True, nfp=s.nfp, stellsym=stellsym
)
bsh.to_vtk('dipole_fields')
t2 = time.time()
#trace_fieldlines(bsh, 'bsh_PMs')
print('Done with Poincare plots with the permanent magnets')

bs.set_points(s.gamma().reshape((-1, 3)))
b_dipole.set_points(s.gamma().reshape((-1, 3)))
# For plotting Bn on the full torus surface at the end with just the dipole fields
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "biot_savart_opt", extra_data=pointData)
pointData = {"B_N": np.sum(b_dipole.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "only_pms_opt", extra_data=pointData)
pointData = {"B_N": np.sum((bs.B() + b_dipole.B()).reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "pms_opt", extra_data=pointData)
plt.show()

# Make QFMs
# make_qfm(s, bs)
# make_qfm(s, bs + b_dipole)
# Send message to Zhu about using paraview or whatever 3D thing they are using (coilPy) 