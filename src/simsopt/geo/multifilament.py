import simsoptpp as sopp
from .._core.graph_optimizable import Optimizable
from .._core.derivative import Derivative
from .curve import Curve
from .jit import jit
import numpy as np
import jax.numpy as jnp
from jax import vjp, jvp


"""
Implementation of the centroid frame introduced in doi:10.1017/S0022377820000756.
Given a curve, one defines a normal and binormal vector and then creates a grid of curves by shifting
along the normal and binormal vector.
In addition, we specify an angle along the curve that allows to optimise for the rotation of the winding pack.
"""


class CurveShiftedRotated(sopp.Curve, Curve):

    def __init__(self, curve, dn, db, rotation=None):
        self.curve = curve
        sopp.Curve.__init__(self, curve.quadpoints)
        deps = [curve]
        if rotation is not None:
            deps.append(rotation)
        Curve.__init__(self, depends_on=deps)
        self.curve = curve
        self.dn = dn
        self.db = db
        if rotation is None:
            rotation = ZeroRotation(curve.quadpoints)
        self.rotation = rotation

    def recompute_bell(self, parent=None):
        self.invalidate_cache()

    def gamma_impl(self, gamma, quadpoints):
        assert quadpoints.shape[0] == self.curve.quadpoints.shape[0]
        assert np.linalg.norm(quadpoints - self.curve.quadpoints) < 1e-15
        c = self.curve
        t, n, b = rotated_centroid_frame(c.gamma(), c.gammadash(), self.rotation.alpha(c.quadpoints))
        gamma[:] = self.curve.gamma() + self.dn * n + self.db * b

    def gammadash_impl(self, gammadash):
        c = self.curve
        td, nd, bd = rotated_centroid_frame_dash(
            c.gamma(), c.gammadash(), c.gammadashdash(),
            self.rotation.alpha(c.quadpoints), self.rotation.alphadash(c.quadpoints)
        )
        gammadash[:] = self.curve.gammadash() + self.dn * nd + self.db * bd

    def dgamma_by_dcoeff_vjp(self, v):
        g = self.curve.gamma()
        gd = self.curve.gammadash()
        a = self.rotation.alpha(self.curve.quadpoints)
        zero = np.zeros_like(v)
        vg = rotated_centroid_frame_dcoeff_vjp0(g, gd, a, (zero, self.dn*v, self.db*v))
        vgd = rotated_centroid_frame_dcoeff_vjp1(g, gd, a, (zero, self.dn*v, self.db*v))
        va = rotated_centroid_frame_dcoeff_vjp2(g, gd, a, (zero, self.dn*v, self.db*v))
        return self.curve.dgamma_by_dcoeff_vjp(v + vg) \
            + self.curve.dgammadash_by_dcoeff_vjp(vgd) \
            + self.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, va)

    def dgammadash_by_dcoeff_vjp(self, v):
        g = self.curve.gamma()
        gd = self.curve.gammadash()
        gdd = self.curve.gammadashdash()
        a = self.rotation.alpha(self.curve.quadpoints)
        ad = self.rotation.alphadash(self.curve.quadpoints)
        zero = np.zeros_like(v)

        vg = rotated_centroid_frame_dash_dcoeff_vjp0(g, gd, gdd, a, ad, (zero, self.dn*v, self.db*v))
        vgd = rotated_centroid_frame_dash_dcoeff_vjp1(g, gd, gdd, a, ad, (zero, self.dn*v, self.db*v))
        vgdd = rotated_centroid_frame_dash_dcoeff_vjp2(g, gd, gdd, a, ad, (zero, self.dn*v, self.db*v))
        va = rotated_centroid_frame_dash_dcoeff_vjp3(g, gd, gdd, a, ad, (zero, self.dn*v, self.db*v))
        vad = rotated_centroid_frame_dash_dcoeff_vjp4(g, gd, gdd, a, ad, (zero, self.dn*v, self.db*v))
        return self.curve.dgamma_by_dcoeff_vjp(vg) \
            + self.curve.dgammadash_by_dcoeff_vjp(v+vgd) \
            + self.curve.dgammadashdash_by_dcoeff_vjp(vgdd) \
            + self.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, va) \
            + self.rotation.dalphadash_by_dcoeff_vjp(self.curve.quadpoints, vad)


class FilamentRotation(Optimizable):

    def __init__(self, quadpoints, order):
        self.order = order
        Optimizable.__init__(self, x0=np.zeros((2*order+1, )))
        self.quadpoints = quadpoints
        self.jac = rotation_dcoeff(quadpoints, order)
        self.jacdash = rotationdash_dcoeff(quadpoints, order)
        self.jax_alpha = jit(lambda dofs, points: jaxrotation_pure(dofs, points, self.order))
        self.jax_alphadash = jit(lambda dofs, points: jaxrotationdash_pure(dofs, points, self.order))

    def alpha(self, quadpoints):
        return self.jax_alpha(self._dofs.full_x, quadpoints)

    def alphadash(self, quadpoints):
        return self.jax_alphadash(self._dofs.full_x, quadpoints)

    def dalpha_by_dcoeff_vjp(self, quadpoints, v):
        return Derivative({self: sopp.vjp(v, self.jac)})

    def dalphadash_by_dcoeff_vjp(self, quadpoints, v):
        return Derivative({self: sopp.vjp(v, self.jacdash)})


class ZeroRotation():

    def __init__(self, quadpoints):
        self.zero = np.zeros((quadpoints.size, ))

    def alpha(self, quadpoints):
        return self.zero

    def alphadash(self, quadpoints):
        return self.zero

    def dalpha_by_dcoeff_vjp(self, quadpoints, v):
        return Derivative({})

    def dalphadash_by_dcoeff_vjp(self, quadpoints, v):
        return Derivative({})


@jit
def rotated_centroid_frame(gamma, gammadash, alpha):
    t = gammadash
    t *= 1./jnp.linalg.norm(gammadash, axis=1)[:, None]
    R = jnp.mean(gamma, axis=0)  # centroid
    delta = gamma - R[None, :]
    n = delta - jnp.sum(delta * t, axis=1)[:, None] * t
    n *= 1./jnp.linalg.norm(n, axis=1)[:, None]
    b = jnp.cross(t, n, axis=1)

    # now rotate the frame by alpha
    nn = jnp.cos(alpha)[:, None] * n - jnp.sin(alpha)[:, None] * b
    bb = jnp.sin(alpha)[:, None] * n + jnp.cos(alpha)[:, None] * b
    return t, nn, bb


rotated_centroid_frame_dash = jit(
    lambda gamma, gammadash, gammadashdash, alpha, alphadash: jvp(rotated_centroid_frame,
                                                                  (gamma, gammadash, alpha),
                                                                  (gammadash, gammadashdash, alphadash))[1])

rotated_centroid_frame_dcoeff_vjp0 = jit(
    lambda gamma, gammadash, alpha, v: vjp(
        lambda g: rotated_centroid_frame(g, gammadash, alpha), gamma)[1](v)[0])

rotated_centroid_frame_dcoeff_vjp1 = jit(
    lambda gamma, gammadash, alpha, v: vjp(
        lambda gd: rotated_centroid_frame(gamma, gd, alpha), gammadash)[1](v)[0])

rotated_centroid_frame_dcoeff_vjp2 = jit(
    lambda gamma, gammadash, alpha, v: vjp(
        lambda a: rotated_centroid_frame(gamma, gammadash, a), alpha)[1](v)[0])

rotated_centroid_frame_dash_dcoeff_vjp0 = jit(
    lambda gamma, gammadash, gammadashdash, alpha, alphadash, v: vjp(
        lambda g: rotated_centroid_frame_dash(g, gammadash, gammadashdash, alpha, alphadash), gamma)[1](v)[0])

rotated_centroid_frame_dash_dcoeff_vjp1 = jit(
    lambda gamma, gammadash, gammadashdash, alpha, alphadash, v: vjp(
        lambda gd: rotated_centroid_frame_dash(gamma, gd, gammadashdash, alpha, alphadash), gammadash)[1](v)[0])

rotated_centroid_frame_dash_dcoeff_vjp2 = jit(
    lambda gamma, gammadash, gammadashdash, alpha, alphadash, v: vjp(
        lambda gdd: rotated_centroid_frame_dash(gamma, gammadash, gdd, alpha, alphadash), gammadashdash)[1](v)[0])

rotated_centroid_frame_dash_dcoeff_vjp3 = jit(
    lambda gamma, gammadash, gammadashdash, alpha, alphadash, v: vjp(
        lambda a: rotated_centroid_frame_dash(gamma, gammadash, gammadashdash, a, alphadash), alpha)[1](v)[0])

rotated_centroid_frame_dash_dcoeff_vjp4 = jit(
    lambda gamma, gammadash, gammadashdash, alpha, alphadash, v: vjp(
        lambda ad: rotated_centroid_frame_dash(gamma, gammadash, gammadashdash, alpha, ad), alphadash)[1](v)[0])


def jaxrotation_pure(dofs, points, order):
    rotation = jnp.zeros((len(points), ))
    rotation += dofs[0]
    for j in range(1, order+1):
        rotation += dofs[2*j-1] * jnp.sin(2*np.pi*j*points)
        rotation += dofs[2*j] * jnp.cos(2*np.pi*j*points)
    return rotation


def jaxrotationdash_pure(dofs, points, order):
    rotation = jnp.zeros((len(points), ))
    for j in range(1, order+1):
        rotation += dofs[2*j-1] * 2*np.pi*j*jnp.cos(2*np.pi*j*points)
        rotation -= dofs[2*j] * 2*np.pi*j*jnp.sin(2*np.pi*j*points)
    return rotation


def rotation_dcoeff(points, order):
    jac = np.zeros((len(points), 2*order+1))
    jac[:, 0] = 1
    for j in range(1, order+1):
        jac[:, 2*j-1] = np.sin(2*np.pi*j*points)
        jac[:, 2*j+0] = np.cos(2*np.pi*j*points)
    return jac


def rotationdash_dcoeff(points, order):
    jac = np.zeros((len(points), 2*order+1))
    for j in range(1, order+1):
        jac[:, 2*j-1] = +2*np.pi*j*np.cos(2*np.pi*j*points)
        jac[:, 2*j+0] = -2*np.pi*j*np.sin(2*np.pi*j*points)
    return jac