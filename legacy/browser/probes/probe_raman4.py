"""Round-15 probe 4: build the CPHF polarizability and pin its convention.

probe 3 settled the physics question -- the finite-field kernel is right and
6-31G* is simply missing the diffuse functions that carry alpha (5.13 a.u.
against 9.89 at aug-cc-pVTZ, experiment 9.6-9.9).  So the remaining question is
mechanical: is there a *cheap* route to alpha?

That matters because a Raman spectrum needs dalpha/dx at all 6N displaced
geometries.  Finite field costs ~18 SCFs per displacement (324 for water);
CPHF costs one SCF plus one Krylov solve, i.e. roughly the same as the IR loop
the app already runs.  But ``pyscf.prop`` is missing, so the CPHF has to be
wired by hand and its conventions established rather than remembered.

What is already known from reading PySCF's own source, not guessed:

* ``cphf.solve_nos1`` builds ``e_ai = 1/(e_a - e_i)`` with ``e_a`` from
  ``mo_occ==0`` and ``e_i`` from ``mo_occ>0``, so the packed vector is laid out
  ``(nvir, nocc)`` -- virtual index first.
* It calls ``lib.krylov(vind_vo, -e_ai*h1)`` where ``vind_vo(mo1) =
  e_ai * fvind(mo1)``, and ``lib.krylov`` solves ``(1 + aop) x = b``.  Chaining
  those:  ``(1 + e_ai*fvind) x = -e_ai h1``  ->  ``(e_a - e_i + fvind) x = -h1``.
  So ``fvind`` must supply ONLY the two-electron part (the diagonal is added by
  the solver), and the solution is ``x = -(A+B)^{-1} h1``.
* ``mf.gen_response(mo_coeff, mo_occ, hermi=1)`` is the supported DFT kernel --
  it picks up the XC second derivative via ``nr_rks_fxc`` and the exact-exchange
  part for hybrids.  Calling ``mf.get_veff`` instead would silently drop the
  whole XC response, which is most of the answer for B3LYP.
* PySCF's own ``hessian/rhf.py:gen_vind`` builds ``dm1`` with a ``*2`` "for
  double occupancy" and then symmetrises, and projects back with
  ``C^T v1 C_occ``.  That is the pattern copied here.

Because ``x = -(A+B)^{-1}h1`` carries a sign the derivation could have gotten
backwards, and because alpha's prefactor is a plain number that no amount of
staring settles, the probe does not assert either.  It evaluates the candidates
and compares against the finite-field tensor, which was independently verified
in probe 2 (Hellmann-Feynman ``dE/dF = -mu`` to 1.7e-6, ``int1e_r`` reproducing
``mf.dip_moment`` to 4.4e-16).  Agreement between two routes built on different
machinery is the only claim worth making.

Run:  python probes/probe_raman4.py
"""
from __future__ import annotations

import time

import numpy as np

import backend.bootstrap as bootstrap

bootstrap.setup()

from pyscf import dft, gto
from pyscf.scf import cphf

WATER = [
    ("O", 0.000000, 0.000000, 0.119262),
    ("H", 0.000000, 0.763239, -0.477047),
    ("H", 0.000000, -0.763239, -0.477047),
]
BOHR3_TO_A3 = 0.52917721092 ** 3


def _scf(basis, xc):
    mol = gto.M(atom=WATER, basis=basis, verbose=0)
    mol.chkfile = None
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.verbose = 0
    mf.conv_tol = 1e-12
    mf.chkfile = None
    mf.kernel()
    return mol, mf


def alpha_ff(basis, xc="b3lyp", h=1e-3):
    """alpha by central differences of the SCF energy in a finite field."""
    mol0, base = _scf(basis, xc)
    dip = mol0.intor("int1e_r")
    h0 = mol0.intor("int1e_kin") + mol0.intor("int1e_nuc")
    dm0 = base.make_rdm1()

    def e_in(field):
        m = gto.M(atom=WATER, basis=basis, verbose=0)
        m.chkfile = None
        k = dft.RKS(m)
        k.xc = xc
        k.verbose = 0
        k.conv_tol = 1e-12
        k.chkfile = None
        hf_ = h0 + np.einsum("x,xij->ij", np.asarray(field, float), dip)
        k.get_hcore = lambda *a, _h=hf_: _h
        k.kernel(dm0=dm0)
        return k.e_tot

    a = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            d = {}
            for si in (1, -1):
                for sj in (1, -1):
                    f = np.zeros(3)
                    f[i] += si * h
                    f[j] += sj * h
                    d[(si, sj)] = e_in(f)
            # d2E/dFi dFj from the four corners; alpha is its negative.
            a[i, j] = a[j, i] = -(d[(1, 1)] - d[(1, -1)] - d[(-1, 1)]
                                  + d[(-1, -1)]) / (4 * h * h)
    return a


def alpha_cphf(basis, xc="b3lyp", prefactor=2.0, flip=False):
    """alpha from the coupled-perturbed SCF equations.

    ``x = -(A+B)^{-1} h1`` (see the module docstring for why the minus is
    there).  ``flip`` undoes it so both sign hypotheses can be scored against
    the finite-field tensor instead of being assumed.
    """
    mol, mf = _scf(basis, xc)
    mo_c, mo_e, mo_occ = mf.mo_coeff, mf.mo_energy, mf.mo_occ
    occ = mo_occ > 0
    vir = mo_occ == 0
    orbo = mo_c[:, occ]
    orbv = mo_c[:, vir]
    nocc, nvir = orbo.shape[1], orbv.shape[1]

    dip_ao = mol.intor("int1e_r")
    # (3, nvir, nocc) -- virtual index first, matching solve_nos1's e_ai.
    h1 = np.einsum("pa,xpq,qi->xai", orbv, dip_ao, orbo, optimize=True)

    vresp = mf.gen_response(mo_c, mo_occ, hermi=1)

    def fvind(x):
        x = x.reshape(nvir, nocc)
        # *2 for double occupancy, then symmetrise -- PySCF's hessian/rhf.py.
        dm1 = np.einsum("pa,ai,qi->pq", orbv, x * 2.0, orbo, optimize=True)
        v1 = vresp(dm1 + dm1.T)
        return np.einsum("pa,pq,qi->ai", orbv, v1, orbo, optimize=True)

    # One field direction at a time.  solve_nos1 forms e_ai as (nvir, nocc) and
    # multiplies it into h1, so h1 has to be 2-D per perturbation: stacking the
    # three components into (3, nvir, nocc) broadcasts against nothing and dies
    # with "operands could not be broadcast together with shapes (3,65) (13,5)".
    x = np.empty((3, nvir, nocc))
    for d in range(3):
        x[d] = cphf.solve(fvind, mo_e, mo_occ, h1[d],
                          max_cycle=200, tol=1e-12, verbose=0)[0]
    if flip:
        x = -x
    return prefactor * np.einsum("xai,yai->xy", h1, x, optimize=True)


print("=" * 78)
print("PART 1 -- which convention reproduces the finite-field tensor?")
print("=" * 78)
for basis in ("6-31g*", "aug-cc-pvdz"):
    t0 = time.time()
    aff = alpha_ff(basis)
    t_ff = time.time() - t0
    print(f"\n  {basis}   (finite field: {t_ff:.1f} s)")
    print("    alpha_ff  =", np.array2string(aff, precision=5,
                                             suppress_small=True)
          .replace("\n", "\n                "))

    best = None
    for pre in (2.0, 1.0, 4.0, -2.0):
        for flip in (False, True):
            try:
                t0 = time.time()
                ac = alpha_cphf(basis, prefactor=pre, flip=flip)
                dt = time.time() - t0
            except Exception as exc:                       # noqa: BLE001
                print(f"    pre={pre:+.0f} flip={int(flip)}  "
                      f"FAILED {type(exc).__name__}: {exc}")
                continue
            err = np.abs(ac - aff).max() / np.abs(aff).max()
            tag = ""
            if best is None or err < best[0]:
                best = (err, pre, flip)
                tag = "   <-- best so far"
            print(f"    pre={pre:+.0f} flip={int(flip)}  "
                  f"max rel err {err:>9.2e}   ({dt:.2f} s){tag}")
    print(f"    => best: prefactor {best[1]:+.0f}, flip={int(best[2])}, "
          f"err {best[0]:.2e}" if best else "    => no convention worked")

print()
print("=" * 78)
print("PART 2 -- the winning convention, full tensors")
print("=" * 78)
win_pre, win_flip = (best[1], best[2]) if best else (2.0, False)
print(f"  using prefactor {win_pre:+.0f}, flip={int(win_flip)}")
for basis in ("6-31g*", "6-31+g*", "aug-cc-pvdz"):
    aff = alpha_ff(basis)
    ac = alpha_cphf(basis, prefactor=win_pre, flip=win_flip)
    iso_f = float(np.trace(aff) / 3)
    iso_c = float(np.trace(ac) / 3)
    print(f"  {basis:<14} ff iso {iso_f:8.4f} a.u. ({iso_f * BOHR3_TO_A3:.4f} A^3)"
          f"   cphf iso {iso_c:8.4f}   max|d| {np.abs(ac - aff).max():.2e}")
    print("    cphf diag", " ".join(f"{v:9.5f}" for v in np.diag(ac)))
