"""Round-15 probe 3: is 6-31G* simply a bad basis for polarizability?

Finite field gives alpha_iso = 5.13 a.u. for water against 9.6-9.9 a.u. from
experiment.  Two explanations: the implementation is wrong, or 6-31G* is
missing the diffuse functions that dominate alpha.  These are easy to tell
apart -- add diffuse functions and watch.

Also fixes the CPHF einsum from probe 1 and compares the two routes, because
two routes built on different machinery agreeing is worth more than either
one agreeing with a remembered number.

Reference values, water, isotropic polarizability in a.u. (1 a.u. = 0.148185
A^3):
    experiment                     9.6 - 9.9
    B3LYP/aug-cc-pVTZ             ~9.6
    basis without diffuse functions   several tenths low, badly

Run:  python probes/probe_raman3.py
"""
from __future__ import annotations

import time

import numpy as np

import backend.bootstrap as bootstrap

bootstrap.setup()

from pyscf import dft, gto

WATER = [
    ("O", 0.000000, 0.000000, 0.119262),
    ("H", 0.000000, 0.763239, -0.477047),
    ("H", 0.000000, -0.763239, -0.477047),
]
BOHR3_TO_A3 = 0.52917721092 ** 3


def alpha_ff(basis, xc="b3lyp", h=1e-3):
    mol = gto.M(atom=WATER, basis=basis, verbose=0)
    mol.chkfile = None
    dip = mol.intor("int1e_r")
    h0 = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    base = dft.RKS(mol)
    base.xc = xc
    base.verbose = 0
    base.conv_tol = 1e-11
    base.chkfile = None
    base.kernel()
    dm0 = base.make_rdm1()

    def e_in(field):
        m = gto.M(atom=WATER, basis=basis, verbose=0)
        m.chkfile = None
        k = dft.RKS(m)
        k.xc = xc
        k.verbose = 0
        k.conv_tol = 1e-11
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
            a[i, j] = a[j, i] = -(d[(1, 1)] - d[(1, -1)] - d[(-1, 1)]
                                  + d[(-1, -1)]) / (4 * h * h)
    return a


def alpha_cphf(basis, xc="b3lyp"):
    """alpha from the CPHF equations (the standard route).

    (A+B) x_i = h_i^{ov}, with the field coupling as +E.r, and
        alpha_ij = 2 sum_{ia} h_i^{ia} x_j^{ia}
    """
    from pyscf.scf import cphf

    mol = gto.M(atom=WATER, basis=basis, verbose=0)
    mol.chkfile = None
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.verbose = 0
    mf.conv_tol = 1e-11
    mf.chkfile = None
    mf.kernel()

    mo_c, mo_occ, mo_e = mf.mo_coeff, mf.mo_occ, mf.mo_energy
    occ = mo_occ > 0
    vir = mo_occ == 0
    nocc, nvir = int(occ.sum()), int(vir.sum())

    dip_ao = mol.intor("int1e_r")
    # dip_mo[x, i, a] = sum_{pq} C[p,i] dip_ao[x,p,q] C[q,a]
    dip_mo = np.einsum("pi,xpq,qa->xia", mo_c[:, occ], dip_ao, mo_c[:, vir],
                       optimize=True)
    h1 = dip_mo.reshape(3, nocc * nvir)

    def fvind(x):
        x = x.reshape(3, nocc, nvir)
        dm = np.einsum("xia,ap,qi->xpq", x, mo_c[:, vir], mo_c[:, occ],
                       optimize=True)
        dm = dm + dm.transpose(0, 2, 1)
        v = mf.get_veff(mol, dm)
        return np.einsum("qi,xpq,pa->xia", mo_c[:, occ], v, mo_c[:, vir],
                         optimize=True).reshape(3, nocc * nvir)

    x = cphf.solve(fvind, mo_e, mo_occ, h1, max_cycle=200, tol=1e-11,
                   verbose=0)[0].reshape(3, nocc, nvir)
    return 2.0 * np.einsum("xia,jia->ij", dip_mo, x, optimize=True)


print("=" * 78)
print("PART 1 -- diffuse functions, and the two routes")
print("=" * 78)
print(f"  {'basis':<16} {'alpha_iso (a.u.)':>17} {'(A^3)':>9} "
      f"{'anisotropy':>11}   {'CPHF diff':>10}")
for basis in ("6-31g", "6-31g*", "6-31+g*", "6-31++g**", "cc-pvdz",
              "aug-cc-pvdz", "cc-pvtz", "aug-cc-pvtz"):
    try:
        t0 = time.time()
        a = alpha_ff(basis)
        iso = float(np.trace(a) / 3)
        d = a - np.eye(3) * iso
        aniso = float(np.sqrt(np.sum(d * d)))
        line = (f"  {basis:<16} {iso:>17.4f} {iso * BOHR3_TO_A3:>9.4f} "
                f"{aniso:>11.4f}")
        try:
            ac = alpha_cphf(basis)
            line += f"   {np.abs(ac - a).max() / np.abs(a).max():>10.2e}"
        except Exception as exc:
            line += f"   {'n/a':>10}  ({type(exc).__name__})"
        print(line, flush=True)
    except Exception as exc:
        print(f"  {basis:<16} FAILED: {type(exc).__name__}: {exc}", flush=True)

print()
print("=" * 78)
print("PART 2 -- the full tensor, best basis vs worst basis")
print("=" * 78)
for basis in ("6-31g*", "aug-cc-pvdz"):
    a = alpha_ff(basis)
    print(f"  {basis}:")
    print("   ", np.array2string(a, precision=4, suppress_small=True)
          .replace("\n", "\n    "))
    print(f"    diagonal {np.diag(a)[0]:.4f} {np.diag(a)[1]:.4f} "
          f"{np.diag(a)[2]:.4f}")
    print("    (water: x is out of plane, y and z are in the molecular plane;")
    print("     the three should be within a few per cent of each other)")
