from diatomic_engine import solve_multi_atom_scf, compute_bond_orders
import numpy as np

atoms_co = [{'name':'C','pos':[0,0,-1.06]},{'name':'O','pos':[0,0,1.06]}]
r = solve_multi_atom_scf(atoms_co, 14)

basis = r['basis']
P = r['P']
S = r['S']
PS = np.dot(P, S)

print("P diagonal:", np.diag(P))
print("S diagonal:", np.diag(S))
print("PS diagonal:", np.diag(PS))
print()
print("P matrix:")
print(np.round(P, 4))
print()

# Atom orbital mapping
atom_orbitals = []
for at in atoms_co:
    pos = np.array(at['pos'], dtype=float)
    orbs = []
    for p in range(len(basis)):
        if np.allclose(basis[p].center, pos, atol=1e-3):
            orbs.append(p)
    atom_orbitals.append(orbs)
    print(f"Atom {at['name']} orbital indices: {orbs}")

# Compute Wiberg bond index between atom 0 (C) and atom 1 (O)
w_idx = 0.0
for mu in atom_orbitals[0]:
    for nu in atom_orbitals[1]:
        contribution = PS[mu, nu] * PS[nu, mu]
        print(f"  PS[{mu},{nu}]={PS[mu,nu]:.6f}, PS[{nu},{mu}]={PS[nu,mu]:.6f}, product={contribution:.8f}")
        w_idx += contribution

print(f"\nWiberg bond order C-O: {w_idx}")
print(f"(Should be > 1 since CO has triple bond)")

# Also try simpler: total electron sharing
print("\nSimpler Mulliken overlap populations:")
for mu in atom_orbitals[0]:
    for nu in atom_orbitals[1]:
        mop = 2.0 * P[mu, nu] * S[mu, nu]
        print(f"  2*P[{mu},{nu}]*S[{mu},{nu}] = {mop:.6f}")
