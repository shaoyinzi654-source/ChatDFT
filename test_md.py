from md_engine import MolecularDynamics
import numpy as np

# Create MD system with Argon (64 atoms)
md = MolecularDynamics(box_length=12.0, temp=150.0, dt=0.001)
md.init_argon_box(n_atoms=32)

print("Starting molecular dynamics verification:")
print("Number of particles:", len(md.positions))
print("Initial temperature:", md.compute_kinetic_energy()[1], "K")

# Compute initial forces
v_pot = md.compute_forces()
ke, temp = md.compute_kinetic_energy()
e_total_initial = ke + v_pot

print("Initial energies (eV) -- Potential:", v_pot, "Kinetic:", ke, "Total:", e_total_initial)

# Run 100 Velocity Verlet steps
for step in range(100):
    md.integrate_verlet_step1()
    v_pot = md.compute_forces()
    md.integrate_verlet_step2()
    
    # Thermostat every 5 steps
    ke, temp = md.compute_kinetic_energy()
    if step % 10 == 0:
        md.apply_thermostat(temp)
        
ke_final, temp_final = md.compute_kinetic_energy()
e_total_final = ke_final + v_pot

print("Verification complete after 100 steps:")
print("Final temperature:", temp_final, "K")
print("Final energies (eV) -- Potential:", v_pot, "Kinetic:", ke_final, "Total:", e_total_final)
print("Energy drift (final - initial):", abs(e_total_final - e_total_initial), "eV")

# Compute RDF
r, g = md.calculate_rdf()
print("RDF computed successfully. Peak position of g(r):", r[np.argmax(g)], "A, Peak height:", np.max(g))
