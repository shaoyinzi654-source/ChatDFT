# PDC molecular dynamics workflow

The periodic PDC workflow in ChatDFT prepares GROMACS input, previews initial boxes, manages remote jobs, and analyzes trajectories. It is designed around the PDC aggregation study by Li and Van Lehn. [Paper](https://doi.org/10.1021/acsomega.4c05431) · [Supplement](https://ndownloader.figshare.com/files/49285305)

## Prepare input

1. Obtain the [authors’ original parameters at the pinned revision](https://github.com/jlianl/PDC_Aggregation/tree/fb868ea2646ecf98aa4faed389a5ef64528069a5). Place the source directory containing `PDC_neutral` and `PDC1_Monovalent` under `runtime/assets/pdc2024` and restart ChatDFT.
2. Open the PDC molecular dynamics tool in the AI menu and select a system. The workflow covers neutral PDC and singly or doubly deprotonated PDC with the supported monovalent and divalent counterions.
3. Start with a short verification run. The paper workflow uses 100 PDC molecules, a 6 nm initial cubic box, 293.15 K, 1 bar, 2 ns equilibration, and 100 ns production. Inspect the prepared files and simulation parameters before submission.
4. Review the provenance of the 12-6-4 ion parameters for the divalent cases. ChatDFT's literature reconstruction transfers published SPC/E kappa values to other atom-type pairs. The authors' original tabulated potentials were unavailable. Use compatible GROMACS settings for tabulated interactions.
5. Check the configured server engine and submit. Monitor stages and warnings. An interrupted run can resume from an available GROMACS checkpoint.

A separately supplied `pdc-paper-models.zip` can be placed in `runtime/assets` to preview the prepared boxes offline. The workbench can rotate the box and toggle the cell, water, and hydrogen display.

## Analyze results

- Download the simulation output and trajectory when needed. Preserve the run manifest, topology, command logs, MDP files, coordinates, and available checkpoints.
- Match atom count, residue and atom order, water, ions, box vectors, and time when importing a trajectory.
- Inspect aggregation and parallel alignment, temperature, pressure, energy, and density. Confirm units, time cutoffs, normalization, and histogram bins before comparing with published results.
- Export spatial distribution grids as Cube or NumPy arrays. The workflow can optionally save a separate AI review of selected parameters, summaries, and log excerpts.

## Reproduction details

- The 21 systems comprise neutral PDC plus each combination of singly or doubly deprotonated PDC with Li, Na, K, Rb, Cs, Be, Mg, Ca, Sr, or Ba. Equilibration uses V-rescale and Berendsen coupling. Production uses Nose–Hoover and Parrinello–Rahman coupling. The standard timestep is 2 fs. Consider 1 fs and regenerate the inputs if 2 fs produces numerical instability.
- The reconstruction uses the authors’ bond connectivity, a fixed random seed, and rigid SPC/E water. For Be, Mg, Ca, Sr, and Ba the transferred kappa values are 980, 98.7, 23, 13.7, and 7.2 nm⁻², respectively. The reconstruction uses C4 = kappa × C6 and Lorentz–Berthelot mixing for water oxygen cross terms. Alternatively, supply a per-type kappa JSON with units and provenance. [Li and Merz parameter source](https://pmc.ncbi.nlm.nih.gov/articles/PMC3960013/)
- The GROMACS 2018.8 user-table workflow uses group cutoffs. The paper reports Verlet. Nonbonded tables are shifted at 1.2 nm, extend to 3 nm, and include the 12-6-4 potential and its analytical negative derivative. The 1–4 table retains ordinary unshifted LJ terms. [GROMACS cutoff support](https://manual.gromacs.org/2018/user-guide/cutoff-schemes.html)
- Aggregation uses connected components of contacts within 0.31 nm under the minimum-image convention. A parallel edge additionally requires an acute angle of at most 20° between the C8/C9/C10 ring-plane normals. The implementation uses Rc = Nc/Lp and Mc = Nc × F(Nc), where F counts parallel clusters of the same size within a sampled frame. An aggregate without a multi-molecule parallel cluster contributes a zero record. Default analysis windows start at 20 ns for neutral and monovalent systems and 30 ns for divalent systems. Histogram outputs are unsmoothed.
- Density grids use particles/nm³. Compare the zero convention, bins, and normalization before interpreting differences from the paper. Check the run manifest for input hashes, actual molecule counts, engine version, commands, and preprocessing warnings.

The short verification run checks the software workflow. Full reproduction requires production trajectories, equilibration checks, and comparison against the published distributions.

[Workbench manual](README.md) · [Repository home](../README.md)
