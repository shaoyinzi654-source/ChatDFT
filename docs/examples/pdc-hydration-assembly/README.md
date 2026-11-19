# PDC–magnesium hydration assembly

[Download the ChatDFT project](pdc-hydration.chatdft.json?raw=1) · [Download XYZ](pdc-hydration.xyz?raw=1)

This example is a nonperiodic starting geometry with **186 atoms and 42 fragments**, a net charge of zero, and a spin multiplicity of one. ChatDFT retains the fragment identities, bonds, and assembly recipe in the project file. Coordinates are in ångströms.

| Component | Count | Formal charge per fragment |
| --- | ---: | ---: |
| PDC dianion | 6 | −2 |
| Mg²⁺ | 6 | +2 |
| H₂O | 30 | 0 |

## Placement and checks

| Fragment indices | Verified starting geometry |
| --- | --- |
| PDC 2 to 1 and 4 to 3 | Two pairs with 3.60 Å normal separations between fitted heavy-atom planes. |
| Mg 7–11 to PDC 1–5 | Each selected Mg is 2.25 Å from the fourth oxygen of its paired PDC. |
| H₂O 13–17 to Mg 7–11 | Each selected water oxygen is 2.30 Å from its paired Mg. |
| Remaining fragments | Initial positions from the assembly search, screened for overlap. |

1. Open the project from the File menu. Check the atom count, charge, and spin in the structure properties.
2. Use the comparison layout to inspect the same geometry in ball-and-stick and space-filling views. Measure contacts using the fragment and atom assignments above.
3. Choose the calculation task, level of theory, solvent if applicable, and computing resources. This example supplies a starting structure for a separate calculation.

[Repository home](../../../README.md) · [Completed PySCF example](../../../README.md#completed-calculation) · [Periodic PDC workflow](../../../standalone/PDC-MD-GUIDE.md)
