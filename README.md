# -202609-PAR_pdb_partitioner

## PAR Monomer Structure Partitioning Tool

`adpr_partition.py` automatically identifies the five structural components of ADP-ribose from PDB bond topology, without relying on input atom serial numbers or ordering. All atoms in the output PDB belong to the same `PAR A 1` residue, so molecular viewers display one PAR monomer rather than five monomers.

## PDB Identity and Grouping

The PDB identity fields are standardized as follows: residue name `PAR`, chain ID `A`, residue number `1`, and segment ID `PAR`. The chain ID can be changed with `--chain`.

The tool divides the structure into five internal groups based on connectivity:

| Group Number | Group Name | Description |
|---:|---|---|
| 1 | `RBT` | Terminal ribose |
| 2 | `P01` | Terminal-side phosphate; the P–O–P bridging oxygen belongs to this group |
| 3 | `P02` | Adenosine-side phosphate |
| 4 | `RBA` | Adenosine ribose |
| 5 | `ADE` | Adenine base |

These groups are recorded in the `group_index` and `group_name` columns of the companion CSV file. The PDB `REMARK 900` records also describe the grouping rules. The traditional PDB format does not support nesting five “sub-residues” inside a single residue, so this representation preserves both a single PAR monomer and five-part atom assignments in a compatible way.

Atom names remain unique within the PAR residue. The two phosphorus atoms are named `P1` and `P2`; their non-bridging oxygen atoms are named `O1P1`/`O2P1` and `O1P2`/`O2P2`, respectively; and the bridging oxygen is named `OPB`. The two sets of ribose atom names end in `T` (terminal side) and `A` (adenosine side), respectively.

## Usage

Python 3.9 or later is required. No third-party packages are needed.

You can also use the command line:

`python adpr_partition.py /test/atom_order_1.pdb -o /test/PAR1.pdb`

To use another ADPR file as a reference and verify that differently ordered inputs have the same standardized bond topology.

This check compares the bond sets between standardized atom names and the atom counts in the five groups. It does not compare original serial numbers, input ordering, or three-dimensional orientation. If the topologies differ, the tool lists the missing or extra standardized bonds.

If the input contains `CONECT` records, the tool uses them preferentially. Otherwise, it infers bonds from element-specific covalent radii and interatomic distances. The tool validates the two phosphorus atoms, the P–O–P bridge, two five-membered ribose rings, and one adenine topology. If the structure does not match, it reports an error rather than silently generating an incorrect partition.
