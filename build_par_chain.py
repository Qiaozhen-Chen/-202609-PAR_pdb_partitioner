#!/usr/bin/env python3
"""Build a linear poly(ADP-ribose) chain from a single ADP-ribose PDB.

This is a PolyRapid-compatible chain-building helper for monomers that cannot be
represented by PolyRapid's original ``*...*`` SMILES convention.  It forms the
linear PAR 2'-to-1'' O-glycosidic linkage:

    O2A(unit i) -- C1T(unit i+1)

At every junction, the hydrogen on O2A of unit i is removed if it is explicit,
and O1T (plus any hydrogen bonded to it) is removed from unit i+1.  The supplied
PAR1.pdb contains heavy atoms only, so in that file only O1T is deleted.

The new monomer is placed rigidly so that its former C1T--O1T direction becomes
the new C1T--O2A direction.  With --twist auto, torsions are sampled and the
least-clashing placement is selected deterministically.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Atom:
    index: int
    record: str
    atom_field: str
    name: str
    altloc: str
    resname: str
    chain: str
    resseq: int
    icode: str
    xyz: np.ndarray
    occupancy: float
    tempfactor: float
    segid: str
    element: str
    charge: str
    input_serial: int


def _as_float(text: str, default: float) -> float:
    try:
        return float(text)
    except ValueError:
        return default


def infer_element(atom_field: str) -> str:
    stripped = atom_field.strip()
    letters = "".join(c for c in stripped if c.isalpha())
    if not letters:
        raise ValueError(f"Cannot infer element from atom name {atom_field!r}")
    # PDB atom names here are C1T/O2A/etc.; their first letter is the element.
    return letters[0].upper()


def read_pdb(path: Path) -> tuple[list[Atom], set[tuple[int, int]]]:
    atoms: list[Atom] = []
    serial_to_index: dict[int, int] = {}
    raw_conect: list[tuple[int, int]] = []

    for line in path.read_text().splitlines():
        rec = line[0:6].strip().upper()
        if rec in {"ATOM", "HETATM"}:
            padded = line.ljust(80)
            serial = int(padded[6:11])
            atom_field = padded[12:16]
            element = padded[76:78].strip().upper() or infer_element(atom_field)
            atom = Atom(
                index=len(atoms),
                record=rec,
                atom_field=atom_field,
                name=atom_field.strip(),
                altloc=padded[16],
                resname=padded[17:20].strip() or "PAR",
                chain=padded[21].strip() or "A",
                resseq=int(padded[22:26] or 1),
                icode=padded[26],
                xyz=np.array(
                    [float(padded[30:38]), float(padded[38:46]), float(padded[46:54])],
                    dtype=float,
                ),
                occupancy=_as_float(padded[54:60].strip(), 1.0),
                tempfactor=_as_float(padded[60:66].strip(), 0.0),
                segid=padded[72:76].strip(),
                element=element,
                charge=padded[78:80].strip(),
                input_serial=serial,
            )
            if serial in serial_to_index:
                raise ValueError(f"Duplicate atom serial {serial}")
            serial_to_index[serial] = atom.index
            atoms.append(atom)
        elif rec == "CONECT":
            fields = line.split()[1:]
            if len(fields) >= 2:
                source = int(fields[0])
                raw_conect.extend((source, int(target)) for target in fields[1:])

    if not atoms:
        raise ValueError(f"No ATOM/HETATM records found in {path}")
    if not raw_conect:
        raise ValueError("Input PDB has no CONECT records; explicit monomer bonds are required")

    bonds: set[tuple[int, int]] = set()
    for serial_a, serial_b in raw_conect:
        if serial_a not in serial_to_index or serial_b not in serial_to_index:
            raise ValueError(f"CONECT references unknown serial: {serial_a} {serial_b}")
        a, b = serial_to_index[serial_a], serial_to_index[serial_b]
        if a != b:
            bonds.add((min(a, b), max(a, b)))
    return atoms, bonds


def adjacency(n_atoms: int, bonds: Iterable[tuple[int, int]]) -> list[set[int]]:
    neighbors = [set() for _ in range(n_atoms)]
    for a, b in bonds:
        neighbors[a].add(b)
        neighbors[b].add(a)
    return neighbors


def resolve_atom(atoms: list[Atom], selector: str, role: str) -> int:
    """Resolve an atom by exact name or original PDB serial number."""
    matches: list[int]
    if selector.isdigit():
        matches = [a.index for a in atoms if a.input_serial == int(selector)]
    else:
        matches = [a.index for a in atoms if a.name.upper() == selector.upper()]
    if len(matches) != 1:
        detail = "none" if not matches else f"{len(matches)} matches"
        raise ValueError(f"Could not uniquely resolve {role} selector {selector!r}: {detail}")
    return matches[0]


def unit(vector: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-10:
        raise ValueError(f"Cannot normalize zero-length vector for {label}")
    return vector / norm


def perpendicular(vector: np.ndarray, axis: np.ndarray, label: str) -> np.ndarray:
    projected = vector - np.dot(vector, axis) * axis
    return unit(projected, label)


def frame(e1: np.ndarray, e2_hint: np.ndarray, label: str) -> np.ndarray:
    a = unit(e1, f"{label} axis 1")
    b = perpendicular(e2_hint, a, f"{label} axis 2")
    c = unit(np.cross(a, b), f"{label} axis 3")
    # Recompute b so the basis is exactly right-handed and orthonormal.
    b = np.cross(c, a)
    return np.column_stack((a, b, c))


def axis_rotation(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    x, y, z = unit(axis, "twist axis")
    angle = math.radians(angle_deg)
    c, s, q = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array(
        [
            [c + x*x*q, x*y*q - z*s, x*z*q + y*s],
            [y*x*q + z*s, c + y*y*q, y*z*q - x*s],
            [z*x*q - y*s, z*y*q + x*s, c + z*z*q],
        ]
    )


def choose_heavy_neighbor(
    atom_index: int,
    exclude: set[int],
    atoms: list[Atom],
    neighbors: list[set[int]],
    role: str,
) -> int:
    candidates = [
        i for i in sorted(neighbors[atom_index])
        if i not in exclude and atoms[i].element != "H"
    ]
    if not candidates:
        raise ValueError(f"No suitable heavy-atom neighbor found for {role} ({atoms[atom_index].name})")
    return candidates[0]


def min_cross_distance(new_xyz: np.ndarray, old_xyz: np.ndarray) -> float:
    if not len(old_xyz) or not len(new_xyz):
        return math.inf
    delta = new_xyz[:, None, :] - old_xyz[None, :, :]
    return float(np.sqrt(np.sum(delta * delta, axis=2)).min())


def format_atom_line(atom: Atom, serial: int, resseq: int, xyz: np.ndarray) -> str:
    if serial > 99999:
        raise ValueError("PDB atom serial exceeds 99999; use fewer monomers or another format")
    if resseq > 9999:
        raise ValueError("PDB residue number exceeds 9999")
    return (
        f"{atom.record:<6}{serial:5d} {atom.atom_field:4s}{atom.altloc:1s}"
        f"{atom.resname:>3s} {atom.chain:1s}{resseq:4d}{atom.icode:1s}   "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
        f"{atom.occupancy:6.2f}{atom.tempfactor:6.2f}      "
        f"{atom.segid:<4s}{atom.element:>2s}{atom.charge:>2s}"
    )


def write_pdb(
    output: Path,
    atoms: list[Atom],
    copies: list[dict[int, np.ndarray]],
    template_bonds: set[tuple[int, int]],
    donor_o: int,
    acceptor_c: int,
    deletion_sets: list[set[int]],
    input_name: str,
) -> None:
    serial_map: dict[tuple[int, int], int] = {}
    lines = [
        f"REMARK 900 LINEAR PAR BUILT FROM {input_name}",
        f"REMARK 900 MONOMERS {len(copies)}; LINK O2A(i)-C1T(i+1)",
        "REMARK 900 DOWNSTREAM O1T REMOVED AT EACH LINK; EXPLICIT LINK H ALSO REMOVED",
    ]

    serial = 1
    for copy_index, coords in enumerate(copies):
        for atom_index in sorted(coords):
            serial_map[(copy_index, atom_index)] = serial
            lines.append(format_atom_line(atoms[atom_index], serial, copy_index + 1, coords[atom_index]))
            serial += 1

    all_bonds: set[tuple[int, int]] = set()
    for copy_index in range(len(copies)):
        deleted = deletion_sets[copy_index]
        for a, b in template_bonds:
            if a not in deleted and b not in deleted:
                sa = serial_map[(copy_index, a)]
                sb = serial_map[(copy_index, b)]
                all_bonds.add((min(sa, sb), max(sa, sb)))
    for copy_index in range(len(copies) - 1):
        sa = serial_map[(copy_index, donor_o)]
        sb = serial_map[(copy_index + 1, acceptor_c)]
        all_bonds.add((min(sa, sb), max(sa, sb)))

    bond_neighbors: dict[int, list[int]] = {i: [] for i in range(1, serial)}
    for a, b in sorted(all_bonds):
        bond_neighbors[a].append(b)
        bond_neighbors[b].append(a)
    for source in sorted(bond_neighbors):
        targets = sorted(bond_neighbors[source])
        for offset in range(0, len(targets), 4):
            lines.append("CONECT" + f"{source:5d}" + "".join(f"{t:5d}" for t in targets[offset:offset + 4]))
    lines.append("END")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def build(args: argparse.Namespace) -> None:
    atoms, bonds = read_pdb(args.input)
    neighbors = adjacency(len(atoms), bonds)

    donor_o = resolve_atom(atoms, args.donor_o, "donor oxygen")
    acceptor_c = resolve_atom(atoms, args.acceptor_c, "acceptor carbon")
    leaving_o = resolve_atom(atoms, args.leaving_o, "leaving oxygen")

    if atoms[donor_o].element != "O":
        raise ValueError(f"Donor atom {atoms[donor_o].name} is not oxygen")
    if atoms[acceptor_c].element != "C":
        raise ValueError(f"Acceptor atom {atoms[acceptor_c].name} is not carbon")
    if atoms[leaving_o].element != "O":
        raise ValueError(f"Leaving atom {atoms[leaving_o].name} is not oxygen")
    if leaving_o not in neighbors[acceptor_c]:
        raise ValueError(f"{atoms[leaving_o].name} is not bonded to {atoms[acceptor_c].name}")

    donor_h = {i for i in neighbors[donor_o] if atoms[i].element == "H"}
    leaving_group = {leaving_o} | {
        i for i in neighbors[leaving_o] if atoms[i].element == "H"
    }
    donor_parent = choose_heavy_neighbor(
        donor_o, set(), atoms, neighbors, "donor oxygen parent"
    )
    donor_side = choose_heavy_neighbor(
        donor_parent, {donor_o}, atoms, neighbors, "donor frame"
    )
    acceptor_side = choose_heavy_neighbor(
        acceptor_c, {leaving_o}, atoms, neighbors, "acceptor frame"
    )

    xyz0 = np.vstack([a.xyz for a in atoms])
    # The O-H vector is ideal when hydrogens exist.  For PAR1.pdb (heavy atoms
    # only), extend the parent-C -> donor-O bond to approximate the missing O-H.
    if donor_h:
        donor_direction_local = xyz0[min(donor_h)] - xyz0[donor_o]
    else:
        donor_direction_local = xyz0[donor_o] - xyz0[donor_parent]

    source_frame = frame(
        xyz0[leaving_o] - xyz0[acceptor_c],
        xyz0[acceptor_side] - xyz0[acceptor_c],
        "acceptor",
    )

    deletion_sets: list[set[int]] = []
    for i in range(args.monomers):
        deleted: set[int] = set()
        if i < args.monomers - 1:
            deleted |= donor_h
        if i > 0:
            deleted |= leaving_group
        deletion_sets.append(deleted)

    rotations = [np.eye(3)]
    translations = [np.zeros(3)]
    copies: list[dict[int, np.ndarray]] = [
        {i: xyz0[i].copy() for i in range(len(atoms)) if i not in deletion_sets[0]}
    ]
    chosen_twists: list[float] = []
    closest_contacts: list[float] = []

    for copy_index in range(1, args.monomers):
        prev_r, prev_t = rotations[-1], translations[-1]
        donor_xyz = prev_r @ xyz0[donor_o] + prev_t
        donor_parent_xyz = prev_r @ xyz0[donor_parent] + prev_t
        donor_side_xyz = prev_r @ xyz0[donor_side] + prev_t
        if donor_h:
            donor_direction = prev_r @ donor_direction_local
        else:
            donor_direction = donor_xyz - donor_parent_xyz
        outward = unit(donor_direction, "donor outgoing direction")
        target_c = donor_xyz + args.bond_length * outward
        target_e1 = -outward  # new C1T -> previous O2A
        target_e2_hint = donor_side_xyz - donor_parent_xyz
        target_frame = frame(target_e1, target_e2_hint, "donor")
        base_rotation = target_frame @ source_frame.T

        if args.twist == "auto":
            candidates = [float(x) for x in range(0, 360, args.twist_step)]
        else:
            candidates = [float(args.twist)]

        best: tuple[float, float, np.ndarray, np.ndarray, dict[int, np.ndarray]] | None = None
        for angle in candidates:
            rotation = axis_rotation(target_e1, angle) @ base_rotation
            translation = target_c - rotation @ xyz0[acceptor_c]
            coords = {
                i: rotation @ xyz0[i] + translation
                for i in range(len(atoms)) if i not in deletion_sets[copy_index]
            }
            # Exclude the newly bonded atoms from the clash score; their 1.43 A
            # separation is intentional and would otherwise dominate every score.
            score_new = np.vstack([v for i, v in coords.items() if i != acceptor_c])
            score_old = np.vstack([
                point
                for old_copy_index, old_copy in enumerate(copies)
                for atom_index, point in old_copy.items()
                if not (old_copy_index == copy_index - 1 and atom_index == donor_o)
            ])
            score = min_cross_distance(score_new, score_old)
            candidate = (score, -angle, rotation, translation, coords)
            if best is None or candidate[:2] > best[:2]:
                best = candidate

        assert best is not None
        score, negative_angle, rotation, translation, coords = best
        rotations.append(rotation)
        translations.append(translation)
        copies.append(coords)
        chosen_twists.append(-negative_angle)
        closest_contacts.append(score)

    write_pdb(
        args.output,
        atoms,
        copies,
        bonds,
        donor_o,
        acceptor_c,
        deletion_sets,
        args.input.name,
    )

    expected_atoms = sum(len(atoms) - len(deleted) for deleted in deletion_sets)
    print(f"Built {args.monomers} PAR monomers -> {args.output}")
    print(f"Atoms: {expected_atoms}; inter-monomer bonds: {args.monomers - 1}")
    print(
        f"Link: {atoms[donor_o].name}(i)-{atoms[acceptor_c].name}(i+1), "
        f"bond length {args.bond_length:.3f} A; leaving group: {atoms[leaving_o].name}"
    )
    if chosen_twists:
        print("Selected twist angles (deg): " + ", ".join(f"{x:g}" for x in chosen_twists))
        print("Closest sampled nonbonded contacts (A): " + ", ".join(f"{x:.2f}" for x in closest_contacts))
        if min(closest_contacts) < args.clash_warning:
            print(
                f"WARNING: a nonbonded contact is below {args.clash_warning:.2f} A; "
                "inspect/minimize the structure before simulation."
            )


def print_atoms(path: Path) -> None:
    atoms, bonds = read_pdb(path)
    neighbors = adjacency(len(atoms), bonds)
    print("serial  name element  bonded-to")
    for atom in atoms:
        linked = ",".join(
            f"{atoms[i].name}({atoms[i].input_serial})" for i in sorted(neighbors[atom.index])
        )
        print(f"{atom.input_serial:6d}  {atom.name:<4s} {atom.element:>3s}      {linked}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build linear poly(ADP-ribose) through O2A(i)-C1T(i+1) links."
    )
    parser.add_argument("input", type=Path, help="single ADP-ribose PDB with CONECT records")
    parser.add_argument("-n", "--monomers", type=int, default=10, help="number of monomers (default: 10)")
    parser.add_argument("-o", "--output", type=Path, default=Path("PAR_chain.pdb"), help="output PDB")
    parser.add_argument("--donor-o", default="O2A", help="2'-oxygen atom name or PDB serial (default: O2A)")
    parser.add_argument("--acceptor-c", default="C1T", help="1''-carbon atom name or PDB serial (default: C1T)")
    parser.add_argument("--leaving-o", default="O1T", help="leaving hydroxyl O name or serial (default: O1T)")
    parser.add_argument("--bond-length", type=float, default=1.43, help="new O-C distance in A (default: 1.43)")
    parser.add_argument(
        "--twist", default="auto",
        help="junction twist in degrees, or 'auto' for clash sampling (default: auto)",
    )
    parser.add_argument("--twist-step", type=int, default=30, help="auto twist sampling step in degrees")
    parser.add_argument("--clash-warning", type=float, default=0.80, help="warn below this nonbonded distance in A")
    parser.add_argument("--list-atoms", action="store_true", help="show atom names/serials/connectivity and exit")
    args = parser.parse_args()
    if args.monomers < 1:
        parser.error("--monomers must be >= 1")
    if args.bond_length <= 0:
        parser.error("--bond-length must be positive")
    if not (1 <= args.twist_step <= 180):
        parser.error("--twist-step must be between 1 and 180")
    if args.twist != "auto":
        try:
            float(args.twist)
        except ValueError:
            parser.error("--twist must be a number or 'auto'")
    return args


def main() -> None:
    args = parse_args()
    if args.list_atoms:
        print_atoms(args.input)
    else:
        build(args)


if __name__ == "__main__":
    main()
