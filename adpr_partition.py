#!/usr/bin/env python3
"""Partition an ADP-ribose graph into five groups inside one PAR PDB residue.

The partition is derived from the molecular graph, not from input atom serials:
RBT (terminal ribose), P01, P02, RBA (adenosine ribose), ADE (adenine).
Every output atom belongs to the single PDB residue ``PAR A 1``.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass
class Atom:
    serial: int
    name: str
    element: str
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    bfactor: float = 0.0
    altloc: str = " "
    icode: str = " "
    charge: str = ""


class PartitionError(ValueError):
    pass


COVALENT_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "P": 1.07, "S": 1.05,
}


def parse_pdb(path: Path) -> Tuple[Dict[int, Atom], Dict[int, Set[int]], bool]:
    atoms: Dict[int, Atom] = {}
    graph: Dict[int, Set[int]] = {}
    saw_conect = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            record = line[:6].strip().upper()
            if record in {"ATOM", "HETATM"}:
                try:
                    serial = int(line[6:11])
                    raw_name = line[12:16].strip()
                    element = line[76:78].strip().upper()
                    if not element:
                        element = "".join(c for c in raw_name if c.isalpha())[:1].upper()
                    atom = Atom(
                        serial=serial,
                        name=raw_name,
                        element=element,
                        x=float(line[30:38]), y=float(line[38:46]), z=float(line[46:54]),
                        occupancy=float(line[54:60].strip() or 1.0),
                        bfactor=float(line[60:66].strip() or 0.0),
                        altloc=(line[16:17] or " "), icode=(line[26:27] or " "),
                        charge=line[78:80].strip(),
                    )
                except (ValueError, IndexError) as exc:
                    raise PartitionError(f"Line {line_no} is not a valid PDB atom record: {exc}") from exc
                if serial in atoms:
                    raise PartitionError(f"Duplicate atom serial number: {serial}")
                atoms[serial] = atom
                graph[serial] = set()
            elif record == "CONECT":
                saw_conect = True
                fields = line[6:].split()
                if len(fields) >= 2:
                    try:
                        source = int(fields[0])
                        for field in fields[1:]:
                            target = int(field)
                            graph.setdefault(source, set()).add(target)
                            graph.setdefault(target, set()).add(source)
                    except ValueError as exc:
                        raise PartitionError(f"Invalid CONECT record on line {line_no}") from exc
    if not atoms:
        raise PartitionError("The input file contains no ATOM/HETATM records")
    for serial, neighbors in graph.items():
        if serial not in atoms or any(n not in atoms for n in neighbors):
            raise PartitionError("A CONECT record references a nonexistent atom serial number")
    return atoms, graph, saw_conect


def infer_bonds(atoms: Dict[int, Atom]) -> Dict[int, Set[int]]:
    graph = {serial: set() for serial in atoms}
    items = list(atoms.items())
    for i, (sa, a) in enumerate(items):
        ra = COVALENT_RADII.get(a.element, 0.77)
        for sb, b in items[i + 1:]:
            rb = COVALENT_RADII.get(b.element, 0.77)
            d = math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))
            if 0.4 <= d <= ra + rb + 0.45:
                graph[sa].add(sb)
                graph[sb].add(sa)
    return graph


def simple_cycles_of_five(graph: Dict[int, Set[int]], allowed: Set[int]) -> Set[frozenset[int]]:
    cycles: Set[frozenset[int]] = set()
    for start in allowed:
        stack: List[Tuple[int, List[int]]] = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if len(path) == 5:
                if start in graph[node]:
                    cycles.add(frozenset(path))
                continue
            for nxt in graph[node]:
                if nxt in allowed and nxt not in path and nxt >= start:
                    stack.append((nxt, path + [nxt]))
    return cycles


def find_sugar_from_phosphate(
    p: int, atoms: Dict[int, Atom], graph: Dict[int, Set[int]], bridge_o: int
) -> Tuple[Set[int], Dict[str, int], Optional[int]]:
    link_oxygens = [
        n for n in graph[p]
        if n != bridge_o and atoms[n].element == "O"
        and any(atoms[x].element == "C" for x in graph[n])
    ]
    if len(link_oxygens) != 1:
        raise PartitionError(
            f"Phosphorus atom {p} should have exactly one oxygen bonded to a ribose carbon; "
            f"found {len(link_oxygens)}"
        )
    o5 = link_oxygens[0]
    c5s = [n for n in graph[o5] if atoms[n].element == "C"]
    if len(c5s) != 1:
        raise PartitionError("Could not identify the ribose C5 atom")
    c5 = c5s[0]
    c4s = [n for n in graph[c5] if atoms[n].element == "C"]
    if len(c4s) != 1:
        raise PartitionError("Could not identify the ribose C4 atom")
    c4 = c4s[0]

    allowed = {s for s, a in atoms.items() if a.element in {"C", "O"}}
    candidate_cycles = [
        cycle for cycle in simple_cycles_of_five(graph, allowed)
        if c4 in cycle
        and sum(atoms[s].element == "O" for s in cycle) == 1
        and sum(atoms[s].element == "C" for s in cycle) == 4
    ]
    if len(candidate_cycles) != 1:
        raise PartitionError(
            f"Expected one five-membered ribose ring around C4 atom {c4}; "
            f"found {len(candidate_cycles)}"
        )
    ring = set(candidate_cycles[0])
    o4 = next(s for s in ring if atoms[s].element == "O")
    c3_candidates = [n for n in graph[c4] if n in ring and atoms[n].element == "C"]
    if len(c3_candidates) != 1:
        raise PartitionError("Could not identify C3 in the ribose ring")
    c3 = c3_candidates[0]
    c2 = next((n for n in graph[c3] if n in ring and atoms[n].element == "C" and n != c4), None)
    if c2 is None:
        raise PartitionError("Could not identify C2 in the ribose ring")
    c1 = next((n for n in graph[c2] if n in ring and atoms[n].element == "C" and n != c3), None)
    if c1 is None or o4 not in graph[c1]:
        raise PartitionError("Could not identify C1/O4 in the ribose ring")

    names: Dict[str, int] = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "O4": o4, "O5": o5}
    sugar = set(ring) | {c5, o5}
    for cname, carbon in (("O1", c1), ("O2", c2), ("O3", c3)):
        oxy = [n for n in graph[carbon] if atoms[n].element == "O" and n not in ring]
        if len(oxy) > 1:
            raise PartitionError(f"Multiple oxygen atoms found at ribose position {cname}")
        if oxy:
            names[cname] = oxy[0]
            sugar.add(oxy[0])
    base_n = next((n for n in graph[c1] if atoms[n].element == "N"), None)
    return sugar, names, base_n


def connected_component(start: int, graph: Dict[int, Set[int]], blocked: Set[int]) -> Set[int]:
    seen: Set[int] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in seen or node in blocked:
            continue
        seen.add(node)
        stack.extend(graph[node] - seen - blocked)
    return seen


def name_adenine(base: Set[int], glyco_n: int, atoms: Dict[int, Atom], graph: Dict[int, Set[int]]) -> Dict[int, str]:
    def inside(node: int, element: Optional[str] = None) -> List[int]:
        return [n for n in graph[node] if n in base and (element is None or atoms[n].element == element)]

    result = {glyco_n: "N9"}
    n9_c = inside(glyco_n, "C")
    if len(n9_c) != 2:
        raise PartitionError("The topology around adenine N9 does not match expectations")
    c8 = next((n for n in n9_c if len(inside(n)) == 2), None)
    c4 = next((n for n in n9_c if n != c8), None)
    if c8 is None or c4 is None:
        raise PartitionError("Could not distinguish adenine C8 from C4")
    result[c8], result[c4] = "C8", "C4"
    n7 = next((n for n in inside(c8, "N") if n != glyco_n), None)
    c5 = next((n for n in inside(n7, "C") if n != c8), None) if n7 else None
    n3 = next((n for n in inside(c4, "N") if n != glyco_n), None)
    c2 = next((n for n in inside(n3, "C") if n != c4), None) if n3 else None
    n1 = next((n for n in inside(c2, "N") if n != n3), None) if c2 else None
    c6 = next((n for n in inside(n1, "C") if n != c2), None) if n1 else None
    n6 = next((n for n in inside(c6, "N") if n != n1), None) if c6 else None
    expected = [(n7, "N7"), (c5, "C5"), (n3, "N3"), (c2, "C2"), (n1, "N1"), (c6, "C6"), (n6, "N6")]
    if any(node is None for node, _ in expected):
        raise PartitionError("The adenine ring topology is incomplete")
    result.update({node: name for node, name in expected if node is not None})
    if set(result) != base:
        extra = sorted(base - set(result))
        raise PartitionError(f"Adenine should contain 10 heavy atoms; unrecognized atoms: {extra}")
    return result


def partition(atoms: Dict[int, Atom], graph: Dict[int, Set[int]]) -> Tuple[List[Tuple[str, Set[int], Dict[int, str], str]], int]:
    phosphorus = [s for s, a in atoms.items() if a.element == "P"]
    if len(phosphorus) != 2:
        raise PartitionError(f"ADPR should contain 2 phosphorus atoms; found {len(phosphorus)}")
    p_a, p_b = phosphorus
    bridge = [
        s for s, a in atoms.items()
        if a.element == "O" and p_a in graph[s] and p_b in graph[s]
    ]
    if len(bridge) != 1:
        raise PartitionError(f"Expected 1 P-O-P bridging oxygen; found {len(bridge)}")
    bridge_o = bridge[0]
    sugar_a, sugar_names_a, base_n_a = find_sugar_from_phosphate(p_a, atoms, graph, bridge_o)
    sugar_b, sugar_names_b, base_n_b = find_sugar_from_phosphate(p_b, atoms, graph, bridge_o)
    if (base_n_a is None) == (base_n_b is None):
        raise PartitionError("Exactly one terminal ribose should be linked to adenine through N9")
    if base_n_a is None:
        terminal_p, terminal_sugar, terminal_names = p_a, sugar_a, sugar_names_a
        adeno_p, adeno_sugar, adeno_names, glyco_n = p_b, sugar_b, sugar_names_b, base_n_b
    else:
        terminal_p, terminal_sugar, terminal_names = p_b, sugar_b, sugar_names_b
        adeno_p, adeno_sugar, adeno_names, glyco_n = p_a, sugar_a, sugar_names_a, base_n_a
    assert glyco_n is not None

    blocked = terminal_sugar | adeno_sugar | {p_a, p_b, bridge_o}
    base = connected_component(glyco_n, graph, blocked)
    base_names = name_adenine(base, glyco_n, atoms, graph)

    terminal_phosphate = {terminal_p, bridge_o} | {
        n for n in graph[terminal_p] if atoms[n].element == "O" and n not in terminal_sugar
    }
    adeno_phosphate = {adeno_p} | {
        n for n in graph[adeno_p]
        if atoms[n].element == "O" and n not in adeno_sugar and n != bridge_o
    }
    # The shared P-O-P oxygen is assigned once, to the terminal-side phosphate P01.
    p01_names = {terminal_p: "P1", bridge_o: "OPB"}
    free1 = sorted(terminal_phosphate - set(p01_names))
    p01_names.update({s: f"O{i + 1}P1" for i, s in enumerate(free1)})
    p02_names = {adeno_p: "P2"}
    free2 = sorted(adeno_phosphate - {adeno_p})
    p02_names.update({s: f"O{i + 1}P2" for i, s in enumerate(free2)})
    rbt_names = {s: name + "T" for name, s in terminal_names.items()}
    rba_names = {s: name + "A" for name, s in adeno_names.items()}

    groups = [
        ("RBT", terminal_sugar, rbt_names, "terminal ribose"),
        ("P01", terminal_phosphate, p01_names, "terminal-side phosphate"),
        ("P02", adeno_phosphate, p02_names, "adenosine-side phosphate"),
        ("RBA", adeno_sugar, rba_names, "adenosine ribose"),
        ("ADE", base, base_names, "adenine base"),
    ]
    assigned: Set[int] = set()
    for name, members, _, _ in groups:
        overlap = assigned & members
        if overlap:
            raise PartitionError(f"Group {name} overlaps an earlier group at atoms {sorted(overlap)}")
        assigned |= members
    if assigned != set(atoms):
        raise PartitionError(f"The partition does not cover all atoms; unassigned: {sorted(set(atoms) - assigned)}")
    return groups, bridge_o


def pdb_atom_name(name: str, element: str) -> str:
    name = name[:4]
    if len(element) == 1 and len(name) < 4:
        return f" {name:<3}"
    return f"{name:<4}"


def write_outputs(
    input_path: Path, output_path: Path, csv_path: Path,
    atoms: Dict[int, Atom], graph: Dict[int, Set[int]],
    groups: List[Tuple[str, Set[int], Dict[int, str], str]], bridge_o: int,
    chain: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ordered: List[Tuple[int, int, str, str, str]] = []
    # Each residue is written contiguously; within it, chemically named atoms are sorted naturally.
    for resseq, (resname, members, atom_names, description) in enumerate(groups, 1):
        for old_serial in sorted(members, key=lambda s: (atom_names[s], s)):
            ordered.append((old_serial, resseq, resname, atom_names[old_serial], description))
    old_to_new = {old: i for i, (old, *_rest) in enumerate(ordered, 1)}

    lines = [
        "REMARK 900 ONE MONOMER: RESIDUE PAR, CHAIN " + chain[:1] + ", RESSEQ 1",
        "REMARK 900 TOPOLOGY GROUPS 1 RBT, 2 P01, 3 P02, 4 RBA, 5 ADE",
        "REMARK 900 GROUP MEMBERSHIP IS RECORDED IN THE COMPANION CSV",
        f"REMARK 900 P-O-P BRIDGE O (INPUT SERIAL {bridge_o}) IS ASSIGNED TO P01",
        f"REMARK 900 SOURCE {input_path.name}",
    ]
    csv_rows = []
    for new_serial, (old_serial, group_index, group_name, atom_name, description) in enumerate(ordered, 1):
        atom = atoms[old_serial]
        line = (
            f"HETATM{new_serial:5d} {pdb_atom_name(atom_name, atom.element)}{atom.altloc[:1]}"
            f"{'PAR':>3} {chain[:1]}{1:4d}{atom.icode[:1]}   "
            f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
            f"{atom.occupancy:6.2f}{atom.bfactor:6.2f}      "
            f"{'PAR':<4}{atom.element:>2}{atom.charge:>2}"
        )
        lines.append(line)
        csv_rows.append({
            "output_serial": new_serial, "input_serial": old_serial,
            "atom_name": atom_name, "element": atom.element,
            "pdb_residue_index": 1, "pdb_residue_name": "PAR",
            "group_index": group_index, "group_name": group_name,
            "chain": chain[:1], "segment_id": "PAR", "description": description,
        })
    for old_serial, *_ in ordered:
        new_serial = old_to_new[old_serial]
        neighbors = sorted(old_to_new[n] for n in graph[old_serial])
        for start in range(0, len(neighbors), 4):
            lines.append(f"CONECT{new_serial:5d}" + "".join(f"{n:5d}" for n in neighbors[start:start + 4]))
    lines.append("END")
    output_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


def topology_signature(
    atoms: Dict[int, Atom], graph: Dict[int, Set[int]]
) -> Tuple[Set[Tuple[str, str]], Tuple[Tuple[str, int], ...]]:
    """Return a serial/order-independent signature using standardized atom names."""
    groups, _bridge = partition(atoms, graph)
    standardized: Dict[int, str] = {}
    group_sizes: List[Tuple[str, int]] = []
    for group_name, members, atom_names, _description in groups:
        standardized.update(atom_names)
        group_sizes.append((group_name, len(members)))
    bonds: Set[Tuple[str, str]] = set()
    for source, neighbors in graph.items():
        for target in neighbors:
            if source < target:
                bonds.add(tuple(sorted((standardized[source], standardized[target]))))
    return bonds, tuple(group_sizes)


def load_graph(path: Path) -> Tuple[Dict[int, Atom], Dict[int, Set[int]]]:
    if not path.is_file():
        raise PartitionError(f"PDB file not found: {path}")
    atoms, graph, saw_conect = parse_pdb(path)
    if not saw_conect or not any(graph.values()):
        graph = infer_bonds(atoms)
    return atoms, graph


def compare_topology(subject: Path, reference: Path) -> None:
    """Raise PartitionError unless two ADPR files normalize to the same graph."""
    if not reference.is_file():
        raise PartitionError(f"Reference file not found: {reference}")
    subject_atoms, subject_graph = load_graph(subject)
    reference_atoms, reference_graph = load_graph(reference)
    subject_sig = topology_signature(subject_atoms, subject_graph)
    reference_sig = topology_signature(reference_atoms, reference_graph)
    if subject_sig != reference_sig:
        subject_bonds, subject_groups = subject_sig
        reference_bonds, reference_groups = reference_sig
        details = []
        if subject_groups != reference_groups:
            details.append(f"group counts differ: input {subject_groups}, reference {reference_groups}")
        missing = sorted(reference_bonds - subject_bonds)
        extra = sorted(subject_bonds - reference_bonds)
        if missing:
            details.append(f"the input is missing standardized bonds {missing}")
        if extra:
            details.append(f"the input has extra standardized bonds {extra}")
        raise PartitionError("The standardized topology does not match the reference ADPR; " + "; ".join(details))


def run(input_path: Path, output_path: Optional[Path], csv_path: Optional[Path], chain: str) -> Tuple[Path, Path]:
    if not input_path.is_file():
        raise PartitionError(f"Input file not found: {input_path}")
    atoms, graph = load_graph(input_path)
    groups, bridge_o = partition(atoms, graph)
    output_path = output_path or input_path.with_name(input_path.stem + "_partitioned.pdb")
    csv_path = csv_path or output_path.with_suffix(".atom_map.csv")
    write_outputs(input_path, output_path, csv_path, atoms, graph, groups, bridge_o, chain)
    return output_path, csv_path


def launch_gui() -> int:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("PAR Monomer Structure Partitioning Tool")
    root.geometry("720x245")
    input_var, output_var = tk.StringVar(), tk.StringVar()

    def choose_input() -> None:
        selected = filedialog.askopenfilename(
            title="Select an ADPR PDB",
            filetypes=[("PDB files", "*.pdb"), ("All files", "*.*")],
        )
        if selected:
            input_var.set(selected)
            output_var.set(str(Path(selected).with_name(Path(selected).stem + "_partitioned.pdb")))

    def choose_output() -> None:
        selected = filedialog.asksaveasfilename(
            title="Save Partitioned PDB",
            defaultextension=".pdb",
            filetypes=[("PDB files", "*.pdb")],
        )
        if selected:
            output_var.set(selected)

    def execute() -> None:
        try:
            out, mapping = run(Path(input_var.get()), Path(output_var.get()) if output_var.get() else None, None, "A")
            messagebox.showinfo("Complete", f"PDB: {out}\nAtom map: {mapping}")
        except Exception as exc:
            messagebox.showerror("Partitioning Failed", str(exc))

    frame = ttk.Frame(root, padding=18)
    frame.pack(fill="both", expand=True)
    ttk.Label(
        frame,
        text="Outputs one PAR monomer and identifies five internal groups from bond connectivity",
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 18))
    ttk.Label(frame, text="Input PDB").grid(row=1, column=0, sticky="w")
    ttk.Entry(frame, textvariable=input_var, width=72).grid(row=1, column=1, padx=8)
    ttk.Button(frame, text="Browse...", command=choose_input).grid(row=1, column=2)
    ttk.Label(frame, text="Output PDB").grid(row=2, column=0, sticky="w", pady=12)
    ttk.Entry(frame, textvariable=output_var, width=72).grid(row=2, column=1, padx=8, pady=12)
    ttk.Button(frame, text="Browse...", command=choose_output).grid(row=2, column=2)
    ttk.Button(frame, text="Start Partitioning", command=execute).grid(row=3, column=1, pady=16)
    root.mainloop()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Keep diagnostics readable even under Windows shells whose inherited stream
    # encoding is not UTF-8 (Codex/redirected consoles included).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Write ADP-ribose as one PAR residue and annotate five internal structural groups."
    )
    parser.add_argument("input", nargs="?", type=Path, help="Input PDB; omit to open the GUI")
    parser.add_argument("-o", "--output", type=Path, help="Output PDB path")
    parser.add_argument("--csv", type=Path, help="Output atom-map CSV path")
    parser.add_argument("--chain", default="A", help="Output chain ID (default: A)")
    parser.add_argument(
        "--reference",
        type=Path,
        help="Optional reference ADPR PDB for validating the standardized topology",
    )
    args = parser.parse_args(argv)
    if args.input is None:
        return launch_gui()
    if len(args.chain) != 1:
        parser.error("--chain must be a single character")
    try:
        if args.reference:
            compare_topology(args.input, args.reference)
        out, mapping = run(args.input, args.output, args.csv, args.chain)
    except PartitionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Complete: {out}")
    print(f"Atom map: {mapping}")
    if args.reference:
        print(f"Topology check passed: standardized bonds exactly match {args.reference}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
