"""B100 molecule tokenization, matching the frozen project encoder."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np

WIDTH = 128
LAYERS = 2
ENV_BUCKETS = 4096
RELATION_COUNT = 6
ATOM_FEATURE_COUNT = 7


@dataclass(frozen=True)
class MoleculeGraph:
    atom_categories: np.ndarray
    environment_ids: np.ndarray
    relations: np.ndarray


def _environment_bucket(identifier: int, radius: int, buckets: int = ENV_BUCKETS) -> int:
    digest = hashlib.sha256(f"morgan-v1|{radius}|{int(identifier)}".encode("ascii")).digest()
    return 1 + int.from_bytes(digest[:8], "big") % (buckets - 1)


def tokenize_smiles(smiles: str) -> MoleculeGraph:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be nonempty text")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() < 1:
        raise ValueError("SMILES is not RDKit-parseable")
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    n_atoms = molecule.GetNumAtoms()
    categories = np.zeros((n_atoms, ATOM_FEATURE_COUNT), dtype=np.int64)
    stereo_e = np.zeros(n_atoms, dtype=np.int64)
    stereo_z = np.zeros(n_atoms, dtype=np.int64)
    for bond in molecule.GetBonds():
        stereo = bond.GetStereo()
        if stereo in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS):
            stereo_e[bond.GetBeginAtomIdx()] += 1
            stereo_e[bond.GetEndAtomIdx()] += 1
        elif stereo in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS):
            stereo_z[bond.GetBeginAtomIdx()] += 1
            stereo_z[bond.GetEndAtomIdx()] += 1
    for index, atom in enumerate(molecule.GetAtoms()):
        atomic_number = int(atom.GetAtomicNum())
        categories[index, 0] = atomic_number if 1 <= atomic_number <= 118 else 119
        formal_charge = int(atom.GetFormalCharge())
        categories[index, 1] = formal_charge + 4 if -4 <= formal_charge <= 4 else 9
        categories[index, 2] = int(atom.GetIsAromatic())
        categories[index, 3] = min(int(atom.GetTotalDegree()), 10)
        cip = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""
        categories[index, 4] = {"R": 1, "S": 2}.get(cip, 0)
        categories[index, 5] = min(int(stereo_e[index]), 4)
        categories[index, 6] = min(int(stereo_z[index]), 4)
    bit_info: dict[int, list[tuple[int, int]]] = {}
    rdMolDescriptors.GetMorganFingerprint(
        molecule, 2, useChirality=True, bitInfo=bit_info,
        includeRedundantEnvironments=True,
    )
    environments = np.zeros((n_atoms, 3), dtype=np.int64)
    for identifier, occurrences in bit_info.items():
        for atom_index, radius in occurrences:
            if 0 <= radius <= 2:
                environments[atom_index, radius] = _environment_bucket(identifier, radius)
    if np.any(environments == 0):
        for radius in range(1, 3):
            environments[:, radius] = np.where(
                environments[:, radius] == 0,
                environments[:, radius - 1], environments[:, radius],
            )
        if np.any(environments[:, 0] == 0):
            raise ValueError("RDKit omitted a radius-0 atom environment")
    distances = np.asarray(Chem.GetDistanceMatrix(molecule, useBO=False), dtype=np.float64)
    relations = np.full((n_atoms, n_atoms), 5, dtype=np.uint8)
    fragment_by_atom = np.empty(n_atoms, dtype=np.int64)
    for fragment_index, atom_indices in enumerate(Chem.GetMolFrags(molecule, asMols=False)):
        fragment_by_atom[list(atom_indices)] = fragment_index
    connected = (fragment_by_atom[:, None] == fragment_by_atom[None, :]) & np.isfinite(distances)
    relations[connected & (distances == 0)] = 0
    relations[connected & (distances == 1)] = 1
    relations[connected & (distances == 2)] = 2
    relations[connected & (distances == 3)] = 3
    relations[connected & (distances >= 4)] = 4
    return MoleculeGraph(categories, environments, relations)


def collate_graphs(graphs: Sequence[MoleculeGraph]) -> tuple[np.ndarray, ...]:
    if not graphs:
        raise ValueError("cannot collate an empty graph batch")
    max_atoms = max(int(graph.atom_categories.shape[0]) for graph in graphs)
    batch = len(graphs)
    atoms = np.zeros((batch, max_atoms, ATOM_FEATURE_COUNT), dtype=np.int64)
    environments = np.zeros((batch, max_atoms, 3), dtype=np.int64)
    relations = np.zeros((batch, max_atoms, max_atoms), dtype=np.int64)
    mask = np.zeros((batch, max_atoms), dtype=np.bool_)
    for row, graph in enumerate(graphs):
        n_atoms = int(graph.atom_categories.shape[0])
        atoms[row, :n_atoms] = graph.atom_categories
        environments[row, :n_atoms] = graph.environment_ids
        relations[row, :n_atoms, :n_atoms] = graph.relations
        mask[row, :n_atoms] = True
    return atoms, environments, relations, mask
