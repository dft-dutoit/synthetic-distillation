import numpy as np
from collections import defaultdict, Counter
import ase  # Assuming ASE is used and 'atoms' is an ASE Atoms object
from locache import persist
import ase.io

def build_water_molecules(atoms):
    """
    Build a list of water molecules from an ASE Atoms object.
    Assumes each water molecule consists of 3 atoms in the order [O, H, H].
    """
    n_atoms = len(atoms)
    n_waters = n_atoms // 3
    water_mols = []
    for i in range(n_waters):
        base = 3 * i
        water_mols.append({
            'O': base,           # Oxygen index
            'H': [base + 1, base + 2]  # Hydrogen indices
        })
        # ensure that this is a sensible water molecule
        assert atoms.get_distance(base, base + 1) < 1.5
        assert atoms.get_distance(base, base + 2) < 1.5

    return water_mols

def find_hydrogen_bonds(atoms, water_mols, distance_cutoff=3.5, angle_cutoff=30):
    """
    Identify hydrogen bonds based on a distance and angle criterion.
    
    Parameters:
        atoms : ASE Atoms object
        water_mols : list of water molecules (from build_water_molecules)
        distance_cutoff : float, maximum O-O distance (default 3.5 Å)
        angle_cutoff : float, maximum angle (default 30°)
    
    Returns:
        hbonds : list of tuples (acceptor oxygen index, donor oxygen index, hydrogen index)
    """
    hbonds = []
    for i, donor in enumerate(water_mols):
        donor_O = donor['O']
        for j, acceptor in enumerate(water_mols):
            if i == j:
                continue  # Skip same molecule
            acceptor_O = acceptor['O']
            OO_dist = atoms.get_distance(donor_O, acceptor_O, mic=True)
            if OO_dist < distance_cutoff:
                for h in donor['H']:
                    theta = atoms.get_angle(acceptor_O, donor_O, h, mic=True)
                    if theta < angle_cutoff:
                        hbonds.append((acceptor_O, donor_O, h))
    return hbonds

def calculate_average_hbonds(hbonds):
    """
    Calculate the average number of hydrogen bonds per oxygen.
    
    Parameters:
        hbonds : list of hydrogen bonds (each a tuple of indices)
    
    Returns:
        avg_hbonds : float, average hydrogen bonds per oxygen
    """
    if not hbonds:
        return 0
    hbond_array = np.array(hbonds)
    # Extract unique oxygen indices that participate as either acceptor or donor
    unique_oxygens = np.unique(np.concatenate([hbond_array[:, 0], hbond_array[:, 1]]))
    num_bonds = len(hbonds)
    avg_hbonds = 2 * num_bonds / len(unique_oxygens)
    return avg_hbonds

def analyze_bond_counts(hbonds):
    """
    Analyze hydrogen bond counts per oxygen, categorizing donor and acceptor roles.
    
    Returns:
        oxygen_bond_counts : dict mapping oxygen index to its 'accepting' and 'donating' counts
        type_counts        : Counter of bond types formatted as "A{accepting}D{donating}"
        percentages        : dict mapping bond type to its percentage among all oxygens
    """
    acceptor_counts = defaultdict(int)
    donor_counts = defaultdict(int)
    
    # Tally counts for acceptor and donor roles
    for acceptor, donor, _ in hbonds:
        acceptor_counts[acceptor] += 1
        donor_counts[donor] += 1

    all_oxygens = set(acceptor_counts.keys()).union(set(donor_counts.keys()))
    oxygen_bond_counts = {}
    for oxygen in all_oxygens:
        oxygen_bond_counts[oxygen] = {
            'accepting': acceptor_counts.get(oxygen, 0),
            'donating': donor_counts.get(oxygen, 0)
        }
    
    # Count bond types using the format "A{# of accepting}D{# of donating}"
    type_counts = Counter()
    for oxygen, counts in oxygen_bond_counts.items():
        bond_type = f"A{counts['accepting']}D{counts['donating']}"
        type_counts[bond_type] += 1

    total_oxygens = len(oxygen_bond_counts)
    percentages = {bond_type: (count / total_oxygens) * 100 
                   for bond_type, count in type_counts.items()}
    
    return oxygen_bond_counts, type_counts, percentages

@persist
def main(file_path):
    """
    Main function to process an ASE Atoms object, compute hydrogen bonds,
    and analyze the bonding environment.
    
    Parameters:
        atoms : ASE Atoms object
    
    Returns:
        A dictionary with average hydrogen bonds per oxygen and detailed bond counts.
    """
    atoms = ase.io.read(file_path)
    water_mols = build_water_molecules(atoms)
    hbonds = find_hydrogen_bonds(atoms, water_mols)
    avg_hbonds = calculate_average_hbonds(hbonds)
    oxygen_bond_counts, type_counts, percentages = analyze_bond_counts(hbonds)
    
    # print("Average hydrogen bonds per oxygen:", avg_hbonds)
    # print("Oxygen bond counts:", oxygen_bond_counts)
    # print("Bond type counts:", type_counts)
    # print("Bond type percentages:", percentages)
    
    return {
        "average_hbonds": avg_hbonds,
        "oxygen_bond_counts": oxygen_bond_counts,
        "type_counts": type_counts,
        "percentages": percentages
    }
