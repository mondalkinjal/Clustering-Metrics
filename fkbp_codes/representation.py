import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.distances import dist
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random

def calculate_average_distance(topology_file, trajectory_file, protein_selection, ligand_selection):
    u = mda.Universe(topology_file, trajectory_file)
    protein_residues = u.select_atoms(protein_selection)
    ligand_atoms = u.select_atoms(ligand_selection)
    num_residues=len(protein_residues)
    avg_distance=np.zeros(num_residues)
    v=0
    for ts in u.trajectory:
        dist = distances.distance_array(protein_residues.positions, ligand_atoms.positions)
        avg_dist_per_residue = dist.mean(axis=1)
        avg_distance=np.add(avg_distance,avg_dist_per_residue)
    average_distance = avg_distance
    return average_distance

topology_file = 'protein_ligand.gro'
trajectory_file = 'final_production.xtc'

protein_selection = 'name CA and protein'
ligand_selection = 'not protein'  

average_ligand = calculate_average_distance(topology_file, trajectory_file, protein_selection, ligand_selection)
matrices = [average_ligand]

result = np.zeros_like(matrices[0])

# Add the matrices
for matrix in matrices:
    result = np.add(result, matrix)
u = mda.Universe(topology_file, trajectory_file)
result=result/(len(u.trajectory))
def standardize_columns(matrix):
    mean = np.max(matrix, axis=0)
    std = np.min(matrix, axis=0)
    standardized_matrix = (matrix - std) / (mean-std)
    return standardized_matrix


def find_elements_less_than(arr, threshold):
    indices = np.where(arr < threshold)[0]
    elements = arr[indices]
    return indices, elements

# Example usage
threshold_value = 30
indices, elements = find_elements_less_than(result, threshold_value)

sel = f"name CA and resid {' '.join(map(str, indices))}"

sel2="not protein"
structure="protein_ligand.gro"
trajectory="final_production.xtc"
data=[]

data = []
def combined_func(selection, separate_selections, structure, trajectory, data):
    u = mda.Universe(structure, trajectory)

    # Select the atoms for the main selection
    selection_1 = u.select_atoms(selection)
    
    
    # Extract the separate selections from the dictionary
    selection_2 = u.select_atoms(separate_selections)

    for ts in u.trajectory:
        # Calculate distances between selection and the separate selections
        dist_arr = distances.distance_array(selection_1.positions, selection_2.positions)

        # Stack the distances without taking mean
        dist_matrix = np.mean(dist_arr, axis=1)

        # Append the result to the data list
        data.append(dist_matrix.copy())
    
    combined_data = np.stack(data, axis=0)

    return combined_data

# Run the function
combined_data = combined_func(sel, sel2, structure, trajectory, data)

# Stack the data into a final array

# Save the combined data
np.save("representation_protein.npy", combined_data)
