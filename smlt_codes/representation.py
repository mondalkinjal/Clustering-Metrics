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
def calculate_average_distance(topology_file, trajectory_file, protein_selection, ligand_selection,num):
    u = mda.Universe(topology_file, trajectory_file)
    protein_residues = u.select_atoms(protein_selection)
    ligand_atoms = u.select_atoms(ligand_selection)
    num_residues=len(protein_residues)
    avg_distance=np.zeros(num_residues)
    v=0
    for ts in u.trajectory:
        dist = distances.distance_array(protein_residues.positions, ligand_atoms.positions, box=[78.80383,78.80383,78.80383,90.,90.,90.])
        avg_dist_per_residue = dist.mean(axis=1)
        avg_distance=np.add(avg_distance,avg_dist_per_residue)
    average_distance = avg_distance
    return average_distance

topology_file = 'protein_ligand.gro'
trajectory_file = 'final_production.xtc'
protein_selection = 'name CA and protein'
ligand_selection = 'resname BMANA'  
nhexa1=7500
average_hexa1 = calculate_average_distance(topology_file, trajectory_file, protein_selection, ligand_selection,nhexa1)
matrices = [average_hexa1]
result = np.zeros_like(matrices[0])


for matrix in matrices:
    result = np.add(result, matrix)

result=result/(nhexa1)
def standardize_columns(matrix):
    mean = np.max(matrix, axis=0)
    std = np.min(matrix, axis=0)
    standardized_matrix = (matrix - std) / (mean-std)
    return standardized_matrix


def find_elements_less_than(arr, threshold):
    indices = np.where(arr < threshold)[0]
    elements = arr[indices]
    return indices, elements


threshold_value = 25
print (result)
indices, elements = find_elements_less_than(result, threshold_value)
print("Indices:", indices)
print("Elements:", elements)

sel = f"name CA and resid {' '.join(map(str, indices))}"
MMMM_1="resname BMANA and resid 309"
MMMM_2="resname BMANA and resid 310"
MMMM_3="resname BMANA and resid 311"
MMMM_4="resname BMANA and resid 312"
MMMM_5="resname BMANA and resid 313"
MMMM_6="resname BMANA and resid 314"
structure="protein_ligand.gro"
trajectory="final_production.xtc"
data=[]

def combined_func(selection,selection_1, selection_2, selection_3, selection_4, selection_5, selection_6, structure, trajectory,data):
    u = mda.Universe(structure, trajectory)
    selection=u.select_atoms(selection)
    selection_1=u.select_atoms(selection_1)
    selection_2=u.select_atoms(selection_2)
    selection_3=u.select_atoms(selection_3)
    selection_4=u.select_atoms(selection_4)
    selection_5=u.select_atoms(selection_5)
    selection_6=u.select_atoms(selection_6)
    for ts in u.trajectory:
        
        dist_1 = distances.distance_array(selection.positions, selection_1.positions, box=[78.80383,78.80383,78.80383,90.,90.,90.])
        dist_2 = distances.distance_array(selection.positions, selection_2.positions, box=[78.80383,78.80383,78.80383,90.,90.,90.])
        dist_3 = distances.distance_array(selection.positions, selection_3.positions, box=[78.80383,78.80383,78.80383,90.,90.,90.])
        dist_4 = distances.distance_array(selection.positions, selection_4.positions, box=[78.80383,78.80383,78.80383,90.,90.,90.])
        dist_5 = distances.distance_array(selection.positions, selection_5.positions, box=[78.80383,78.80383,78.80383,90.,90.,90.])
        dist_6 = distances.distance_array(selection.positions, selection_6.positions, box=[78.80383,78.80383,78.80383,90.,90.,90.])
        
        avg_dist_per_residue_1 = dist_1.mean(axis=1)
        avg_dist_per_residue_2 = dist_2.mean(axis=1)
        avg_dist_per_residue_3 = dist_3.mean(axis=1)
        avg_dist_per_residue_4 = dist_4.mean(axis=1)
        avg_dist_per_residue_5 = dist_5.mean(axis=1)
        avg_dist_per_residue_6 = dist_6.mean(axis=1)
        final_array=np.vstack((avg_dist_per_residue_1, avg_dist_per_residue_2, avg_dist_per_residue_3, avg_dist_per_residue_4, avg_dist_per_residue_5, avg_dist_per_residue_6 ))
        data.append(final_array)
    return data
data=combined_func(sel,MMMM_1,MMMM_2,MMMM_3,MMMM_4,MMMM_5,MMMM_6,structure,trajectory,data)
combined_data=np.stack(data,axis=0)
print (combined_data.shape)
np.save("combined_data.npy",combined_data)

