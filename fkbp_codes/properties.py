import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
from MDAnalysis.analysis import rms, diffusionmap, align
from MDAnalysis.analysis.distances import dist
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
import matplotlib.colors
import matplotlib.ticker as ticker

subplot_labels = ['A', 'B', 'C', 'D']
def rmsd_for_atomgroups(selection1, selection2=None):
    universe=mda.Universe("protein_ligand.gro","final_production.xtc")
    universe.trajectory[0]
    ref = universe
    rmsd_analysis = rms.RMSD(universe, ref, select=selection1, groupselections=selection2)
    rmsd_analysis.run()
    final_rmsd = rmsd_analysis.results.rmsd.T   # transpose makes it easier for plotting
    return final_rmsd
final_rmsd = rmsd_for_atomgroups("backbone", ["protein", "not protein"])
np.save("backbone_rmsd",final_rmsd[2])
np.save("ligand_rmsd",final_rmsd[4])
u=mda.Universe("protein_ligand.gro","final_production.xtc")
fig, axes = plt.subplots(3, 1, figsize=(12, 20))
num_frames_in_trajectory=len(u.trajectory)
multiplier = 0.001
times = np.array([(i + 1) * multiplier for i in range(num_frames_in_trajectory)])


axes = axes.flatten()
backbone_rmsd=final_rmsd[2]
mana_rmsd=final_rmsd[4]
axes[0].plot(times, backbone_rmsd)
axes[0].set_title(subplot_labels[0], fontweight='bold')
axes[0].set_ylabel(r"RMSD ($\AA$)",fontweight='bold')
axes[1].plot(times, mana_rmsd)
axes[1].set_title(subplot_labels[1], fontweight='bold')
axes[1].set_ylabel(r"RMSD ($\AA$)",fontweight='bold')
ligand = u.select_atoms("not protein")
protein = u.select_atoms("protein")

def calculate_average_distance(topology_file, trajectory_file, protein_selection, ligand_selection):
    u = mda.Universe(topology_file, trajectory_file)
    protein_residues = u.select_atoms(protein_selection)
    ligand_atoms = u.select_atoms(ligand_selection)
    num_residues=len(protein_residues)
    avg_distance=np.zeros(num_residues)
    v=0
    for ts in u.trajectory:
        dist = mda.analysis.distances.distance_array(protein_residues.positions, ligand_atoms.positions)
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

def find_elements_less_than(arr, threshold):
    indices = np.where(arr < threshold)[0]
    elements = arr[indices]
    return indices, elements

# Example usage
threshold_value = 12
print (result)
indices, elements = find_elements_less_than(result, threshold_value)
print("Indices:", indices)
print("Elements:", elements)

print (len(indices))
final_array=[]
for i in range (0,len(u.trajectory)):
        u.trajectory[i]
        selection_1=u.select_atoms(f"name CA and resid {' '.join(map(str, indices))}")
        selection_2=u.select_atoms("not protein")
        dist_arr = mda.analysis.distances.distance_array(selection_1.positions,selection_2.positions)
        dist_arr_mean=np.mean(dist_arr, axis=1)
        final_array.append(dist_arr_mean.copy())        
distances_array = np.stack(final_array, axis=0)
np.save("distances_array",distances_array)
residues = u.select_atoms(protein_selection)[indices]

# Prepare final array for distances
final_array = []
residue_ids = [residue.resid for residue in residues]
print (len(residue_ids))
x = []
y = []
z = []

num_frames, num_residues = distances_array.shape

for i in range(1, num_frames + 1):
    for j in range(1, num_residues + 1):
        y.append(j)
        x.append(i * multiplier)  # Assuming the time step is 0.1 ns
        z.append(distances_array[i - 1, j - 1])

sc = axes[2].scatter(x, y, c=z, cmap='viridis')

# Set y-ticks using residue IDs from the selection
axes[2].set_yticks(range(1, num_residues+1))
axes[2].set_yticklabels(residue_ids, fontsize=12, fontweight='bold')

# Set labels
axes[2].set_ylabel('Residue ID', fontweight='bold')
axes[2].set_xlabel('Time (ns)', fontweight='bold')

# Add colorbar
pos = axes[2].get_position()
cbar = plt.colorbar(sc, ax=axes[2], orientation='vertical', fraction=0.05, pad=0.04)
cbar.ax.set_position([pos.x0 + pos.width + 0.01, axes[2].get_position().y0, 0.02, pos.height])
cbar.ax.tick_params(labelsize=20)
cbar.set_label(r"Distance ($\AA$)", fontsize=13, fontweight='bold')

# Set title
axes[2].set_title(subplot_labels[2], fontweight='bold')

# Adjust layout and show the plot
fig.savefig("total_analysis.jpg")


