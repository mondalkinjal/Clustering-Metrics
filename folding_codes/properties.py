import MDAnalysis as mda
import numpy as np
from MDAnalysis import transformations
import MDAnalysis.transformations as trans
from MDAnalysis.transformations import unwrap
from MDAnalysis.analysis import rms, diffusionmap, align
from MDAnalysis.analysis.distances import dist
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
import matplotlib.pyplot as plt
from MDAnalysis.lib.distances import calc_bonds
from MDAnalysis.analysis.align import AlignTraj
import MDAnalysis.analysis.rms
topology_file="protein_folding.pdb"
ref = mda.Universe("crystal_structure.pdb")
dcd_files = [f'pnas2012-1yrf-WT-345K-protein-{i:03d}.dcd' for i in range(0, 199)]
u = mda.Universe(topology_file, dcd_files)
protein = u.select_atoms('name CA')

backbone = u.select_atoms('backbone')
R = MDAnalysis.analysis.rms.RMSD(u, ref, select="backbone")
R.run()
rmsd_values = R.rmsd.T
num_frames_in_trajectory=len(u.trajectory)
multiplier = 0.000200135
times = np.array([(i + 1) * multiplier for i in range(num_frames_in_trajectory)])
plt.figure(figsize=(8, 6))
plt.plot(times, rmsd_values[2], label="Backbone RMSD", color='blue')
plt.xlabel("Frame")
plt.ylabel("RMSD")
plt.title("RMSD of Protein Backbone Over Time")
plt.legend()
plt.savefig("RMSD_plot.png")
plt.show()
final_rmsd=rmsd_values[2]
final_rmsd=final_rmsd[99::100]
np.save("final_rmsd",final_rmsd)
selection1 = u.select_atoms('resid 42:46')
selection2 = u.select_atoms('resid 72:76')
end_to_end_distances = []

# Loop through each frame in the trajectory
for ts in u.trajectory:
    box = ts.dimensions
    # Calculate the distance between the N- and C-terminal C-alpha atoms
    com1 = selection1.center_of_mass(pbc=True)
    com2 = selection2.center_of_mass(pbc=True)
    distance = calc_bonds(com1, com2, box=u.dimensions)
    end_to_end_distances.append(distance)

# Create a time array for plotting (assuming the trajectory includes time information)

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(times, end_to_end_distances)
plt.xlabel('Time (ps)')
plt.ylabel('End-to-End Distance (Å)')
plt.title('End-to-End C-alpha Distance Over Time')
plt.show()
end_to_end_distances=np.array(end_to_end_distances)
end_to_end_distances=end_to_end_distances[99::100]
np.save("end_to_end_distances",end_to_end_distances)


