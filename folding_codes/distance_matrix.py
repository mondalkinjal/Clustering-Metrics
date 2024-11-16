from MDAnalysis.analysis import distances
import MDAnalysis as mda
import numpy as np
u=mda.Universe("protein_folding.pdb")
dcd_files = [f'pnas2012-1yrf-WT-345K-protein-{i:03d}.dcd' for i in range(0, 199)]
u.load_new(dcd_files)
ca_atoms = u.select_atoms('name CA')
multiplier = 0.0002000000
num_frames_in_trajectory=len(u.trajectory)
times = np.array([(i + 1) * multiplier for i in range(num_frames_in_trajectory)])
dist_array=[]
for ts in u.trajectory:
    box = ts.dimensions
    dist = distances.distance_array(ca_atoms.positions, ca_atoms.positions, box=box)
    dist_array.append(dist.copy())
combined_array = np.stack(dist_array, axis=0)
combined_array = combined_array[99::100,...]
np.save ("protein_folding.npy",combined_array)
