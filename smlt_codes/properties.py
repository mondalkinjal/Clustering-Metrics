import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
from MDAnalysis.analysis import rms, diffusionmap, align
from MDAnalysis.analysis.distances import dist
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
def rmsd_for_atomgroups(selection1, selection2=None):
    universe=mda.Universe("protein_ligand.gro","final_production.xtc")
    universe.trajectory[0]
    ref = universe
    rmsd_analysis = rms.RMSD(universe, ref, select=selection1, groupselections=selection2)
    rmsd_analysis.run()
    final_rmsd = rmsd_analysis.results.rmsd.T   # transpose makes it easier for plotting
    return final_rmsd
final_rmsd = rmsd_for_atomgroups("backbone", ["protein", "resname BMANA"])
np.save("backbone_rmsd",final_rmsd[2])
np.save("mana_rmsd",final_rmsd[4])
u=mda.Universe("protein_ligand.gro","final_production.xtc")
final_array=[]
for i in range (0,len(u.trajectory)):
        u.trajectory[i]
        selection_1=u.select_atoms("name CA and (resid 21 or resid 94 or resid 141 or resid 142 or resid 146 or resid 147 or resid 150 or resid 194 or resid 197 or resid 201 or resid 291)")
        selection_2=u.select_atoms("resname BMANA")
        dist_arr = mda.analysis.distances.distance_array(selection_1.positions,selection_2.positions,box=[78.80383,78.80383,78.80383,90.,90.,90.])
        dist_arr_mean=np.mean(dist_arr, axis=1)
        final_array.append(dist_arr_mean.copy())        
final_array = np.stack(final_array, axis=0)
np.save("distances_array",final_array)
u=mda.Universe("protein_ligand.gro","final_production.xtc")
os.chdir("/scratch/zt1/project/energybio/user/kinjal14/clustering_paper/protein_ligand/hbonds")
cutoff=1000
residues=[]
residue_codes=[]
for i in range (1,308):
    file_name = f"{i}.dat"
    data=np.loadtxt(file_name)
    second_column=data[:,1]
    second_column[second_column>0]=1
    no_bonds=np.sum(second_column)
    if (no_bonds>cutoff):
        residues.append(i)
for residue in residues:
        selection_2="resid "+str(residue)
        u2=u.select_atoms(selection_2)
        resname=u2.residues.resnames[0]
        final_string=str(resname)+str(residue)
        residue_codes.append(final_string)

original_options = np.get_printoptions()
# Set print options to show all elements
np.set_printoptions(threshold=np.inf)

h_bondlist=np.zeros((len(residues), len(u.trajectory)+1))

for i in range (0,len(residues)):
    file_name = f"{residues[i]}.dat"
    data=np.loadtxt(file_name)
    second_column=data[:,1]
    second_column[second_column>0]=1
    h_bondlist[i,:]=second_column
os.chdir("/scratch/zt1/project/energybio/user/kinjal14/clustering_paper/protein_ligand")
np.save("h_bond.npy",h_bondlist.T)
np.save("h_list.npy",residue_codes)
u=mda.Universe("protein_ligand.gro","final_production.xtc")
cutoff=3.5
u1=u.select_atoms("(resname ARG and (name NE or name NH1 or name NH2)) or (resname LYS and name NZ) or (resname HSP and (name NE2 or name ND1))")
k=u1.resids
distinct_count=len(set(k))
element_count=Counter(k)
sorted_k=sorted(element_count.keys())

final_matrix=np.zeros((distinct_count,len(u.trajectory)))
for i in range (0,len(u.trajectory)):
    u.trajectory[i]
    u1=u.select_atoms("(resname ARG and (name NE or name NH1 or name NH2)) or (resname LYS and name NZ) or (resname HSP and (name NE2 or name ND1))")
    u2=u.select_atoms("resname BMANA and (name O61 or name O62)")
    k=u1.resids
    element_count=Counter(k)
    sorted_keys=sorted(element_count.keys())
    sorted_dict={key:element_count[key] for key in sorted_keys}
    key_values=[sorted_dict[key] for key in sorted_dict]
    dist_arr=mda.analysis.distances.distance_array(u1.positions,u2.positions,box=[78.80383,78.80383,78.80383,90.,90.,90.])
    result_arr= np.where(dist_arr<cutoff,1,0)
    sum_result_arr=np.sum(result_arr, axis=1)
    p=0
    for j in range (0,len(sorted_dict)):
        final_matrix[j,i]=np.sum(sum_result_arr[p:p+key_values[j]])
        p=p+key_values[j]
final_matrix[final_matrix>0]=1
import_indices=[]
imp_residues=[]
residue_codes=[]
cutoff=1000
for i in range (0,distinct_count):
    if (np.sum(final_matrix[i,:])>cutoff):
        imp_residues.append(sorted_k[i])
        import_indices.append(i)

important_matrix=np.zeros((len(imp_residues),len(u.trajectory)))
for i in range (0,len(imp_residues)):
    important_matrix[i,:]=final_matrix[import_indices[i],:]
for residue in imp_residues:
        selection_2="resid "+str(residue)
        u2=u.select_atoms(selection_2)
        resname=u2.residues.resnames[0]
        final_string=str(resname)+str(residue)
        residue_codes.append(final_string)

np.save("salt_bridge.npy",important_matrix.T)
np.save("salt_list.npy",imp_residues)

