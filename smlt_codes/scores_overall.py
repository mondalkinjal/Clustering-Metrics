import numpy as np
import matplotlib.pyplot as plt
from Clustering import Clustering
from scoring import scoring 
from sklearn.decomposition import PCA
import MDAnalysis as mda
import matplotlib.colors
import matplotlib.ticker as ticker

def dimensionality_reduction(data):
    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(data)
    pca_result = pca_result.flatten()
    return pca_result

u=mda.Universe("protein_ligand.gro","final_production.xtc")
num_frames_in_trajectory=len(u.trajectory)
multiplier = 0.1
times = np.array([(i + 1) * multiplier for i in range(num_frames_in_trajectory)])
Clustering = Clustering()

backbone_rmsd=np.load("backbone_rmsd.npy")
backbone_rmsd=backbone_rmsd.reshape(-1,1)
backbone_rmsd_plot=backbone_rmsd.flatten()
mana_rmsd=np.load("mana_rmsd.npy")
mana_rmsd=mana_rmsd.reshape(-1,1)
mana_rmsd_plot=mana_rmsd.flatten()
distances_array=np.load("distances_array.npy")
distances_array_reduced=dimensionality_reduction(distances_array)
h_bond=np.load("h_bond.npy")
h_bond=h_bond[1:,:]
h_bond_reduced=dimensionality_reduction(h_bond)
h_list=np.load("h_list.npy")
salt_bridge=np.load("salt_bridge.npy")
salt_bridge_reduced=dimensionality_reduction(salt_bridge)
salt_list=np.load("salt_list.npy")
subplot_labels = ['A', 'B', 'C', 'D', 'E']

bounds=[0,11,22,33,44,55,66]
colors=["red","yellow","green","pink","orange","blue"]
cmap = matplotlib.colors.ListedColormap(colors)
norm=matplotlib.colors.BoundaryNorm(bounds,len(colors))

fig, axes = plt.subplots(5, 1, figsize=(12, 20))

axes = axes.flatten()
axes[0].plot(times, backbone_rmsd.reshape(-1))
axes[0].set_title(subplot_labels[0], fontweight='bold')
axes[0].set_ylabel(r"RMSD ($\AA$)",fontweight='bold')
axes[1].plot(times, mana_rmsd.reshape(-1))
axes[1].set_title(subplot_labels[1], fontweight='bold')
axes[1].set_ylabel(r"RMSD ($\AA$)",fontweight='bold')
x=[]
y=[]
z=[]
for i in range (1,7502):
    for j in range (1,12):
        y.append(j)
        x.append((i)/10.0)
        z.append(distances_array[i-1,j-1])
pos = axes[1].get_position()
sc = axes[2].scatter(x, y, c=z, cmap=cmap, norm=norm)
axes[2].set_yticks([1,2,3,4,5,6,7,8,9,10,11])
axes[2].set_yticklabels(['42','115','162','163','167','168','171','215','218','222','312'])
axes[2].set_ylabel('residue',fontweight='bold')
cbar=plt.colorbar(sc,ax=axes[2], orientation='vertical', fraction=0.05, pad=0.04)
cbar.ax.set_position([pos.x0 + pos.width + 0.01, axes[2].get_position().y0, 0.02, pos.height])
cbar.ax.tick_params(labelsize=20)
cbar.set_label(r"distance ($\AA$)",fontsize=13,fontweight='bold')
axes[2].set_title(subplot_labels[2],fontweight='bold')
axes[2].set_position([pos.x0, axes[2].get_position().y0, pos.width, pos.height])
x=[]
y=[]
z=[]
for i in range (0,len(h_list)):
    for j in range (0,h_bond.shape[0]):
        y.append(i)
        x.append((j)/10.0)
        z.append(h_bond[j,i])
bounds=[0,0.5,1.5]
colors=["green","black"]
cmap = matplotlib.colors.ListedColormap(colors)
norm = matplotlib.colors.BoundaryNorm(bounds,len(colors))
sc = axes[3].scatter(x, y, c=z, cmap=cmap, norm=norm)
number_list = list(range(len(h_list)))
h_list= h_list.tolist()
h_list = [int(item[3:]) + 21 for item in h_list]
axes[3].set_yticks(number_list)
axes[3].set_yticklabels(h_list)
axes[3].set_ylabel('residue',fontweight='bold')
handles_hbond, labels_hbond = sc.legend_elements()
axes[3].legend(handles_hbond, labels_hbond, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title="H_Bond")
axes[3].set_title(subplot_labels[3],fontweight='bold')
x=[]
y=[]
z=[]
for i in range (0,len(salt_list)):
    for j in range (0,salt_bridge.shape[0]):
        y.append(i)
        x.append((j)/10.0)
        z.append(salt_bridge[j,i])
bounds=[0,0.5,1.5]
colors=["green","black"]
cmap = matplotlib.colors.ListedColormap(colors)
norm = matplotlib.colors.BoundaryNorm(bounds,len(colors))
sc = axes[4].scatter(x, y, c=z, cmap=cmap, norm=norm)
number_list = list(range(len(salt_list)))
salt_list= salt_list.tolist()
salt_list= [int(item) + 21 for item in salt_list]
axes[4].set_yticks(number_list)
axes[4].set_yticklabels(salt_list)
handles_saltbridge, labels_saltbridge = sc.legend_elements()
axes[4].legend(handles_saltbridge, labels_saltbridge, bbox_to_anchor=(1.02, 1.00), loc='upper left', title="Salt_Bridge")
axes[4].set_title(subplot_labels[4],fontweight='bold')
axes[4].set_xlabel('times(in ns)',fontweight='bold')
axes[4].set_ylabel('residue',fontweight='bold')

fig.savefig("total_analysis.jpg")

Clustering_Algorithms={'KMeans':Clustering.KMeans, 'AgglomerativeClustering':Clustering.AgglomerativeClustering, 'GaussianMixture':Clustering.GaussianMixture, 'BirchClustering':Clustering.BirchClustering}

n_clusters=[2,3,4,5,6]

backbone_rmsd_property_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
backbone_rmsd_label_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
mana_rmsd_property_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
mana_rmsd_label_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
distances_property_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
distances_label_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
hbond_property_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
hbond_label_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
saltbridge_property_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
saltbridge_label_scores = {alg: [] for alg in Clustering_Algorithms.keys()}

final_protein_rep = np.load("representation_protein.npy")
plt.rcParams.update({'font.size': 30})
fig_actual_backbone_rmsd, axes_actual_backbone_rmsd = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_actual_mana_rmsd, axes_actual_mana_rmsd = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_actual_distances, axes_actual_distances = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_actual_hbond, axes_actual_hbond = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_actual_saltbridge, axes_actual_saltbridge = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_backbone_rmsd, axes_backbone_rmsd = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_mana_rmsd, axes_mana_rmsd = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_distances, axes_distances = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_hbond, axes_hbond = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_saltbridge, axes_saltbridge = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')

subplot_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
row_labels = ['A', 'B', 'C', 'D']
col_labels = ['(i)', '(ii)', '(iii)', '(iv)', '(v)']
x_ax=0
xticks=[100,300,500,700]
xticklabels=['100','300','500','700']
for name,algorithm in Clustering_Algorithms.items():
    y_ax=0
    for n in n_clusters:
        cluster_labels=algorithm(final_protein_rep,n)
        print (name,algorithm)
        backbone_rmsd_property,backbone_rmsd_label,backbone_rmsd_clusters=scoring(backbone_rmsd,cluster_labels,name,n,0.20,algorithm)
        mana_rmsd_property,mana_rmsd_label,mana_rmsd_clusters=scoring(mana_rmsd,cluster_labels,name,n,0.20,algorithm)
        distance_property,distance_label,dist_clusters=scoring(distances_array,cluster_labels,name,n,0.20,algorithm)
        hbond_property,hbond_label,hbond_clusters=scoring(h_bond,cluster_labels,name,n,0.20,algorithm)
        saltbridge_property,saltbridge_label,saltbridge_clusters=scoring(salt_bridge,cluster_labels,name,n,0.20,algorithm) 
        
        backbone_rmsd_property_scores[name].append(backbone_rmsd_property)
        mana_rmsd_property_scores[name].append(mana_rmsd_property)
        distances_property_scores[name].append(distance_property)
        hbond_property_scores[name].append(hbond_property)
        saltbridge_property_scores[name].append(saltbridge_property)

        backbone_rmsd_label_scores[name].append(backbone_rmsd_label)
        mana_rmsd_label_scores[name].append(mana_rmsd_label)
        distances_label_scores[name].append(distance_label)
        hbond_label_scores[name].append(hbond_label)
        saltbridge_label_scores[name].append(saltbridge_label)

        y_ax=y_ax+1
    x_ax=x_ax+1

def weighted_average_dicts(d_1, d_2, d_3, d_4, w1, w2, w3, w4):
    # Ensure that w1 + w2 = 1
#    if not (w1 + w2 == 1):
#        raise ValueError("Weights w1 and w2 must sum to 1")

    # Initialize the result dictionary
    weighted_avg_dict = {}

    # Iterate over the keys and compute the weighted average
    for key in d_1:
        multiplied_d_1 = [x * w1 for x in d_1[key]]
        multiplied_d_2 = [x * w2 for x in d_2[key]]
        multiplied_d_3 = [x * w3 for x in d_3[key]]
        multiplied_d_4 = [x * w4 for x in d_4[key]]
        weighted_score = [x + y + z + u for x, y, z, u  in zip(multiplied_d_1,multiplied_d_2,multiplied_d_3,multiplied_d_4)]
        weighted_avg_dict[key] = weighted_score

    return weighted_avg_dict

w1 = 0.35
w2 = 0.35
w3 = 0.15
w4 = 0.15

result_average_property = weighted_average_dicts(mana_rmsd_property_scores, distances_label_scores, hbond_label_scores, saltbridge_label_scores,  w1, w2, w3, w4)
result_average_label = weighted_average_dicts(mana_rmsd_label_scores, distances_label_scores, hbond_label_scores, saltbridge_label_scores,  w1, w2, w3, w4)

def get_min_max(scores):
    all_values = np.array([score for alg_scores in scores.values() for score in alg_scores])
    return np.min(all_values), np.max(all_values)

plt.rcParams.update({'font.size': 10})
fig, axes = plt.subplots(1, 2, figsize=(14, 10), sharex=True)
axes = axes.flatten()

def plot_bars(ax, scores, title):
    bar_width = 0.2
    bar_positions = np.arange(len(n_clusters))
    for i, alg in enumerate(Clustering_Algorithms):
        ax.bar(bar_positions + i * bar_width, scores[alg], bar_width, label=alg)
    ax.set_title(title.replace('_', ' ').title(), fontweight='bold')
    ax.set_xlabel('Clusters', fontweight='bold')
    ax.set_ylabel('Scores', fontweight='bold')
    ax.set_xticks(bar_positions + bar_width * 1.5)
    ax.set_xticklabels(n_clusters)
    ax.legend()
    ymin, ymax = get_min_max(scores)
    ax.set_ylim(ymin-0.05, ymax+0.05)

plot_bars(axes[0], result_average_property, 'A')
plot_bars(axes[1], result_average_label, 'B')

plt.tight_layout()
plt.savefig("scores_overall.jpg")

