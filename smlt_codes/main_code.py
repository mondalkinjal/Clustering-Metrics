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

        label = f"{row_labels[x_ax]}{col_labels[y_ax]}"
        
        ax_actual_backbone_rmsd = axes_actual_backbone_rmsd[x_ax,y_ax]
        scatter_actual_backbone_rmsd=ax_actual_backbone_rmsd.scatter(times, backbone_rmsd_plot, c=cluster_labels, cmap='viridis', marker='o')
        ax_actual_backbone_rmsd.text(375, 4, row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_actual_backbone_rmsd.grid(True)
        handles_actual_backbone_rmsd, labels_actual_backbone_rmsd = scatter_actual_backbone_rmsd.legend_elements()
        ax_actual_backbone_rmsd.legend(handles_actual_backbone_rmsd, labels_actual_backbone_rmsd, title="Clusters",loc='lower left')

        ax_backbone_rmsd = axes_backbone_rmsd[x_ax,y_ax]
        scatter_backbone_rmsd=ax_backbone_rmsd.scatter(times, backbone_rmsd_plot, c=backbone_rmsd_clusters, cmap='viridis', marker='o')
        ax_backbone_rmsd.text(375, 4, row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_backbone_rmsd.grid(True)
        handles_backbone_rmsd, labels_backbone_rmsd = scatter_backbone_rmsd.legend_elements()
        ax_backbone_rmsd.legend(handles_backbone_rmsd, labels_backbone_rmsd, title="Clusters",loc='lower left')

        ax_actual_mana_rmsd = axes_actual_mana_rmsd[x_ax,y_ax]
        scatter_actual_mana_rmsd=ax_actual_mana_rmsd.scatter(times, mana_rmsd_plot, c=cluster_labels, cmap='viridis', marker='o')
        ax_actual_mana_rmsd.text(375, 32, row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_actual_mana_rmsd.grid(True)
        handles_actual_mana_rmsd, labels_actual_mana_rmsd = scatter_actual_mana_rmsd.legend_elements()
        ax_actual_mana_rmsd.legend(handles_actual_mana_rmsd, labels_actual_mana_rmsd, title="Clusters",loc='upper left')

        ax_mana_rmsd = axes_mana_rmsd[x_ax,y_ax]
        scatter_mana_rmsd=ax_mana_rmsd.scatter(times, mana_rmsd_plot, c=mana_rmsd_clusters, cmap='viridis', marker='o')
        ax_mana_rmsd.text(375, 32, row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_mana_rmsd.grid(True)
        handles_mana_rmsd, labels_mana_rmsd = scatter_mana_rmsd.legend_elements()
        ax_mana_rmsd.legend(handles_mana_rmsd, labels_mana_rmsd, title="Clusters",loc='upper left')

        ax_actual_distances = axes_actual_distances[x_ax,y_ax]
        scatter_actual_distances=ax_actual_distances.scatter(times, distances_array_reduced, c=cluster_labels, cmap='viridis', marker='o')
        ax_actual_distances.text(375,25,row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_actual_distances.grid(True)
        handles_actual_distances, labels_actual_distances = scatter_actual_distances.legend_elements()
        ax_actual_distances.legend(handles_actual_distances, labels_actual_distances, title="Clusters",loc='upper left')

        ax_distances = axes_distances[x_ax,y_ax]
        scatter_distances=ax_distances.scatter(times, distances_array_reduced, c=dist_clusters, cmap='viridis', marker='o')
        ax_distances.text(375,25,row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_distances.grid(True)
        handles_distances, labels_distances = scatter_distances.legend_elements()
        ax_distances.legend(handles_distances, labels_distances, title="Clusters",loc='upper left')

        ax_actual_hbond = axes_actual_hbond[x_ax,y_ax]
        scatter_actual_hbond=ax_actual_hbond.scatter(times, h_bond_reduced, c=cluster_labels, cmap='viridis', marker='o')
        ax_actual_hbond.text(375,0.8,row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_actual_hbond.grid(True)
        handles_actual_hbond, labels_actual_hbond = scatter_actual_hbond.legend_elements()
        ax_actual_hbond.legend(handles_actual_hbond, labels_actual_hbond, title="Clusters",loc='upper left')

        ax_hbond = axes_hbond[x_ax,y_ax]
        scatter_hbond=ax_hbond.scatter(times, h_bond_reduced, c=hbond_clusters, cmap='viridis', marker='o')
        ax_hbond.text(375,0.8,row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_hbond.grid(True)
        handles_hbond, labels_hbond = scatter_hbond.legend_elements()
        ax_hbond.legend(handles_hbond, labels_hbond, title="Clusters",loc='upper left')
        
        ax_actual_saltbridge = axes_actual_saltbridge[x_ax,y_ax]
        scatter_actual_saltbridge=ax_actual_saltbridge.scatter(times, salt_bridge_reduced, c=cluster_labels, cmap='viridis', marker='o')
        ax_actual_saltbridge.text(375,0.8,row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_actual_saltbridge.grid(True)
        handles_actual_saltbridge, labels_actual_saltbridge = scatter_actual_saltbridge.legend_elements()
        ax_actual_saltbridge.legend(handles_actual_saltbridge, labels_actual_saltbridge, title="Clusters",loc='upper left')

        ax_saltbridge = axes_saltbridge[x_ax,y_ax]
        scatter_saltbridge=ax_saltbridge.scatter(times, salt_bridge_reduced, c=saltbridge_clusters, cmap='viridis', marker='o')
        ax_saltbridge.text(375,0.8,row_labels[x_ax], ha='center', va='center',fontweight='bold')
        ax_saltbridge.grid(True)
        handles_saltbridge, labels_saltbridge = scatter_saltbridge.legend_elements()
        ax_saltbridge.legend(handles_saltbridge, labels_saltbridge, title="Clusters",loc='upper left')

        y_ax=y_ax+1
    x_ax=x_ax+1

xticks=[100,300,500,700]
xticklabels=['100','300','500','700']

fig_actual_backbone_rmsd.tight_layout(h_pad=0.0, w_pad=0.0)
fig_actual_backbone_rmsd.text(0.5, 0.0005, "Time(in ns)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_actual_backbone_rmsd.text(0.001, 0.52, "RMSD", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_actual_backbone_rmsd.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_actual_backbone_rmsd.savefig("backbone_rmsd_actual.jpg", bbox_inches='tight')

fig_backbone_rmsd.tight_layout(h_pad=0.0, w_pad=0.0)
fig_backbone_rmsd.text(0.5, 0.0005, "Time(in ns)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_backbone_rmsd.text(0.001, 0.52, "RMSD", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_backbone_rmsd.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_backbone_rmsd.savefig("backbone_rmsd.jpg", bbox_inches='tight')

fig_actual_mana_rmsd.tight_layout(h_pad=0.0, w_pad=0.0)
fig_actual_mana_rmsd.text(0.5, 0.0005, "Time(in ns)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_actual_mana_rmsd.text(0.001, 0.52, "RMSD", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_actual_mana_rmsd.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_actual_mana_rmsd.savefig("mana_rmsd_actual.jpg", bbox_inches='tight')

fig_mana_rmsd.tight_layout(h_pad=0.0, w_pad=0.0)
fig_mana_rmsd.text(0.5, 0.0005, "Time(in ns)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_mana_rmsd.text(0.001, 0.52, "RMSD", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_mana_rmsd.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_mana_rmsd.savefig("mana_rmsd.jpg", bbox_inches='tight')

fig_actual_distances.tight_layout(h_pad=0.0, w_pad=0.0)
fig_actual_distances.text(0.5, 0.0005, "Time(in ns)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_actual_distances.text(0.001, 0.52, "distances", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_actual_distances.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_actual_distances.savefig("distances_actual.jpg", bbox_inches='tight')

fig_distances.tight_layout(h_pad=0.0, w_pad=0.0)
fig_distances.text(0.5, 0.0005, "Time(in ns)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_distances.text(0.001, 0.52, "distances", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_distances.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_distances.savefig("distances.jpg", bbox_inches='tight')

fig_actual_hbond.tight_layout(h_pad=0.0, w_pad=0.0)
fig_actual_hbond.text(0.5, 0.0005, "Time(in ns)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_actual_hbond.text(0.001, 0.52, "hbond", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_actual_hbond.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_actual_hbond.savefig("hbond_actual.jpg", bbox_inches='tight')

fig_hbond.tight_layout(h_pad=0.0, w_pad=0.0)
fig_hbond.text(0.5, 0.0005, "Time(in ns)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_hbond.text(0.001, 0.52, "hbond", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_hbond.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_hbond.savefig("hbond.jpg", bbox_inches='tight')

fig_actual_saltbridge.tight_layout(h_pad=0.0, w_pad=0.0)
fig_actual_saltbridge.text(0.5, 0.0005, "Time(in ns)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_actual_saltbridge.text(0.001, 0.52, "Salt-Bridge", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_actual_saltbridge.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_actual_saltbridge.savefig("saltbridge_actual.jpg", bbox_inches='tight')

fig_saltbridge.tight_layout(h_pad=0.0, w_pad=0.0)
fig_saltbridge.text(0.5, 0.0005, "Time(in ns)", ha='center', va='center', fontweight='bold', fontsize=40)
fig_saltbridge.text(0.001, 0.52, "Salt-Bridge", ha='center', va='center', rotation='vertical', fontweight='bold', fontsize=40)
for ax in axes_saltbridge.flat:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
plt.subplots_adjust(wspace=0, hspace=0)
fig_saltbridge.savefig("saltbridge.jpg", bbox_inches='tight')

def get_min_max(scores):
    all_values = np.array([score for alg_scores in scores.values() for score in alg_scores])
    return np.min(all_values), np.max(all_values)

plt.rcParams.update({'font.size': 10})
fig, axes = plt.subplots(5, 2, figsize=(10, 15), sharex=True)
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

plot_bars(axes[0], backbone_rmsd_property_scores, 'A')
plot_bars(axes[1], backbone_rmsd_label_scores, 'B')
plot_bars(axes[2], mana_rmsd_property_scores, 'C')
plot_bars(axes[3], mana_rmsd_label_scores, 'D')
plot_bars(axes[4], distances_property_scores, 'E')
plot_bars(axes[5], distances_label_scores, 'F')
plot_bars(axes[6], hbond_property_scores, 'G')
plot_bars(axes[7], hbond_label_scores, 'H')
plot_bars(axes[8], saltbridge_property_scores, 'I')
plot_bars(axes[9], saltbridge_label_scores, 'J')

plt.tight_layout()
plt.savefig("scores_property_label.jpg")

