import numpy as np
import matplotlib.pyplot as plt
from Clustering import Clustering
from properties import IsingProperties
from scoring import scoring 
Clustering = Clustering()
lattice_size=30
Ising_properties = IsingProperties(lattice_size=30)
loaded_array=np.load("representation_macrostates.npy")
properties={'specific_heat':Ising_properties.specific_heat,'sus':Ising_properties.sus}
Clustering_Algorithms={'KMeans':Clustering.KMeans, 'Agglomerative Clustering':Clustering.AgglomerativeClustering,'Gaussian Mixture':Clustering.GaussianMixture, 'Birch Clustering':Clustering.BirchClustering}
n_clusters=[2,3,4,5,6]
final_lattice_arr=[]
temperatures = np.linspace(1.0, 5.0, 200)
for j in range (0,len(temperatures)):
    temp = str(temperatures[j])
    part2 = "_macrostates"
    ext = ".npy"
    combined_name = temp + part2 + ext
    final_arr=np.load(combined_name)
    final_lattice_arr.append(final_arr)

final_comb_array = np.stack(final_lattice_arr, axis=0)
final_comb_array = final_comb_array[:,99::100,:,:]
cv=properties['specific_heat'](final_comb_array,temperatures)
sus=properties['sus'](final_comb_array,temperatures)
xticks=np.linspace(0,len(final_comb_array),10)
xticklabels=np.linspace(1.0,5.0,10)
cv_property_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
cv_label_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
sus_property_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
sus_label_scores = {alg: [] for alg in Clustering_Algorithms.keys()}
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(cv, marker='o', linestyle='',color='b')
ax1.set_xlabel('Temperature')
ax1.set_ylabel('Specific Heat')
ax1.set_xticks(xticks)
ax1.set_xticklabels(np.round(xticklabels,2))
ax1.grid(True, axis='x')
ax2.plot(sus,marker='o', linestyle='',color='b')
ax2.set_xlabel('Temperature')
ax2.set_ylabel('Susceptibility')
ax2.set_xticks(xticks)
ax2.set_xticklabels(np.round(xticklabels,2))
ax2.grid(True, axis='x')
plt.tight_layout()
plt.savefig("combined_cv_sus.jpg")
plt.show()
fig, axes = plt.subplots(4, len(n_clusters), figsize=(40, 40))
temperature=np.arange(len(final_comb_array))
axes = axes.flatten()
#plt.rcParams.update({'font.size': 30})
fig_actual_cv, axes_actual_cv = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_actual_sus, axes_actual_sus = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_cv, axes_cv = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
fig_sus, axes_sus = plt.subplots(4, len(n_clusters), figsize=(45, 40), sharex='all', sharey='all')
plt.subplots_adjust(wspace=0, hspace=0)
subplot_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
row_labels = ['A', 'B', 'C', 'D']
col_labels = ['(i)', '(ii)', '(iii)', '(iv)', '(v)']
x_ax=0
for name,algorithm in Clustering_Algorithms.items():
    y_ax=0
    for n in n_clusters:
        cluster_labels=algorithm(loaded_array,n)
        
        cv=properties['specific_heat'](final_comb_array,temperatures)
        sus=properties['sus'](final_comb_array,temperatures)
        cv=cv*lattice_size*lattice_size
        sus=sus*lattice_size*lattice_size
        cv_property,cv_label,cv_clusters=scoring(cv,cluster_labels,name,n,0.40,algorithm)
        sus_property,sus_label,sus_clusters=scoring(sus,cluster_labels,name,n,0.40,algorithm)
        cv_property_scores[name].append(cv_property)
        cv_label_scores[name].append(cv_label)
        sus_property_scores[name].append(sus_property)
        sus_label_scores[name].append(sus_label)
        
        y_ax=y_ax+1
    x_ax=x_ax+1
plt.savefig("clusters_cv_sus.jpg")

xticks = np.linspace(0, len(final_comb_array), 5)
xticklabels = ['1.0', '2.0', '3.0', '4.0', '5.0']

def weighted_average_dicts(d_1, d_2, w1, w2):
    # Ensure that w1 + w2 = 1
    if not (w1 + w2 == 1):
        raise ValueError("Weights w1 and w2 must sum to 1")

    # Initialize the result dictionary
    weighted_avg_dict = {}

    # Iterate over the keys and compute the weighted average
    for key in d_1:
        multiplied_d_1 = [x * w1 for x in d_1[key]]
        multiplied_d_2 = [x * w2 for x in d_2[key]]
        weighted_score = [x + y for x, y in zip(multiplied_d_1,multiplied_d_2)]
        weighted_avg_dict[key] = weighted_score

    return weighted_avg_dict

w1 = 0.5
w2 = 0.5

result_average_property = weighted_average_dicts(cv_property_scores, sus_property_scores, w1, w2)
result_average_label = weighted_average_dicts(cv_label_scores, sus_label_scores, w1, w2)

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
plt.savefig("overall_scores_cv_sus.jpg")

