# To perform clustering on employees to group similar employees together.

import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

To group employees into clusters, a hierarchical clustering approach is taken.

data_clust = hr_data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company', 'Work_accident', 'left', 'promotion_last_5years', 'sales_coded', 'salary_coded']]

## Visualizing the hierarchical clustering with a dendrogram
dend = shc.dendrogram(shc.linkage(data_clust, method='ward'))

## The number of clusters may be taken as 6, since the longest distance without horizontal cuts give 6 clusters if broken into parts.
## Euclidean distance is further used for cluster assignments, with Ward's linkage.

cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')
pred_cluster = cluster.fit_predict(data_clust)

print('Cluster assignments: \n')
cluster_number, count = np.unique(pred_cluster, return_counts = True)
cluster_assignments = pd.DataFrame({'Cluster number': cluster_number, 'Total count': count})
print(cluster_assignments.to_string(index=False))
