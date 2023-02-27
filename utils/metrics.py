import torch
import numpy as np
import scanpy as sc

import matplotlib.pyplot as plt
# plt.ion(); plt.style.use('seaborn-pastel')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


import os, scvi
data_dir = "data/COVID_Stephenson/"

#####################
# Reconstruction error
from sklearn.metrics import mean_squared_error


def rmse(model, adata_test):
	imputed_values = model.get_normalized_expression(adata_test, return_mean=True)
	return mean_squared_error(adata_test.X, imputed_values)


########################
# Latent space/ data integration metrics from scib 
# most of these methods will be wrappers for or modified versions of scib metrics
# from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.cluster import KMeans
import scib

# #implement nmi, adapted from scVI benchmarking.py 
# # -> can also extend to use louvain clustering (as they do in scib)
def kmeans_clustering(adata, label_key, embed_key, cluster_key):
	if embed_key not in adata.obsm.keys():
        print(adata.obsm.keys())
        raise KeyError(f"{embed_key} not in obsm")

    K = len(adata.obs[label_key].unique())

	latent_space = adata.obsm[embed_key]
	labels_pred = KMeans(K, n_init=200).fit_predict(latent_space)
	adata.obs[cluster_key] = labels_pred


# note: cluster functions to use are leiden, louvain, kmeans
# adapted optimal clustering function for latent space metrics from scib
def make_clusterings(adata, embed_key, label_key, k = 15,
			   neighbor_key = None, cluster_methods = ['kmeans', 'leiden'], # resolution is default for leiden clustering
			   force = False, **kwargs):
	# clusters and returns adata with knn graph and clusters
	if neighbor_key == None:
		neighbor_key = embed_key # set neighbor_key s.t. it's associated with latent space embedding method

	# make clusterings
	adata_copy = adata.copy()
	sc.pp.neighbors(adata_copy, use_rep=embed_key, key_added = neighbor_key, n_neighbors = k, copy = False)

	cluster_keys = [f'{embed_key}_{cluster_method}' for cluster_method in cluster_methods]
    for i in range(len(cluster_methods)):
    	cluster_key = cluster_keys[i]

    	# message if throwing error
    	if cluster_key in adata_copy.obs.columns:
        	if force:
            	print(
                	f"WARNING: cluster key {cluster_key} already exists in adata.obs and will be overwritten because "
                	"force=True "
            	)
        	else:
            	raise ValueError(
                	f"cluster key {cluster_key} already exists in adata, please remove the key or choose a different "
                	"name. If you want to force overwriting the key, specify `force=True` "
            	)

        # add clusterings to data
		if(cluster_methods[i] == 'kmeans'):
			kmeans_clustering(adata_copy, label_key, embed_key, cluster_key)
		elif(cluster_methods[i] =='leiden'): # use default res = 1
			sc.tl.leiden(adata_copy, neighbors_key = neighbor_key, key_added=cluster_key)
		elif(cluster_methods[i]  == 'louvain'): # use default res = 1
			sc.tl.louvain(adata_copy, neighbors_key = neighbor_key, key_added=cluster_key)
		else:
			raise ValueError(
				f"clustering does not handle {cluster_methods[i]}, please choose among kmeans, leiden, and louvain"
			)
	return adata_copy, cluster_keys


# 1. bio conservation metrics
# 1.1 label-dependent
# TODO: figure out how to do the neighbors in a data efficient manner
def calc_bio_metrics(adata, embed_key, 
				batch_key = 'Site', label_key = 'harmonized_celltype', k = 15,
				cluster_methods = ['kmeans', 'leiden'], # line for clustering 
				metrics_list = ['nmi', 'ari', 'iso_labels_f1', 'cellASW', 'iso_labels_asw', 'cLisi']):
	
	adata_copy, cluster_keys = make_clusterings(adata, embed_key = embed_key,label_key = label_key, k = k, cluster_methods = cluster_methods)

	bio_metrics = {}
	if('nmi' in metrics_list):
		for i in range(len(cluster_keys)):
			bio_metrics[f'nmi_{cluster_methods[i]}'] = scib.me.nmi(adata_copy, cluster_key=cluster_keys[i], label_key=label_key) 
		print(bio_metrics)
	if('ari' in metrics_list):
		for i in range(len(cluster_keys)):
			bio_metrics[f'ari_{cluster_methods[i]}'] = scib.me.ari(adata_copy, cluster_key=cluster_keys[i], label_key=label_key) 
		print(bio_metrics)
	if('iso_labels_f1' in metrics_list):
		bio_metrics['iso_labels_f1'] = scib.me.isolated_labels_f1(adata_copy, batch_key=batch_key, label_key=label_key, embed = embed_key)
		print(bio_metrics)
	if('cellASW' in metrics_list):
		bio_metrics['cellASW'] = scib.me.silhouette(adata_copy, label_key = label_key, embed=embed_key)
		print(bio_metrics)
	if('iso_labels_asw' in metrics_list):
		bio_metrics['iso_labels_asw'] = scib.me.isolated_labels_asw(adata_copy, batch_key=batch_key, label_key=label_key, embed=embed_key)
		print(bio_metrics)
	if('cLisi' in metrics_list):
		bio_metrics['cLisi'] = scib.me.clisi_graph(adata_copy, label_key=label_key, type_="embed", use_rep=embed_key)
		print(bio_metrics)

	return bio_metrics, adata_copy

# def nmi(adata, embed_key, label_key, 
# 		neighbor_key = None, cluster_key, cluster_fn_name = 'KMeans',
# 		force = False, verbose = True):
# 	# nmi with optimal clustering
# 	# default clustering is KMeans with n_init = 200
# 	# if Louvain or Leiden clustering specified, resolutions are optimized for 10 vals ranging between 0.1 and 2
# 	return clustering(data, embed_key, label_key, scib.me.nmi,
# 			 			neighbor_key, cluster_key, cluster_fn_name = cluster_fn_name,
# 			   			force = force, verbose = verbose)
     


# def ari(adata, embed_key, label_key,
# 		neighbor_key = None, cluster_key, cluster_fn_name = 'KMeans',
# 		force = False, verbose = True): 
# 	# use sklearn's ARI implementation
# 	# Similarity score between -0.5 and 1.0. 
# 	# Random labelings have an ARI close to 0.0,  1.0 stands for perfect match.
# 	return clustering(data, embed_key, label_key, scib.me.ari,
# 			 			neighbor_key, cluster_key, cluster_fn_name = cluster_fn_name,
# 			   			force = force, verbose = verbose)


# def iso_labels_f1(adata, embed_key, label_key, batch_key):
# 	sc.pp.neighbors(adata, use_rep=embed_key)
# 	return scib.me.isolated_labels_f1(adata, batch_key=batch_key, label_key=label_key)

# def asw(adata, embed_key, label_key):
# 	return scib.me.silhouette(adata, label_key = label_key, embed=embed_key)

# def cLisi(adata, embed_key, label_key):
# 	return scib.me.clisi_graph(adata, label_key=label_key, type="embed", use_rep=embed_key)

# def iso_labels_asw(adata, embed_key, label_key, batch_key):
# 	return scib.me.isolated_labels_asw(adata, batch_key=batch_key, label_key=label_key, embed=embed_key)


# 2. batch removal metrics
def calc_batch_metrics(adata, embed_key, 
				  batch_key = 'Site', label_key = 'harmonized_celltype', k = 15, 
				  metrics_list = ['batchASW', 'iLisi', 'graph_connectivity']):
	# default: include all metrics
	# knn graph constructed with k = 15, euclidean metric
	batch_metrics = {}
	print('batch metrics for', embed_key)
	if('batchASW' in metrics_list):
		batch_metrics['batchASW']  = scib.me.silhouette_batch(adata, batch_key=batch_key, label_key=label_key, embed=embed_key)
		print('batchASW:', batch_metrics['batchASW'])
	if('iLisi' in metrics_list):
		batch_metrics['iLisi'] = scib.me.ilisi_graph(adata, batch_key=batch_key, type_="embed", use_rep=embed_key)
		print('iLisi score:', batch_metrics['iLisi'])
	if('kBET' in metrics_list): # issues with lacking R implementation
		batch_metrics['kBET'] = scib.me.kBET(adata, batch_key=batch_key, label_key=label_key, type_="embed", embed=embed_key)
		print('kBET score:', batch_metrics['kBET'])
	if('graph_connectivity' in metrics_list):
		adata_copy = sc.pp.neighbors(adata, use_rep=embed_key, n_neighbors = k, copy = True)
		batch_metrics['graph_connectivity'] = scib.me.graph_connectivity(adata_copy, label_key=label_key)
		print('Graph connectivity score:', batch_metrics['graph_connectivity'])
	return batch_metrics
##########################
# imputation functions # from scVI replication code
def dropout(X, rate=0.1):
    """
    X: original testing set
    ========
    returns:
    X_zero: copy of X with zeros
    i, j, ix: indices of where dropout is applied
    """
    X_zero = np.copy(X)
    # select non-zero subset
    i,j = np.nonzero(X_zero)
    
    # choice number 1 : select 10 percent of the non zero values (so that distributions overlap enough)
    ix = np.random.choice(range(len(i)), int(np.floor(0.1 * len(i))), replace=False)
    X_zero[i[ix], j[ix]] *= np.random.binomial(1, rate)
       
    # choice number 2, focus on a few but corrupt binomially
    #ix = np.random.choice(range(len(i)), int(slice_prop * np.floor(len(i))), replace=False)
    #X_zero[i[ix], j[ix]] = np.random.binomial(X_zero[i[ix], j[ix]].astype(np.int), rate)
    return X_zero, i, j, ix

# IMPUTATION METRICS
def imputation_error(X_mean, X, X_zero, i, j, ix):
    """
    X_mean: imputed dataset
    X: original dataset
    X_zero: zeros dataset
    i, j, ix: indices of where dropout was applied
    ========
    returns:
    median L1 distance between datasets at indices given
    """
    all_index = i[ix], j[ix]
    x, y = X_mean[all_index], X[all_index]
    return np.median(np.abs(x - y))
##########################
# wrapper function:

# # adapted from scvi replicating expts benchmarking.py
# def entropy_batch_mixing(adata, label_key, embed, batches):
#     def entropy(hist_data):
#         n_batches = len(np.unique(hist_data))
#         if n_batches > 2:
#             raise ValueError("Should be only two clusters for this metric")
#         frequency = np.mean(hist_data == 1)
#         if frequency == 0 or frequency == 1:
#             return 0
#         return -frequency * np.log(frequency) - (1 - frequency) * np.log(1 - frequency)

#     nne = NearestNeighbors(n_neighbors=51)
#     nne.fit(adata[embed])
#     kmatrix = nne.kneighbors_graph(latent_space) - scipy.sparse.identity(latent_space.shape[0])

#     score = 0
#     for t in range(50):
#         indices = np.random.choice(np.arange(latent_space.shape[0]), size=100)
#         score += np.mean([entropy(batches[kmatrix[indices].nonzero()[1]\
#                                  [kmatrix[indices].nonzero()[0] == i]]) for i in range(100)])
#     return score / 50.

##########################
# testing
adata = sc.read_h5ad(data_dir + "Stephenson.subsample.100k.h5ad")
adata_ref = adata.copy()

linear_scvi = scvi.model.LinearSCVI.load('models/linearSCVI.pt', adata_ref) 
adata_ref.obsm["X_linear_scVI"] = linear_scvi.get_latent_representation()
adata.obsm["X_linear_scVI"] = linear_scvi.get_latent_representation(adata_ref)

# gplvm = torch.load_#
linear_scVI_batch_metrics = calc_bio_metrics(adata, embed_key = 'X_linear_scVI', batch_key = 'Site', label_key = 'harmonized_celltype')

nonlinear_nb_scVI_batch_metrics = calc_batch_metrics(adata, embed_key = 'X_nonlinear_nb_scVI', batch_key = 'Site', label_key = 'harmonized_celltype')



