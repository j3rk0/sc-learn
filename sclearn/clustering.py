from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import NearestNeighbors
import numpy as np
import networkx as nx


class PhenoClust(BaseEstimator, ClusterMixin):
    """
    implementation of PhenoGraph, a clustering method designed for high-dimensionality single cell data.
    Use scikit-learn for nearest neighbors estimation and networkx for community detection algorithms to
    scale efficently with data size.
    """
    def __init__(self, k=15, metric='minkowski', p=2, resolution=2, n_jobs=-1, random_state=666):
        """
        constructor
        :param k:  number of neighbors considered to build the graph
        :param metric: distance metric
        :param p:  parameter for minkowski distance
        :param resolution: louvain algorithm parameter if this value is set < 1 larger ( and fewer)
               communities are detected. if set > 1 smaller ( and more) communities will be detected
        :param n_jobs:  number of parallel process
        :param random_state: random seed
        """
        self.k = k
        self.resolution = resolution
        self.metric = metric
        self.p = p
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.nn_ = None
        self.graph_ = None
        self.comms_ = None

    def fit(self, X, y=None):
        """
        fit a clustering model using phenograph
        :param X: dataset
        :param y: unused, only for compatibility
        :return: fitted estimator
        """
        # build netowrk from datapoints
        self.nn_ = NearestNeighbors(n_neighbors=self.k, metric=self.metric, p=self.p, n_jobs=self.n_jobs)
        self.nn_.fit(X)
        g = self.nn_.kneighbors_graph(X).toarray()

        # compute jaccard coefficients and set it as the weight of the graph
        g = nx.from_numpy_array(g)
        for (i, j, w) in nx.jaccard_coefficient(g, g.edges):
            g[i][j]['weight'] = w

        self.graph_ = g
        # using louvian to detect communities
        self.comms_ = nx.community.louvain_communities(g, resolution=self.resolution, seed=self.random_state)
        return self

    def fit_predict(self, X, y=None):
        """
        fit the model and return predicted clusters
        :param X: dataset
        :param y: unused, only for compatibility
        :return: ndarray indicating predicted cluster
        """
        self.fit(X)
        ret = np.zeros(X.shape[0])
        for i in range(len(self.comms_)):
            for j in self.comms_[i]:
                ret[j] = int(i)
        return ret

    def predict(self, X, y=None):
        """
        assign samples unseen during training to fitted clusters.
        for each sample the neighbors and the edges weights are computed. Using this information
        the node representing the sample is added to the graph and is assigned to the neighbor's
        community that minimize the modularity. Once the community is selected the node will be
        removed from the graph so that the next sample can be processed.

        :param X: ndarray (N_sample,N_features)
        :param y: ignored
        :return:  ndarray of labels of shape (N_sample)
        """
        pred = np.zeros(X.shape[0])

        # get graph representation of new nodes
        adjmat = self.nn_.kneighbors_graph(X)

        # add a temp node
        new_node_id = self.graph_.number_of_nodes()
        self.graph_.add_node(new_node_id)

        # iterate over all data
        for i in range(adjmat.shape[0]):

            # compute edges to add
            new_edges = [(new_node_id, v) for v in adjmat[i, :].indices]

            # compute neighbor communities
            neig_comms = [c for c in range(len(self.comms_)) if self.comms_[c].intersection(adjmat[i, :].indices)]

            if len(neig_comms) == 1:  # there is only 1 neighbor community
                pred[i] = neig_comms[0]
            else:
                self.graph_.add_edges_from(new_edges)  # add new edges

                for (u, v, w) in nx.jaccard_coefficient(self.graph_, new_edges):
                    self.graph_[u][v]['weight'] = w  # compute weights

                best_mod = -np.inf
                for c in neig_comms:  # for each neighbor community
                    self.comms_[c].add(new_node_id)  # add new node to community
                    # compute modularity
                    mod = nx.community.modularity(self.graph_, self.comms_, resolution=self.resolution)
                    if mod > best_mod:  # update result if improvements are found
                        best_mod = mod
                        pred[i] = c
                    self.comms_[c].remove(new_node_id)  # remove node from community

                self.graph_.remove_edges_from(new_edges)  # remove new edges
        self.graph_.remove_node(new_node_id)  # remove temp node
        return pred
