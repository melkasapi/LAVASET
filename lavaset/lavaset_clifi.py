from best_cut_node import best_cut_node
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KDTree, NearestNeighbors
from sklearn.decomposition import PCA
import pandas as pd
from numpy.linalg import svd
from sklearn.preprocessing import StandardScaler, normalize
import pandas as pd
import random
from joblib import Parallel, delayed
import csv
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.collections as mpl_collections
import matplotlib.path as mpl_path
# import networkx as nx
import os

class LAVASET: 

    def __init__(self, n_neigh, distance=False, minparent=2, minleaf=1, nvartosample=None, ntrees=100, nsamtosample=None, method='g', oobe=False, weights=None):
        self.n_neigh = n_neigh+1
        self.distance = distance
        self.minparent = minparent
        self.minleaf = minleaf
        self.nvartosample = nvartosample
        self.ntrees = ntrees
        self.nsamtosample = nsamtosample
        self.method = method
        self.oobe = oobe
        self.weights = weights

    def knn_calculation(self, data, data_type):

        """ This function calculated the nearest neighbors based on either distance matrix, 1D data (signals or spectra), VCG data,
        3D data (like images) or any other type of data.

        Args: data (numpy array or pandas dataframe/columns): the data to calculate the nearest neighbors for
            data_type (str): the type of data to calculate the nearest neighbors for. 
            It can be either 'distance_matrix', '1D', 'VCG', or 'other' where it calculates the nearest neighbors based on the 2D data input.
            'distance_matrix' is used for distance matrix input, '1D' is used for 1D data like signals or spectra, 'VCG' is used for VCG data, 
            and 'other' is used for any other type of data. 
        """
        if self.distance != False and data_type=='distance_matrix':       #### here we take a distance matrix 
            nn = NearestNeighbors().fit(data)
            points = nn.radius_neighbors(data, self.distance)[1]
            max_knn = []
            for i in points:
                max_knn.append(i.shape[0])
            max_knn = max(np.array(max_knn)) ### the maximum number of neighbors based on distance 
            self.n_neigh = max_knn
            return points
        elif data.ndim == 1 and data_type == '1D': ## for 1D data
            X = data.to_numpy(dtype=float)
            X = np.append([X], [data.to_numpy(dtype=float)], axis=0)
            kdtree = KDTree(X.T)
            points = kdtree.query(X.T,self.n_neigh)[1]
            return points
        elif data.ndim == 2 and data_type == 'VCG': # for VCG data
            m = int(data.shape[1])
            original_array = np.arange((data[:, :int(m/3)].shape[1]))
            n = self.n_neigh # Number of points to take from each side
            result_array = []
            for i, num in enumerate(original_array):
                lower_bound = (num - n) % original_array.shape[0]
                upper_bound = (num + n + 1) % original_array.shape[0]
                if upper_bound > lower_bound:
                    subarray = original_array[lower_bound:upper_bound]
                else:
                    subarray = np.concatenate((original_array[lower_bound:], original_array[:upper_bound]))
                result_array.append(subarray)
            result_array = np.array(result_array)
            neighbors = np.zeros((m, 3*result_array[0].shape[0]), dtype=int) # creating neigbors array for each of the points (all XYZ) and their neighbors
            for i in range(result_array.shape[0]):
                neighbors[i] = np.array((result_array[i], result_array[i]+750, result_array[i]+1500)).ravel() 
                neighbors[i+750] = np.array((result_array[i], result_array[i]+750, result_array[i]+1500)).ravel() # adding 750 to each of the points to get the YZ points too
                neighbors[i+1500] = np.array((result_array[i], result_array[i]+750, result_array[i]+1500)).ravel()
            self.n_neigh = 3*result_array[0].shape[0]
        else:
            X = data.to_numpy(dtype=float)
            kdtree = KDTree(X.T)
            points = kdtree.query(X.T,self.n_neigh)[1]
            return points 

    def tree_fit(self, Data, Labels, knn, random_state, minparent, minleaf, nvartosample, method, weights):
        n = len(Labels)
        L = int(2 * np.ceil(n / minleaf) - 1)
        m = Data.shape[1]
        # n_neigh = knn.shape[1]
        nodeDataIndx = {0: np.arange(n)}

        nodeCutVar = np.zeros(int(L))
        nodeCutValue = np.zeros(int(L))

        # RelatedCutVar={key: np.zeros(self.n_neigh) for key in range(L)}
        # CenteredCutVar={key: np.zeros(self.n_neigh) for key in range(L)}
        # VarianceCutVar={key: np.zeros(self.n_neigh) for key in range(L)}
        # LoadingsCutVar={key: np.zeros(self.n_neigh) for key in range(L)}
        RelatedCutVar={key: dict() for key in range(L)}
        CenteredCutVar={key: dict() for key in range(L)}
        VarianceCutVar={key: dict() for key in range(L)}
        LoadingsCutVar={key: dict() for key in range(L)}
        
        nodeflags = np.zeros(int(L+1))
        nodelabel = np.full(int(L), np.inf)
        nodelabel = np.zeros(int(L))

        childnode = np.zeros(int(L))
        nodeflags[0] = 1
        giniii = np.zeros((m,3))
        prox=np.zeros(n)

        if method.lower() in ['c', 'g']:
            unique_labels = np.unique(Labels)
            max_label = len(unique_labels)
        else:
            max_label = None
        current_node = 0

        #------------------------------------------------------------------------------------------------#
        # Variables to be used in feature importance assignment
        class_feature_impt_pos = np.zeros((m, 1+(max_label)))
        class_feature_impt_neg = np.zeros((m, 1+(max_label)))
        class_feature_impt = np.zeros((m, 1+(max_label)))
        ###########################################################
        class_feature_all_val = np.zeros((1,1+(max_label)))
        ########################################################### 
        giniii_all_val = np.zeros((1,2))
        rows = []
        rows_gini = []        
        #------------------------------------------------------------------------------------------------#


        while nodeflags[current_node] == 1:
            free_node = np.where(nodeflags == 0)[0][0]
            currentDataIndx = nodeDataIndx[current_node]# samples in this node 

            if len(np.unique(Labels[currentDataIndx])) == 1:
                if method.lower() in ['c', 'g']:
                    nodelabel[current_node] = unique_labels[Labels[currentDataIndx[0]].astype(np.int64)]
                elif method.lower() == 'r':
                    nodelabel[current_node] = Labels[currentDataIndx[0]]
                nodeCutVar[current_node] = 0
                nodeCutValue[current_node] = 0
            else:
                if len(currentDataIndx) >= minparent:
                    random_instance = np.random.RandomState(random_state)
                    # node_var = random_instance.randint(0, m, nvartosample)
                    
                    node_var = random_instance.permutation(range(0,m))[:nvartosample]
                    #node_var = np.random.permutation(range(0,m))[:nvartosample]
                    giniii[node_var,0]+=1

                    if weights is not None:
                        Wcd = weights[currentDataIndx]
                    else:
                        Wcd = None
                    
                    #-------------------------------------------------------------------------#
                        # Calculating latent variables
                    # NV = np.zeros((nvartosample, self.n_neigh), dtype=int)
                    # MC = np.zeros((nvartosample, self.n_neigh))
                    # VA = np.zeros((nvartosample, self.n_neigh))
                    # P = np.zeros((nvartosample, self.n_neigh))
                    # scores = np.zeros(currentDataIndx.shape[0])
                    # X_pca = np.zeros((Data.shape[0], 1))  # initiating nested array with array number = # of samples
                    
                    NV = dict()
                    MC = dict()
                    VA = dict()
                    P = dict()
                    scores = np.zeros(currentDataIndx.shape[0])
                    X_pca = np.zeros((Data.shape[0], 1)) 

                    for i, feat in enumerate(node_var):
                        NV[i] = knn[feat] # a dict of all the variables picked for s node (node_var) and their neighbors in the form of list (the first element of each list is the feature originally picked) 
                        matrix = Data[currentDataIndx][:, NV[i]]
                        # matrix_mean = np.mean(matrix, axis=0)
                        # matrix_std = np.std(matrix, axis=0)
                        # pca_df = (matrix - matrix_mean) / matrix_std
                        scaler = StandardScaler()
                        pca_df = scaler.fit_transform(matrix)

                        # Perform the Singular Value Decomposition (SVD)
                        u, s, vt = svd(pca_df, full_matrices=False)
                        U = np.matrix(u[:,0])
                        loadings = np.conj(U@pca_df)
                        # loadings = abs(np.array(loadings).ravel())
                        # loadings_norm = (loadings - np.min(loadings)) / (np.max(loadings)-np.min(loadings))
                        lsv = np.array((pca_df@loadings.T)).ravel()
                        scores = np.vstack((scores, lsv))
                        P[i] = np.array(loadings).ravel()
                        # MC[i] = matrix_mean #np.ravel(scaler.mean_)
                        # VA[i] = matrix_std #np.ravel(scaler.var_)
                        MC[i] = np.array(scaler.mean_).ravel()
                        VA[i] = np.array(scaler.var_).ravel()
                    scores = scores[1:, :].T
                    bestCutVar, bestCutValue = best_cut_node(method, scores, Labels[currentDataIndx], minleaf, max_label)
                    bestCutVar = int(bestCutVar)
                    random_state+=1
                    #------------------------------------------------------------------------------------------------------#

                    #bestCutVar here is the index from the node_var variables 
                    if bestCutVar != -1:
                        #-----------------------------#
                        # Feature importance variable assignment
                        gini_temp_array = np.zeros((1,2))
                        gini_temp_array[0,0] = node_var[bestCutVar]
                        #-----------------------------#

                        nodeCutVar[current_node] = node_var[bestCutVar] # actual feature name 
                        nodeCutValue[current_node] = bestCutValue
                        giniii[NV[bestCutVar],1]+= 1
                        RelatedCutVar[current_node]=NV[bestCutVar]
                        CenteredCutVar[current_node]=MC[bestCutVar]
                        VarianceCutVar[current_node]=VA[bestCutVar]
                        LoadingsCutVar[current_node]=P[bestCutVar]
                        
                        nodeDataIndx[free_node] = currentDataIndx[scores[:, bestCutVar] <= bestCutValue]
                        nodeDataIndx[free_node+1] = currentDataIndx[scores[:, bestCutVar] > bestCutValue]
                        
                        y_left=Labels[nodeDataIndx[free_node]]
                        y_right=Labels[nodeDataIndx[free_node+1]]
                        #y_parent = list(y_left)+list(y_right)
                        y_parent = np.array (list(y_left)+list(y_right))
        
                        proportion_left = len(y_left) / len(y_parent)
                        proportion_right = len(y_right) / len(y_parent)
                        p_parent = (np.bincount(np.array(y_parent, dtype=np.int64)))/len(y_parent)

                        p_left = (np.bincount(np.array(y_left, dtype=np.int64)))/len(y_left)
                        p_right = (np.bincount(np.array(y_right, dtype=np.int64)))/len(y_right)
                        gini_l = 1-np.sum(p_left**2)
                        gini_r = 1-np.sum(p_right**2)
                        gini_p = 1-np.sum(p_parent**2)
                        gini_gain = gini_p - (proportion_left*gini_l + proportion_right*gini_r)
                        # Labs = abs(P[bestCutVar])
                        # Pn = (Labs - np.min(Labs)) / (np.max(Labs)-np.min(Labs))

                        #Pn = first_loading_value / sum(abs(P[bestCutVar]))
                        Pn=abs(P[bestCutVar])/sum(abs(P[bestCutVar]))
                        for idx, feature in enumerate(NV[bestCutVar]):
                            giniii[feature, 2] += gini_gain * Pn[idx]
                            new_row = [feature, gini_gain* Pn[idx]]
                            rows_gini.append(new_row)
                        #giniii[NV[bestCutVar], 2] += gini_gain * Pn#abs(P[bestCutVar])

                        #new_row = [node_var[bestCutVar], gini_gain]
                        #rows_gini.append(new_row)

                        nodeflags[free_node:(free_node + 2)] = 1
                        childnode[current_node] = free_node

                        #############################################################################################
                        ####### Main Code for Class Feature Importance Direction algorith ######
                        #----------------------------------------------------------------------#
                        class_feature_impt_pos[node_var[bestCutVar], 0] += 1
                        class_feature_impt_neg[node_var[bestCutVar], 0] += 1
                        class_feature_impt[node_var[bestCutVar], 0] += 1

                        for j in range (max_label):
                            # Ancestor, Parent and perfect split calculations
                            ancestor = np.array (np.sum([Labels == j]))
                            parent = np.array (np.sum (y_parent == j))
                            if parent == 0:
                                normalized_Mi = np.nan
                                #class_feature_impt[NV[bestCutVar],j+1] += normalized_Mi*Pn
                                class_feature_impt[NV[bestCutVar],j+1] += 0
                            else:
                                Ei = parent/2
                                #Count of Left and Right nodes
                                Li = np.sum (y_left == j)
                                Ri = np.sum (y_right == j)
                                #G-test score for Left and Right no+des per class
                                if Li == 0: 
                                    M_Li = 2*Li*(np.log1p(Li/Ei))
                                else:
                                    M_Li = 2*Li*(np.log(Li/Ei))
                                if Ri == 0: 
                                    M_Ri = 2*Ri*(np.log1p(Ri/Ei))
                                else: 
                                     M_Ri = 2*Ri*(np.log(Ri/Ei))

                                Mi = M_Li + M_Ri

                                ###################################
                                di = np.sign(M_Ri - M_Li)
                                perfect_split = (2*parent*(np.log(2)))
                                normalized_magnitude = parent/ancestor
                                if Li > 0 or Ri > 0:
                                    
                                    normalized_Mi = ((Mi/perfect_split)*(normalized_magnitude))* di
                                    for idx, feature in enumerate(NV[bestCutVar]):
                                        class_feature_impt[feature,j+1] += normalized_Mi*Pn[idx]
                                        new_row = [feature, j, normalized_Mi*Pn[idx]]
                                        rows.append(new_row)

                                    #class_feature_impt[NV[bestCutVar],j+1] += normalized_Mi*Pn
                                    #class_feature_impt[NV[bestCutVar],j+1] += normalized_Mi
                                    #new_row = [node_var[bestCutVar], j, normalized_Mi*Pn]
                                    #rows.append(new_row)
                                else: 
                                    normalized_Mi = 0
                                    for idx, feature in enumerate(NV[bestCutVar]):
                                        class_feature_impt[feature,j+1] += normalized_Mi*Pn[idx]
                                        new_row = [feature, j, normalized_Mi*Pn[idx]]
                                        rows.append(new_row)

                                    #class_feature_impt[NV[bestCutVar],j+1] += normalized_Mi*Pn
                                    #class_feature_impt[NV[bestCutVar],j+1] += normalized_Mi
                                    #new_row = [node_var[bestCutVar], j, np.nan]
                                    #rows.append(new_row)
                        ############################################################################################
                                                            # END #
                        #------------------------------------------------------------------------------------------#                                  

                    else:
                        if method.lower() in ['c', 'g']:
                            leaf_label = np.argmax(np.bincount(Labels[currentDataIndx], minlength=max_label))
                            nodelabel[current_node] = unique_labels[leaf_label]
                        elif method.lower() == 'r':
                            nodelabel[current_node] = np.mean(Labels[currentDataIndx])

                else:

                    if method.lower() in ['c', 'g']:
                        leaf_label = np.argmax(np.bincount(Labels[currentDataIndx].astype(np.int64), minlength=max_label))
                        nodelabel[current_node] = unique_labels[leaf_label]
                    elif method.lower() == 'r':
                        nodelabel[current_node] = np.mean(Labels[currentDataIndx])
            current_node+=1
            random_state+=1

        #feat_impo = giniii
        feat_impo = pd.DataFrame(rows_gini, columns=['feature', 'gini_gain'])
        class_feature_impt_disbu = pd.DataFrame(rows, columns=['feature', 'class', 'v-value'])
        #class_feature_impt_disbu = class_feature_impt_disbu[class_feature_impt_disbu['class'] != 0]

        return nodeCutVar, nodeCutValue, childnode, nodelabel, feat_impo, class_feature_impt, class_feature_impt_disbu, RelatedCutVar, CenteredCutVar, VarianceCutVar, LoadingsCutVar


    def tree_predict(self, Data, cut_var, cut_val, nodechilds, nodelabel, RelatedCutVar, CenteredCutVar, VarianceCutVar, LoadingsCutVar, proximity):
        n, m = Data.shape
        tree_output = np.zeros(n)
        proximity_matrix = np.zeros(n)

        for i in range(n):
            current_node = 0
            while nodechilds[current_node] != 0:
                cvar = RelatedCutVar[current_node]
                Centered = (Data[i, cvar] - CenteredCutVar[current_node]) / VarianceCutVar[current_node]
                score = np.dot(Centered, LoadingsCutVar[current_node])

                if score <= cut_val[current_node]:
                    current_node = int(nodechilds[current_node])
                else:
                    current_node = int(nodechilds[current_node]+1)
            tree_output[i] = nodelabel[current_node]
            #prox[i]=current_node
            if proximity == True:
                proximity_matrix[i] = current_node
        
        if proximity == True:
            return tree_output, proximity_matrix
        else:
            return tree_output

    def build_tree(self, i, random_state, Data, Labels, knn, nsamtosample, minparent, minleaf, method, nvartosample, weights, oobe):
        print(f'building tree {i}')
        random_instance = np.random.RandomState(random_state)
        TDindx = random_instance.choice(len(Labels), nsamtosample, replace=False)
        #print(TDindx)

        # Get unique values in Labels
        unique_values = np.unique(Labels)

        # Initialize TDindx with one index for each unique value
        # TDindx = np.concatenate([random_instance.choice(np.where(Labels == value)[0], 1) for value in unique_values])

        # If there are more samples to select, randomly select the rest without replacement
        if nsamtosample > len(unique_values):
            additional_indices = random_instance.choice(len(Labels), nsamtosample - len(unique_values), replace=False)
            TDindx = np.concatenate([TDindx, additional_indices])

        # Shuffle the indices to ensure randomness
        random_instance.shuffle(TDindx)

        Random_ForestT = self.tree_fit(Data[TDindx,:], Labels[TDindx], knn=knn, random_state=random_state, minparent=minparent, minleaf=minleaf, method=method, nvartosample=nvartosample, weights=weights)
        Random_ForestT_dict = {'tree_cut_var': Random_ForestT[0], 'tree_cut_val': Random_ForestT[1],
                                'tree_nodechilds': Random_ForestT[2], 'tree_nodelabel': Random_ForestT[3], 
                                'feature_importances': Random_ForestT[4], 'class_importance_uncomb_val': Random_ForestT[5],
                                'class_feature_impt_disbu': Random_ForestT[6], 'RelatedCutVar': Random_ForestT[7], 'CenteredCutVar':Random_ForestT[8], 
                                'VarianceCutVar': Random_ForestT[9], 'LoadingsCutVar': Random_ForestT[10],
                                'method': method, 'oobe':oobe}
        oobe_val = 1
        if oobe:
            NTD = np.setdiff1d(np.arange(len(Labels)), TDindx)
            tree_output = self.tree_predict(Data[NTD,:], Random_ForestT_dict['tree_cut_var'], Random_ForestT_dict['tree_cut_val'],Random_ForestT_dict['tree_nodechilds'], 
            Random_ForestT_dict['tree_nodelabel'], Random_ForestT_dict['RelatedCutVar'], 
            Random_ForestT_dict['CenteredCutVar'], Random_ForestT_dict['VarianceCutVar'], Random_ForestT_dict['LoadingsCutVar'], proximity=False)

            if method in ['c', 'g']:
                # oobe_val = np.mean(tree_output != Labels[NTD])
                oobe_val = np.sum(tree_output - Labels[NTD] == 0) / len(NTD)
            elif method == 'r':
                oobe_val = np.mean(np.square(tree_output - Labels[NTD]))

            Random_ForestT_dict['oobe'] = oobe_val

        return Random_ForestT_dict


    def fit_lavaset(self, Data, Labels, knn, random_state):

        if self.nsamtosample is None:
            self.nsamtosample = Data.shape[0]
        elif self.nsamtosample > 1:
            self.nsamtosample = self.nsamtosample
        else:
            self.nsamtosample = int(Data.shape[0]*self.nsamtosample)
        
        if self.nvartosample is None:
            self.nvartosample = Data.shape[1]
        elif self.nvartosample == 'sqrt':
            self.nvartosample = int(np.sqrt(Data.shape[1]))

       
        Random_Forest = []
        Random_Forest = Parallel(n_jobs=-1, verbose=10)(delayed(self.build_tree)(i, random_state+i, Data, Labels, knn,
        self.nsamtosample, self.minparent, self.minleaf, self.method, self.nvartosample, self.weights, self.oobe) for i in range(self.ntrees))

        return Random_Forest


    def predict_lavaset(self, Data, Random_Forest):
        """
        Returns the output of the ensemble (f_output) as well
        as a [num_treesXnum_samples] matrix (f_votes) containing
        the outputs of the individual trees.
        The 'oobe' flag allows the out-of-bag error to be used to 
        weight the final response (only for classification).

        Args:
        - Data: numpy array of shape (num_samples, num_features) containing the samples
        - Random_Forest: list of CARTree objects
        - oobe: bool flag for out-of-bag error calculation (default=False)

        Returns:
        - f_output: numpy array of shape (num_samples,) containing the output of the ensemble
        - f_votes: numpy array of shape (num_trees, num_samples) containing the output of each tree
        """
        n, m = Data.shape
        proximity_matrix = np.zeros((n, len(Random_Forest)))

        f_votes = np.zeros((len(Random_Forest), Data.shape[0]), dtype=object)
        oobe_values = np.zeros((len(Random_Forest),))
        for i, tree in enumerate(Random_Forest):
            x, proximity_matrix[:, i] = self.tree_predict(Data, Random_Forest[i]['tree_cut_var'], Random_Forest[i]['tree_cut_val'],
                                                          Random_Forest[i]['tree_nodechilds'], Random_Forest[i]['tree_nodelabel'], 
                                                          Random_Forest[i]['RelatedCutVar'], Random_Forest[i]['CenteredCutVar'], 
                                                          Random_Forest[i]['VarianceCutVar'], Random_Forest[i]['LoadingsCutVar'], 
                                                          proximity=True)
            f_votes[i,:] = x.ravel()
            #f_votes[i, :] =  self.tree_predict(Data, Random_Forest[i]['tree_cut_var'], Random_Forest[i]['tree_cut_val'],Random_Forest[i]['tree_nodechilds'], 
            #Random_Forest[i]['tree_nodelabel'], Random_Forest[i]['RelatedCutVar'], 
            #Random_Forest[i]['CenteredCutVar'], Random_Forest[i]['VarianceCutVar'], Random_Forest[i]['LoadingsCutVar']).ravel()
            oobe_values[i] = Random_Forest[i]['oobe']
        
        method = Random_Forest[0]['method']

        if method in ['c', 'g']:
            unique_labels, indices = np.unique(f_votes, return_inverse=True)
            f_votes = indices.reshape((len(Random_Forest), Data.shape[0]))

            # if oobe:
            #     weights = ~oobe + oobe * oobe_values
            # else:
            #     weights = None
            weights = None
            f_output = np.apply_along_axis(lambda x: np.bincount(x, weights=weights, minlength=len(unique_labels)).argmax(),
                                        axis=0, arr=f_votes)
            f_output = unique_labels[f_output]
        elif method == 'r':
            f_output = np.mean(f_votes, axis=0)
        
        oobe_mean = np.mean(oobe_values)
        return f_output, f_votes, oobe_mean, proximity_matrix

    def feature_evaluation (self, Data, Random_Forest):
        all_importances = np.zeros((Data.shape[1], 3))
        for i, tree in enumerate(Random_Forest):
            importance_per_tree = np.array(Random_Forest[i]['feature_importances'])
            all_importances += importance_per_tree
        return all_importances
    
    #Extract Summed V-values: 
    def class_feature_evaluation (self, Data, Random_Forest, nclasses):
        all_class_features = np.zeros(((Data.shape[1], 1+nclasses)))
        for i, tree in enumerate(Random_Forest):
             class_impt_per_tree = np.array(Random_Forest[i]['class_importance_uncomb_val'])
             all_class_features += class_impt_per_tree

        return all_class_features
    
    def class_feature_distribution_evaluation (self, Data, Random_Forest, nclasses):
        all_class_features = pd.DataFrame(columns=['feature', 'class', 'v-value'])
        for i, tree in enumerate(Random_Forest):
             class_impt_per_tree = np.array(Random_Forest[i]['class_feature_impt_disbu'])
             class_impt_per_tree = pd.DataFrame(class_impt_per_tree, columns=['feature', 'class', 'v-value'])
             all_class_features = pd.concat([all_class_features, class_impt_per_tree], ignore_index=True)

        return all_class_features

 