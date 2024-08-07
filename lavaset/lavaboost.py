from best_cut_node import best_cut_node
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KDTree, NearestNeighbors
from sklearn.decomposition import PCA
import pandas as pd
from numpy.linalg import svd
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from joblib import Parallel, delayed
import csv
from sklearn.tree import DecisionTreeRegressor 
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KDTree
import math
import matplotlib.pyplot as plt
import matplotlib.collections as mpl_collections
import matplotlib.path as mpl_path
import networkx as nx
import os


class LAVABOOST: 

    def __init__(self, n_neigh, distance=False, minparent=2, minleaf=1, nvartosample=None, ntrees=100, nsamtosample=None, method=None, oobe=False, weights=None, n_estimators=None, learning_rate=None, boosters=[], n_classes=None):
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
        self.n_estimators=n_estimators; 
        self.learning_rate=learning_rate
        self.boosters=boosters
        self.n_classes=n_classes

    def knn_calculation(self, data):
        """ if data.ndim == 1:
            X = data.to_numpy(dtype=float)
            X = np.append([X], [data.to_numpy(dtype=float)], axis=0)
            kdtree = KDTree(X.T)
            points = kdtree.query(X.T,self.n_neigh)[1]
            return points 
        else:
            X = data.to_numpy(dtype=float)
            kdtree = KDTree(X.T)
            points = kdtree.query(X.T,self.n_neigh)[1]
            return points """ 
        if self.distance != False: #### here we take a distance matrix 
            nn = NearestNeighbors().fit(data)
            points = nn.radius_neighbors(data, self.distance)[1]
            max_knn = []
            for i in points:
                max_knn.append(i.shape[0])
            max_knn = max(np.array(max_knn)) ### the maximum number of neighbors based on distance 
            self.n_neigh = max_knn
            print('here')
            return points
        elif data.ndim == 1:
            X = data.to_numpy(dtype=float)
            X = np.append([X], [data.to_numpy(dtype=float)], axis=0)
            kdtree = KDTree(X.T)
            points = kdtree.query(X.T,self.n_neigh)[1]
            return points
        elif data.ndim == 2: # for VCG data
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
            neighbors = np.zeros((m, 3*result_array[0].shape[0]), dtype=int) # creating neigbrs array for each of the points (all XYZ) and their neighbors
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

    def tree_fit(self, Data, Labels, TDindx, knn, random_state, minparent, minleaf, nvartosample, method, weights, negative_gradients, hessians, true_subset):
        n = len(Labels)
        # L = Number of leaves, so L=2^max_depth
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
        ###########################################################
        unique_true_labels = np.unique(true_subset)
        max_true_label = len(unique_true_labels)
        rows = []
        rows_gini = []   
        #------------------------------------------------------------------------------------------------#


        while nodeflags[current_node] == 1:
            
            '''
            Code before: free_node = np.where(nodeflags == 0)[0][0]
            Problem: Debugging showed that for the original LAVASET and Random Forest, 
            the current node would never go through all of the possible nodeflags.
            This caused the last few positions of node labels to not have a value assigned to them.

            Problem: For LAVABOOST I set the depth of the tree so, nodeflags is dependent on this.
            So the loop will reach the end of the free nodes avaliable before assigning the labels.
            Thus I made two main changes.
            
            free_node = np.where(nodeflags == 0)[0][0]
            ==>
            if np.any(nodeflags == 0):
                # free_node is the index of the first unvisited node (i.e., first occurrence of 0)
                free_node = np.where(nodeflags == 0)[0][0]
            else:
                free_node = None

            Added to the end
            if current_node ==L:
                break
            (L is the total number of leaves required to build a binary tree structure, 
            from the number of patients.)

            This is needed when the splits still contain different classes 
            and the program wants to continue splitting.
            '''
            #free_node = np.where(nodeflags == 0)[0][0]
            if np.any(nodeflags == 0):
                # free_node is the index of the first unvisited node (i.e., first occurrence of 0)
                free_node = np.where(nodeflags == 0)[0][0]
            else:
                free_node = None


            currentDataIndx = nodeDataIndx[current_node]# samples in this node 

            if len(np.unique(Labels[currentDataIndx])) == 1:
                if method.lower() in ['c', 'g']:
                    #nodelabel[current_node] = unique_labels[Labels[currentDataIndx[0]].astype(np.int64)]
                    negative_gradients_in_leaf = negative_gradients.take(TDindx[currentDataIndx], axis=0)
                    hessians_in_leaf = hessians.take(TDindx[currentDataIndx], axis=0)
                    
                    if np.sum(hessians_in_leaf) == 0:
                        val = 0  # Set a default value or handle division by zero appropriately
                    else:
                        val = np.sum(negative_gradients_in_leaf) / np.sum(hessians_in_leaf)
                    nodelabel[current_node] = val
                elif method.lower() == 'r':
                    #nodelabel[current_node] = Labels[currentDataIndx[0]]
                    negative_gradients_in_leaf = negative_gradients.take(TDindx[currentDataIndx], axis=0)
                    hessians_in_leaf = hessians.take(TDindx[currentDataIndx], axis=0)
                    
                    if np.sum(hessians_in_leaf) == 0:
                        val = 0  # Set a default value or handle division by zero appropriately
                    else:
                        val = np.sum(negative_gradients_in_leaf) / np.sum(hessians_in_leaf)
                    if current_node < len(nodelabel):
                        nodelabel[current_node] = val
                    else:
                        break
                    #nodelabel[current_node] = val
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
                        matrix_mean = np.mean(matrix, axis=0)
                        matrix_std = np.std(matrix, axis=0)
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
                        MC[i] = matrix_mean #np.ravel(scaler.mean_)
                        VA[i] = matrix_std #np.ravel(scaler.var_)
                        # MC[i] = np.ravel(scaler.mean_)
                        # VA[i] = np.ravel(scaler.var_)
                    scores = scores[1:, :].T 
                    bestCutVar, bestCutValue = best_cut_node(method, scores, Labels[currentDataIndx], minleaf, max_label)
                    #bestCutVar, bestCutValue = best_cut_node(method, Data[currentDataIndx][:, node_var], Labels[currentDataIndx], minleaf, max_label)
                    bestCutVar = int(bestCutVar)
                    random_state+=1
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

                        Pn=abs(P[bestCutVar])/sum(abs(P[bestCutVar]))
                        giniii[NV[bestCutVar], 2] += gini_gain * Pn#abs(P[bestCutVar])
                        for idx, feature in enumerate(NV[bestCutVar]):
                            giniii[feature, 2] += gini_gain * Pn[idx]
                            new_row = [feature, gini_gain* Pn[idx]]
                            rows_gini.append(new_row)
                        
                        nodeflags[free_node:(free_node + 2)] = 1
                        childnode[current_node] = free_node
                        #############################################################################################
                        ####### Main Code for Class Feature Importance Direction algorith ######
                        #----------------------------------------------------------------------#
                        class_feature_impt_pos[node_var[bestCutVar], 0] += 1
                        class_feature_impt_neg[node_var[bestCutVar], 0] += 1
                        class_feature_impt[node_var[bestCutVar], 0] += 1
                        true_left = nodeDataIndx[free_node]
                        true_right = nodeDataIndx[free_node+1]
                        true_parent = list(true_left)+list(true_right)

                        for j in range (max_true_label):
                            for x in range (max_label):
                                # Ancestor, Parent and perfect split calculations

                                # True/False of original labels (0,1) for positions of current j
                                ori_label_mask = (true_subset == j)
                                # True/False of error labels (....)  for positions of current error x also at the same position in original labels where position = j
                                error_label_mask = (Labels == unique_labels[x]) & ori_label_mask
                                # Combine the masks using logical AND
                                combined_ancestor_mask = ori_label_mask & error_label_mask
                                # Count the number of True values in the combined mask
                                count_of_ancestor_true_positions = np.sum(combined_ancestor_mask)

                                #Labels_subset_sum = count_of_j_in_true_positions
                                #ancestor = np.array (np.sum([true_subset == j]))
                                ancestor = np.array(count_of_ancestor_true_positions)

                                # Get original and error labels of parent
                                y_parent_ori = true_subset[true_parent]
                                y_parent_error = Labels[true_parent]
                                
                                # True/False of original parent labels (0,1) for positions of current j
                                ori_label_parent_mask = (y_parent_ori == j)
                                # True/False of error parent labels (....)  for positions of current error x also at the same position in original parent labels where position = j
                                error_label_parent_mask = (y_parent_error == unique_labels[x]) & ori_label_parent_mask
                                # Combine the masks using logical AND
                                combined_parent_mask = ori_label_parent_mask & error_label_parent_mask
                                # Count the number of True values in the combined mask
                                count_of_parent_true_positions = np.sum(combined_parent_mask)

                                #parent = np.array (np.sum (true_subset[true_parent] == j))
                                parent = np.array(count_of_parent_true_positions)

                                ancestor_class_sum = np.sum(ori_label_mask)
                                ancestor_error_subset_sum = np.sum(error_label_mask)

                                if parent == 0:
                                    normalized_Mi = np.nan
                                    class_feature_impt[NV[bestCutVar],j+1] += 0 
                                    
                                else:
                                    Ei = parent/2
                                    #Count of Left and Right nodes
                                    #Li = np.sum (true_subset[true_left] == j)
                                    #Ri = np.sum (true_subset[true_right] == j)
                                    Li = np.sum (y_left == unique_labels[x])
                                    Ri = np.sum (y_right == unique_labels[x])
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
                                        
                                        #class_feature_impt[NV[bestCutVar],j+1] += normalized_Mi*Pn * (ancestor_error_subset_sum/ancestor_class_sum)
                                        #new_row = [node_var[bestCutVar], j, normalized_Mi *Pn * ancestor_error_subset_sum/ancestor_class_sum]
                                        #rows.append(new_row)
                                        for idx, feature in enumerate(NV[bestCutVar]):
                                            class_feature_impt[feature,j+1] += normalized_Mi*Pn[idx] * (ancestor_error_subset_sum/ancestor_class_sum)
                                            new_row = [feature, j, normalized_Mi*Pn[idx] * (ancestor_error_subset_sum/ancestor_class_sum)]
                                            rows.append(new_row)   
                                        
                                    else: 
                                        normalized_Mi = 0
                                        #class_feature_impt[NV[bestCutVar],j+1] += normalized_Mi*Pn *  (ancestor_error_subset_sum/ancestor_class_sum)
                                        #new_row = [node_var[bestCutVar], j, np.nan]
                                        #rows.append(new_row)
                                        for idx, feature in enumerate(NV[bestCutVar]):
                                            class_feature_impt[feature,j+1] += normalized_Mi*Pn[idx] * (ancestor_error_subset_sum/ancestor_class_sum)
                                            new_row = [feature, j, normalized_Mi*Pn[idx] * (ancestor_error_subset_sum/ancestor_class_sum)]
                                            rows.append(new_row)  
                                        
                            ############################################################################################
                                                                # END #
                            #------------------------------------------------------------------------------------------#                                  

                    else:
                        if method.lower() in ['c', 'g']:
                            #leaf_label = np.argmax(np.bincount(Labels[currentDataIndx], minlength=max_label))
                            #nodelabel[current_node] = unique_labels[leaf_label]
                            negative_gradients_in_leaf = negative_gradients.take(TDindx[currentDataIndx], axis=0)
                            hessians_in_leaf = hessians.take(TDindx[currentDataIndx], axis=0)
                            #val = np.sum(negative_gradients_in_leaf) / np.sum(hessians_in_leaf)
                            # Check if the sum of hessians_in_leaf is zero
                            if np.sum(hessians_in_leaf) == 0:
                                val = 0  # Set a default value or handle division by zero appropriately
                            else:
                                val = np.sum(negative_gradients_in_leaf) / np.sum(hessians_in_leaf)
                            if current_node < len(nodelabel):
                                nodelabel[current_node] = val
                            else:
                                break
                            #nodelabel[current_node] = val
                        elif method.lower() == 'r':
                            #nodelabel[current_node] = np.mean(Labels[currentDataIndx])
                            #samples_in_this_leaf = currentDataIndx
                            #negative_gradients_in_leaf = negative_gradients.take(samples_in_this_leaf, axis=0)
                            #hessians_in_leaf = hessians.take(samples_in_this_leaf, axis=0)
                            negative_gradients_in_leaf = negative_gradients.take(Labels[TDindx], axis=0)
                            hessians_in_leaf = hessians.take(Labels[TDindx], axis=0)
                            # Check if the sum of hessians_in_leaf is zero
                            if np.sum(hessians_in_leaf) == 0:
                                val = 0  # Set a default value or handle division by zero appropriately
                            else:
                                val = np.sum(negative_gradients_in_leaf) / np.sum(hessians_in_leaf)
                            #val = np.sum(negative_gradients_in_leaf) / np.sum(hessians_in_leaf)
                            nodelabel[current_node] = val

                else:

                    if method.lower() in ['c', 'g']:
                        #leaf_label = np.argmax(np.bincount(Labels[currentDataIndx].astype(np.int64), minlength=max_label))
                        #nodelabel[current_node] = unique_labels[leaf_label]
                        negative_gradients_in_leaf = negative_gradients.take(TDindx[currentDataIndx], axis=0)
                        hessians_in_leaf = hessians.take(TDindx[currentDataIndx], axis=0)
                        #val = np.sum(negative_gradients_in_leaf) / np.sum(hessians_in_leaf)
                        # Check if the sum of hessians_in_leaf is zero
                        if np.sum(hessians_in_leaf) == 0:
                            val = 0  # Set a default value or handle division by zero appropriately
                        else:
                            val = np.sum(negative_gradients_in_leaf) / np.sum(hessians_in_leaf)
                        if current_node < len(nodelabel):
                            nodelabel[current_node] = val
                        else:
                            break
                        #nodelabel[current_node] = val
                    elif method.lower() == 'r':
                        #nodelabel[current_node] = np.mean(Labels[currentDataIndx])
                        negative_gradients_in_leaf = negative_gradients.take(TDindx[currentDataIndx], axis=0)
                        hessians_in_leaf = hessians.take(TDindx[currentDataIndx], axis=0)
                        
                        if np.sum(hessians_in_leaf) == 0:
                            val = 0  # Set a default value or handle division by zero appropriately
                        else:
                            val = np.sum(negative_gradients_in_leaf) / np.sum(hessians_in_leaf)
                        nodelabel[current_node] = val

            current_node+=1
            if current_node ==L:
                break
            random_state+=1
        feat_impo = giniii
        class_feature_impt_disbu = pd.DataFrame(rows, columns=['feature', 'class', 'v-value'])
        # Below removing the other classes group '0'
        class_feature_impt_disbu = class_feature_impt_disbu[class_feature_impt_disbu['class'] != 0]

        return nodeCutVar, nodeCutValue, childnode, nodelabel, feat_impo, class_feature_impt, class_feature_impt_disbu, RelatedCutVar, CenteredCutVar, VarianceCutVar, LoadingsCutVar


    def tree_predict(self, Data, cut_var, cut_val, nodechilds, nodelabel, RelatedCutVar, CenteredCutVar, VarianceCutVar, LoadingsCutVar):
        n, m = Data.shape
        tree_output = np.zeros(n)
        #prox=np.zeros(n)
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
            proximity_matrix[i] = current_node

        return tree_output, proximity_matrix
        

    def build_tree(self, i, random_state, Data, Labels, knn, nsamtosample, minparent, minleaf, method, nvartosample, weights, oobe, negative_gradients, hessians, true_label):
        print(f'building tree {i}')
        random_instance = np.random.RandomState(random_state)
        #TDindx = random_instance.choice(len(Labels), nsamtosample, replace=False)
        #print(TDindx)
        #nsamtosample = 2*(2 ** max_depth)
        
        # Get unique values in Labels
        unique_values = np.unique(Labels)

        # Initialize TDindx with one index for each unique value
        TDindx = np.concatenate([random_instance.choice(np.where(Labels == value)[0], 1) 
                                 for value in unique_values if np.any(Labels == value)])

        # If there are more samples to select, randomly select the rest without replacement
        if nsamtosample > len(unique_values):
            additional_indices = random_instance.choice(len(Labels), nsamtosample - len(unique_values), replace=False)
            TDindx = np.concatenate([TDindx, additional_indices])

        # Shuffle the indices to ensure randomness
        random_instance.shuffle(TDindx)

        Random_ForestT = self.tree_fit(Data[TDindx], Labels[TDindx], TDindx=TDindx, knn=knn, random_state=random_state, minparent=minparent, minleaf=minleaf, method=method, nvartosample=nvartosample, weights=weights, negative_gradients=negative_gradients, hessians=hessians, true_subset=true_label[TDindx])
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
            Random_ForestT_dict['CenteredCutVar'], Random_ForestT_dict['VarianceCutVar'], Random_ForestT_dict['LoadingsCutVar'])

            if method in ['c', 'g']:
                # oobe_val = np.mean(tree_output != Labels[NTD])
                oobe_val = np.sum(tree_output - Labels[NTD] == 0) / len(NTD)
            elif method == 'r':
                oobe_val = np.mean(np.square(tree_output - Labels[NTD]))

            Random_ForestT_dict['oobe'] = oobe_val

        return Random_ForestT_dict


    def fit_lavaset(self, Data, Labels, knn, random_state):
        #self.n_classes = pd.Series(Labels).nunique()

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

        '''Fit the GBM
            
            Parameters
            ----------
            X (data) : ndarray of size (number observations, number features)
                design matrix
                
            y (labels) : ndarray of size (number observations,)
                integer-encoded target labels in {0,1,...,k-1}
            '''
        
        self.n_classes = pd.Series(Labels).nunique()
        
        y_ohe = self._one_hot_encode_labels(Labels)

        
        raw_predictions = np.zeros(shape=y_ohe.shape)
        

        i = 0
        # Initial prediction 
        probabilities = self._softmax(raw_predictions)

        self.boosters = []
        for m in range(self.n_estimators):
            class_trees = []
            i+=1
            for k in range(self.n_classes):
                
                # Compute the pseudo residuals (negative gradients)
                negative_gradients = self._negative_gradients(y_ohe[:, k], probabilities[:, k])
                hessians = self._hessians(probabilities[:, k])
                #tree = DecisionTreeRegressorFromScratch(max_depth=self.max_depth)
                #tree = LAVASET(ntrees=1,n_neigh=0, distance=2, nvartosample=None, oobe=True)
                #knn = tree.knn_calculation(df.columns[1:])
                #lavaset = tree.fit_lavaset(X, Labels=negative_gradients, knn=knn, random_state=i, negative_gradients=negative_gradients, hessians=hessians)
                if self.ntrees==1:
                    lavaset = self.build_tree(i, random_state+i, Data, negative_gradients, knn, self.nsamtosample, 
                                        self.minparent, self.minleaf, self.method, self.nvartosample, 
                                        self.weights, self.oobe, negative_gradients=negative_gradients, hessians=hessians, true_label=y_ohe[:, k])
                else:
                    lavaset = Parallel(n_jobs=-1, verbose=10)(delayed(self.build_tree)
                                                            (i, random_state+i, Data, negative_gradients, knn, self.nsamtosample, 
                                                            self.minparent, self.minleaf, self.method, self.nvartosample, 
                                                            self.weights, self.oobe, negative_gradients=negative_gradients, hessians=hessians, true_label=y_ohe[:, k]) 
                                                            for i in range(self.ntrees))
                    """ gini0 = lavasets[0]['gini_gain_sum']
                    for i in range(self.ntrees):
                        gini = lavasets[i]['gini_gain_sum']
                        if gini0 < gini:
                            gini0 = gini
                            x = i
                        else:
                            x = 0
                    lavaset = lavasets[x] """
                
                #y_preds, votes, oobe = tree.predict_lavaset(Data, lavaset, self.ntrees)
                y_preds= self.predict(Data, lavaset, training=True, n_classes=self.n_classes)
                #print(tree.tree_fit)
                #tree = StochasticBosque(ntrees=1, nvartosample='sqrt', nsamtosample=95, oobe=True)
                #tree.fit(X, negative_gradients);
                #self._update_terminal_nodes(tree, X, negative_gradients, hessians)
                y_preds = pd.to_numeric(y_preds, errors='coerce')
                
                raw_predictions[:, k] += self.learning_rate * y_preds
                probabilities = self._softmax(raw_predictions)
                class_trees.append(lavaset)
                
            self.boosters.append(class_trees)

            """ Random_Forest = []
            Random_Forest = Parallel(n_jobs=-1, verbose=10)(delayed(self.build_tree)(i, random_state+i, Data, Labels, knn,
            self.nsamtosample, self.minparent, self.minleaf, self.method, self.nvartosample, self.weights, self.oobe, negative_gradients, hessians) for i in range(self.ntrees))
            """
        #return Random_Forest
        return self.boosters


    def predict_lavaset(self, Data, Random_Forest, ntrees, proximity):
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
        #f_votes = np.zeros((len(Random_Forest), Data.shape[0]), dtype=object)
        #oobe_values = np.zeros((len(Random_Forest),))

        n, m = Data.shape
        leaf_indices = np.zeros((n, ntrees))

        f_votes = np.zeros((ntrees, Data.shape[0]), dtype=object)
        oobe_values = np.zeros((ntrees,))
        #for i, tree in enumerate(Random_Forest):
        
        if ntrees == 1:
            x, leaf_indices[:, 0] = self.tree_predict(Data, Random_Forest['tree_cut_var'], Random_Forest['tree_cut_val'],
                                                          Random_Forest['tree_nodechilds'], Random_Forest['tree_nodelabel'], 
                                                          Random_Forest['RelatedCutVar'], Random_Forest['CenteredCutVar'], 
                                                          Random_Forest['VarianceCutVar'], Random_Forest['LoadingsCutVar'])
            f_votes[0,:] = x.ravel()
            """ f_votes[0, :] =  self.tree_predict(Data, Random_Forest['tree_cut_var'], Random_Forest['tree_cut_val'],Random_Forest['tree_nodechilds'], 
            Random_Forest['tree_nodelabel'], Random_Forest['RelatedCutVar'], 
            Random_Forest['CenteredCutVar'], Random_Forest['VarianceCutVar'], Random_Forest['LoadingsCutVar']).ravel()
             """
            oobe_values[0] = Random_Forest['oobe']
            method = Random_Forest['method']
        else:
            i = 0
            while i != ntrees:
                x, leaf_indices[:, i] = self.tree_predict(Data, Random_Forest[i]['tree_cut_var'], Random_Forest[i]['tree_cut_val'],
                                                          Random_Forest[i]['tree_nodechilds'], Random_Forest[i]['tree_nodelabel'], 
                                                          Random_Forest[i]['RelatedCutVar'], Random_Forest[i]['CenteredCutVar'], 
                                                          Random_Forest[i]['VarianceCutVar'], Random_Forest[i]['LoadingsCutVar'])
                f_votes[i,:] = x.ravel()
                """ f_votes[i, :] =  self.tree_predict(Data, Random_Forest[i]['tree_cut_var'], Random_Forest[i]['tree_cut_val'],Random_Forest[i]['tree_nodechilds'], 
                Random_Forest[i]['tree_nodelabel'], Random_Forest[i]['RelatedCutVar'], 
                Random_Forest[i]['CenteredCutVar'], Random_Forest[i]['VarianceCutVar'], Random_Forest[i]['LoadingsCutVar']).ravel()
                 """
                oobe_values[i] = Random_Forest[i]['oobe']
                i+=1
            method = Random_Forest[0]['method'] 


        if method in ['c', 'g']:
            unique_labels, indices = np.unique(f_votes, return_inverse=True)
            #f_votes = indices.reshape((len(Random_Forest), Data.shape[0]))
            f_votes = indices.reshape((ntrees, Data.shape[0]))
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

        if proximity==False:
            return f_output, f_votes, oobe_mean
        else:
            return f_output, f_votes, oobe_mean, leaf_indices

    
    def feature_evaluation (self, Data, Random_Forest):
        all_importances = np.zeros((Data.shape[1], 3))
        for i, tree in enumerate(Random_Forest):
            importance_per_tree = np.array(Random_Forest[i]['feature_importances'])
            all_importances += importance_per_tree
        return all_importances
    
    #Extract Summed V-values: 
    def class_feature_evaluation (self, Data, Random_Forest, nclasses):
        all_class_features = np.zeros(((Data.shape[1], nclasses)))

        if (len(Random_Forest[0][0])==1):
                n=0
                while (n != len(Random_Forest)):
                    for k in range(len(Random_Forest[0])):
                        class_impt_per_tree = np.array(Random_Forest[n][k]['class_importance_uncomb_val'][:,2])
                        all_class_features[:,k] += class_impt_per_tree
                    n+=1
        else:
            if(len(Random_Forest[0][0])==13):
                n=0
                while (n != len(Random_Forest)):
                    for k in range(len(Random_Forest[0])):
                        class_impt_per_tree = np.array(Random_Forest[n][k]['class_importance_uncomb_val'][:,2])
                        all_class_features[:,k] += class_impt_per_tree
                    n+=1
            else:
                n=0
                while (n != len(Random_Forest)):
                    for k in range(len(Random_Forest[0])):
                        for x in range(len(Random_Forest[0][0])):
                            class_impt_per_tree = np.array(Random_Forest[n][k][x]['class_importance_uncomb_val'][:,2])
                            all_class_features[:,k] += class_impt_per_tree
                            
                    n+=1

        return all_class_features
    
    #Extract V-values distribution: 
    def class_feature_distribution_evaluation (self, Data, Random_Forest, nclasses):
        all_class_features = pd.DataFrame(columns=['feature', 'class', 'v-value'])

        if (len(Random_Forest[0][0])==1):
                n=0
                while (n != len(Random_Forest)):
                    for k in range(len(Random_Forest[0])):
                        class_impt_per_tree = pd.DataFrame(Random_Forest[n][k]['class_feature_impt_disbu'][['feature', 'v-value']])
                        class_impt = pd.DataFrame(class_impt_per_tree, columns=['feature', 'v-value'])
                        class_impt['class'] = k
                        all_class_features = pd.concat([all_class_features, class_impt], ignore_index=True)
                    n+=1
        else:
            if(len(Random_Forest[0][0])==13):
                n=0
                while (n != len(Random_Forest)):
                    for k in range(len(Random_Forest[0])):
                        class_impt_per_tree = pd.DataFrame(Random_Forest[n][k]['class_feature_impt_disbu'][['feature', 'v-value']])
                        class_impt = pd.DataFrame(class_impt_per_tree, columns=['feature', 'v-value'])
                        class_impt['class'] = k
                        all_class_features = pd.concat([all_class_features, class_impt], ignore_index=True)
                    n+=1
            else:
                n=0
                while (n != len(Random_Forest)):
                    for k in range(len(Random_Forest[0])):
                        for x in range(len(Random_Forest[0][0])):
                            class_impt_per_tree = pd.DataFrame(Random_Forest[n][k][x]['class_feature_impt_disbu'][['feature', 'v-value']])
                            class_impt = pd.DataFrame(class_impt_per_tree, columns=['feature', 'v-value'])
                            class_impt['class'] = k
                            all_class_features = pd.concat([all_class_features, class_impt], ignore_index=True)
                            
                    n+=1

        return all_class_features

    def fit(self, X, y, knn):
        '''Fit the GBM
        
        Parameters
        ----------
        X : ndarray of size (number observations, number features)
            design matrix
            
        y : ndarray of size (number observations,)
            integer-encoded target labels in {0,1,...,k-1}
        '''
        
        self.n_classes = pd.Series(y).nunique()
        
        y_ohe = self._one_hot_encode_labels(y)

        
        raw_predictions = np.zeros(shape=y_ohe.shape)
        

        i = 0
        # Initial prediction 
        probabilities = self._softmax(raw_predictions)

        self.boosters = []
        for m in range(self.n_estimators):
            class_trees = []
            for k in range(self.n_classes):
                i+=1
                # Compute the pseudo residuals (negative gradients)
                negative_gradients = self._negative_gradients(y_ohe[:, k], probabilities[:, k])
                hessians = self._hessians(probabilities[:, k])
                #tree = DecisionTreeRegressorFromScratch(max_depth=self.max_depth)
                tree = LAVABOOST(ntrees=1,n_neigh=10, nvartosample=None, nsamtosample=None, oobe=True)
                #knn = tree.knn_calculation(df.columns[1:])
                lavaset = tree.fit_lavaset(X, Labels=negative_gradients, knn=knn, random_state=i, negative_gradients=negative_gradients, hessians=hessians)

                #print(type(lavaset))
                y_preds, votes, oobe = tree.predict_lavaset(X, lavaset)
                #print(tree.tree_fit)
                #tree = StochasticBosque(ntrees=1, nvartosample='sqrt', nsamtosample=95, oobe=True)
                #tree.fit(X, negative_gradients);
                #self._update_terminal_nodes(tree, X, negative_gradients, hessians)
                y_preds = pd.to_numeric(y_preds, errors='coerce')
                
                raw_predictions[:, k] += self.learning_rate * y_preds
                probabilities = self._softmax(raw_predictions)
                class_trees.append(lavaset)
                
            self.boosters.append(class_trees)
    
    def _one_hot_encode_labels(self, y):
        if isinstance(y, pd.Series): y = y.values
        ohe = OneHotEncoder()
        y_ohe = ohe.fit_transform(y.reshape(-1, 1)).toarray()
        return y_ohe
        
    def _negative_gradients(self, y_ohe, probabilities):
        # Calculating residuals
        # y_ohe (Observed prediction, which remains constant for each class) - probabilities (Predicted prediction) = Residual
        return y_ohe - probabilities
        #return np.where(y_ohe - probabilities == 0, 0, y_ohe - probabilities)
    
    def _hessians(self, probabilities): 
        # Used leaf assignment transformation of raw predictions log(odds)
        return probabilities * (1 - probabilities)

    def _softmax(self, raw_predictions):
        # Used to convert raw predictions into a probability
        # Done after leaf assignments before calculating the new residuals
        # Or for predicting labels after running through the tree and calculating log(odds) prediction
        # Turning the log(odds) into a probability
        
        max_value = np.max(raw_predictions, axis=1, keepdims=True)
        numerator = np.exp(raw_predictions - max_value)
        denominator = np.sum(numerator, axis=1, keepdims=True)
        
        return numerator / denominator
        
    def _update_terminal_nodes(self, tree, X, negative_gradients, hessians):
        '''Update the terminal node predicted values'''
        # terminal node id's
        #leaf_nodes = np.nonzero(tree.tree_.children_left == -1)[0]
        leaf_nodes = np.nonzero(tree.tree_.get('left', {}).get('feature_idx', None) is None)[0]
        # compute leaf for each sample in ``X``.
        #leaf_node_for_each_sample = tree.apply(X)
        leaf_assignments = tree.get_leaf_assignments(X)
        for leaf in leaf_nodes:
            samples_in_this_leaf = np.where(leaf_assignments == leaf)[0]
            negative_gradients_in_leaf = negative_gradients.take(samples_in_this_leaf, axis=0)
            hessians_in_leaf = hessians.take(samples_in_this_leaf, axis=0)
            val = np.sum(negative_gradients_in_leaf) / np.sum(hessians_in_leaf)
            tree.tree_['left']['feature_idx'] = val
            #tree.tree_.value[leaf, 0, 0] = val
                    # Check for NaN values
            #if np.any(np.isnan(negative_gradients_in_leaf)) or np.any(np.isnan(hessians_in_leaf)):
                #val = 0  # Set a default value or handle NaN appropriately
            #else:
                # Ensure denominator is not zero
                #sum_hessians = np.sum(hessians_in_leaf)
                #val = np.sum(negative_gradients_in_leaf) / sum_hessians if sum_hessians != 0 else 0
        
            #tree.tree_.value[leaf, 0, 0] = val
          
    def predict_proba(self, X, model, training, nclasses):
        '''Generate probability predictions for the given input data.'''
        #tree = LAVASET(ntrees=1,n_neigh=0, distance=2, nvartosample=None, oobe=True)
        #raw_predictions =  np.full((X.shape[0], self.n_classes), 0.5)
        #raw_predictions =  np.zeros(shape=(X.shape[0], self.n_classes))
        if training==True:
            if len(model)==13:
                raw_predictions =  np.zeros(shape=(X.shape[0], 1))
                y_preds, votes, oobe = self.predict_lavaset(X, model, ntrees=1, proximity=False)
                y_preds = pd.to_numeric(y_preds, errors='coerce')
                raw_predictions[:, 0] +=self.learning_rate * y_preds            
            else:   
                raw_predictions =  np.zeros(shape=(X.shape[0], 1))         
                y_preds, votes, oobe = self.predict_lavaset(X, model, ntrees=len(model), proximity=False)
                y_preds = pd.to_numeric(y_preds, errors='coerce')
                raw_predictions[:, 0] +=self.learning_rate * y_preds
        else:
            if (len(model[0][0])==1):
                raw_predictions =  np.zeros(shape=(X.shape[0], len(model[0])))
                proximity_matrix = pd.DataFrame()
                n=0
                while (n != len(model)):
                    class_indices = pd.DataFrame()
                    for k in range(len(model[0])):
                        y_preds, votes, oobe, leaf_indices = self.predict_lavaset(X, model[n][k], ntrees=1, proximity=True)
                        # Getting predictions
                        y_preds = pd.to_numeric(y_preds, errors='coerce')
                        raw_predictions[:, k] +=self.learning_rate * y_preds
                        # Getting leaf indices for proximity matrix
                        leaf_indices = pd.DataFrame(leaf_indices)
                        class_indices = pd.concat([class_indices, leaf_indices], axis=1)
                    n+=1
                    proximity_matrix = pd.concat([proximity_matrix, class_indices], axis=1)
            else:
                if(len(model[0][0])==13):
                    raw_predictions =  np.zeros(shape=(X.shape[0], len(model[0])))
                    proximity_matrix = pd.DataFrame()
                    n=0
                    while (n != len(model)):
                        class_indices = pd.DataFrame()
                        for k in range(len(model[0])):
                            y_preds, votes, oobe, leaf_indices = self.predict_lavaset(X, model[n][k], ntrees=1, proximity=True)
                            # Getting predictions
                            y_preds = pd.to_numeric(y_preds, errors='coerce')
                            raw_predictions[:, k] +=self.learning_rate * y_preds
                            # Getting leaf indices for proximity matrix
                            leaf_indices = pd.DataFrame(leaf_indices)
                            class_indices = pd.concat([class_indices, leaf_indices], axis=1)
                        n+=1
                        proximity_matrix = pd.concat([proximity_matrix, class_indices], axis=1)
                else:
                    raw_predictions =  np.zeros(shape=(X.shape[0], len(model[0])))
                    proximity_matrix = pd.DataFrame()
                    n=0
                    while (n != len(model)):
                        class_indices = pd.DataFrame()
                        for k in range(len(model[0])):
                            y_preds, votes, oobe, leaf_indices = self.predict_lavaset(X, model[n][k], ntrees=len(model[0][0]), proximity=True)
                            # Getting predictions
                            y_preds = pd.to_numeric(y_preds, errors='coerce')
                            raw_predictions[:, k] +=self.learning_rate * y_preds
                            # Getting leaf indices for proximity matrix
                            leaf_indices = pd.DataFrame(leaf_indices)
                            class_indices = pd.concat([class_indices, leaf_indices], axis=1)
                        n+=1
                        proximity_matrix = pd.concat([proximity_matrix, class_indices], axis=1)
            
        # Converting the log(odds) prediction to a probability of it being in each class
        probabilities = self._softmax(raw_predictions)
        if training == True:
            return probabilities
        else:
            return probabilities, proximity_matrix
        
    def predict(self, X, model, training, n_classes):
        '''Generate predicted labels (as 1-d array)'''
        if training == True:
            probabilities = self.predict_proba(X, model, training, n_classes)
            return np.argmax(probabilities, axis=1)
        else:
            probabilities, proximity_matrix = self.predict_proba(X, model, training, n_classes)
            return np.argmax(probabilities, axis=1), proximity_matrix
        
    def assign_hexagonal_positions(self, G, pos, num_cols, num_rows):
    
        # Calculate the center of the graph
        center = np.mean(list(pos.values()), axis=0)

        # Calculate the Euclidean distance from each node to the center
        distances = {node: np.linalg.norm(pos[node] - center) for node in G.nodes}
        
        # Sort nodes by distance from the center in descending order
        sorted_nodes = sorted(distances, key=distances.get, reverse=True)
        

        # Assign positions on the hexagonal grid to the nodes
        final_pos = {}
        used_positions = set()

        # Map positions to a hexagonal grid
        m = int(4*2.5)
        n = int(5*2.5)
        m = num_rows
        n = num_cols
        grid_pos = self.hexagonal_grid_positions(m, n, scale=2) # change this to a suitable size
        
        # Assign positions on the hexagonal grid to the nodes
        final_pos = {}
        used_positions = set()
        
        for node in sorted_nodes:
            node_pos = pos[node]
            best_grid_pos = None
            min_distance = float('inf')
            
            for grid_node, grid_node_pos in grid_pos.items():
                distance = np.linalg.norm(node_pos - grid_node_pos)
                
                # Check if the position is already occupied by another node
                if grid_node_pos in used_positions:
                    continue
                
                # Check if the position will cause an edge to cross the node exactly
                edge_crosses_node = False
                for neighbor in G.neighbors(node):
                    neighbor_pos = final_pos.get(neighbor)
                    if neighbor_pos is not None:
                        edge = (node_pos, neighbor_pos)
                        if np.all(np.isclose(np.cross(neighbor_pos - node_pos, grid_node_pos - node_pos), 0)):
                            edge_crosses_node = True
                            break
                
                if distance < min_distance and not edge_crosses_node:
                    min_distance = distance
                    best_grid_pos = grid_node_pos
            
            final_pos[node] = best_grid_pos
            used_positions.add(best_grid_pos)
        
        # Convert hexagonal grid positions to a dictionary format compatible with NetworkX
        #final_pos = {node: grid_pos for node, grid_pos in zip(sorted_nodes, grid_pos.values())}

        return final_pos
    
    def hexagonal_grid_positions(self, m, n, scale):
        pos = {}
        max_x = 0
        max_y = 0
        for i in range(m):
            for j in range(n):
                x = j * 1.5 if i % 2 == 0 else j * 1.5 + 3  # Offset every other column
                y = i * (2 * 0.866) + (j % 2) * 0.866 
                #y = i * 2 + 1 if j % 2 == 1 else i * 2  # Offset every other column
                pos[(i * n) + j + 1] = (x, y)
        return pos  
    
    def draw_graph_with_curved_edges(self, G, pos, node_colors, top_edges, high_nodes, low_nodes, final_pos, name, colormap, norm, vmin, vmax, n):
        plt.figure(figsize=(14.5, 6.5))
        #nx.draw_networkx(G, pos, with_labels=False, node_size=400, node_color=node_colors, alpha=0.7, ax=ax)
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=400, node_color=node_colors)
        center = np.mean(list(pos.values()), axis=0)
        
        # Find adjacent straight edges
        edge_dist = 1.5 # Adjust the threshold value (1.5) to control the proximity of nodes
        straight_edges = [(u, v) for u, v in top_edges if np.linalg.norm(np.array(pos[v]) - np.array(pos[u])) <= edge_dist]
        
        # Find non-crossing straight edges
        node_dist = 0.1
        remaining_edges = [(u, v) for u, v in top_edges if np.linalg.norm(np.array(pos[v]) - np.array(pos[u])) > edge_dist]
        for u, v in remaining_edges:
            straight_edge = True
            for node in G.nodes:
                if node != u and node != v:
                    # checks for line crossing and accounts for the fact that lines continue infinitely, but edges do not
                    if (np.linalg.norm(np.cross(np.array(pos[v]) - np.array(pos[u]), np.array(pos[u]) - np.array(pos[node]))) <= node_dist) and np.isclose(np.linalg.norm(np.array(pos[v]) - np.array(pos[u])), (np.linalg.norm(np.array(pos[v]) - np.array(pos[node])) + np.linalg.norm(np.array(pos[node]) - np.array(pos[u]))), rtol=1e-4, atol=1e-4):
                        straight_edge = False
                        break
            if straight_edge:
                straight_edges.append((u, v))
        # Draw straight edges
        nx.draw_networkx_edges(G, pos, edgelist=straight_edges, width=1.0, edge_color='lightgray')
        
        # Draw curved edges
        curvature = 0.2 # Adjust the fraction to control the amount of curvature
        curved_edges = [(u, v) for u, v in top_edges if np.allclose(np.cross(np.array(pos[v]) - np.array(pos[u]), np.array(pos[u]) - np.array(pos[v])), 0) and (u, v) not in straight_edges]
        for edge in curved_edges:
            u, v = edge
            mean_point = (np.array(pos[u]) + np.array(pos[v])) / 2
            edge_vector = np.array(pos[v]) - np.array(pos[u])
            perpendicular_vector = np.array([-edge_vector[1], edge_vector[0]])
            normalised_perpendicular_vector = perpendicular_vector / np.linalg.norm(perpendicular_vector)
            opt1 = mean_point + edge_dist * (1/3) * normalised_perpendicular_vector
            opt2 = mean_point + edge_dist * (-1/3) * normalised_perpendicular_vector
            distance_opt1 = np.linalg.norm(np.array(opt1) - np.array(center))
            distance_opt2 = np.linalg.norm(np.array(opt2) - np.array(center))
            if distance_opt1 > distance_opt2:
                control_pos = opt1
            else:
                control_pos = opt2
            points = np.vstack([pos[u], control_pos, pos[v]])
            cubic_bezier = mpl_path.Path(points, [mpl_path.Path.MOVETO, mpl_path.Path.CURVE3, mpl_path.Path.CURVE3])
            curve = mpl_collections.PathCollection([cubic_bezier], edgecolor='lightgray', facecolor='none', linewidth=1.0)
            plt.gca().add_collection(curve)

        # Add labels with custom colors
        labels = nx.draw_networkx_labels(G, pos, font_size=15)

        # Adjust font color for top nodes and their neighbors
        for node in labels:
            if (node in high_nodes)==True:
                labels[node].set_color('red')
                labels[node].set_weight('bold')
                #x, y = grid_positions[node]
                x, y = pos[node]
                labels[node].set_position((x, y + 0.78))
            elif (node in low_nodes)==True:
                labels[node].set_color('blue')
                labels[node].set_weight('bold')
                #x, y = grid_positions[node]
                x, y = pos[node]
                labels[node].set_position((x, y + 0.78))
            else:
                #labels[node].set_color('darkblue')
                #labels[node].set_weight('normal')
                labels[node].set_text('')

        # Adjust labels at the edges
        x_coords = [pos[node][0] for node in G.nodes]
        y_coords = [pos[node][1] for node in G.nodes]
        x_min, x_max = min(x_coords), max(x_coords)
        y_max = max(y_coords)
        for node, label in labels.items():
            x, y = pos[node]
            if np.isclose(x, x_min):
                label.set_position((x + 0.3, y + 0.78))
            elif np.isclose(x, x_max):
                label.set_position((x - 0.4, y + 0.78))
            elif np.isclose(y, y_max):
                label.set_position((x, y + 0.52))

        # Adjust labels in the second row from the top
        sorted_y_coords = sorted(set(y_coords), reverse=True)
        if len(sorted_y_coords) > 1:
            second_y_max = sorted_y_coords[1]
            for node, label in labels.items():
                x, y = pos[node]
                if np.isclose(y, second_y_max):
                    label.set_position((x, y + 0.68))

        # Add a title
        plt.title("LAVABOOST Feature Importance Assignment in " + name, fontsize=18)

        # Add a colorbar on the right axis
        cax = plt.axes([0.92, 0.1, 0.02, 0.8])
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])

        #cbar = plt.colorbar(sm, cax=cax, orientation='vertical', ticks=[vmin, vmax])
        #cbar.set_ticklabels([round(vmin, 1), round(vmax, 1)], fontsize=15)
        cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
        #cbar.set_ticklabels([-1, 1], fontsize=15)
        cbar.set_ticks([-1, 1])
        cbar.ax.set_yticklabels(['-1', '1'], fontsize=15)
        cbar.set_label('Normalised Aggregated CLIFI value', rotation=270, labelpad=18, fontsize=15)
        
        # Adjust figure margins to create more space
        plt.subplots_adjust(left=0.02, right=0.91, top=0.95, bottom=0.05)

        # Creating filename to save files
        first_word = name.split()[0]
        folder_path = "C:\\Users\\Eloisa\\Documents\\ICL\\3_Project_1\\Results\\Pub_results\\Final\\NetworkX\\Aggregated\\LAVABOOST"
        filename = f"{n}-{first_word}.jpg"
        filepath = os.path.join(folder_path, filename)
        
        # Save plot with DPI=350
        plt.savefig(filepath, dpi=350)

        # Show the graph
        #plt.show()

    def scale_dataframe_to_minus_one_to_one(self, df):
        # Find the absolute maximum value in the entire DataFrame
        max_val = np.max(np.abs(df.values))
        
        # Avoid division by zero
        if max_val == 0:
            return df
        
        # Scale the entire DataFrame to the range -1 to 1
        scaled_df = df / max_val
        
        return scaled_df, max_val
    
    def create_v_value_df(self, df):
        # Ensure features and classes are sorted
        sorted_features = sorted(df['feature'].unique())
        sorted_classes = sorted(df['class'].unique())

        # Create an empty DataFrame with features as rows and classes as columns
        v_value_df = pd.DataFrame(index=sorted_features, columns=sorted_classes, dtype=float)

        # Iterate over each feature and class to calculate the v-value
        for feature in sorted_features:
            for cls in sorted_classes:
                # Filter the original DataFrame for the current feature and class
                filtered_rows = df[(df['feature'] == feature) & (df['class'] == cls)]
                
                if not filtered_rows.empty:
                    # Calculate the summed v-value
                    mean_v_value = filtered_rows['v-value'].sum() 
                    v_value_df.at[feature, cls] = mean_v_value
                else:
                    v_value_df.at[feature, cls] = 0
        
        return v_value_df
