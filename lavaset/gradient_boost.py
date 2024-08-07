from best_cut_node import best_cut_node
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import csv
from joblib import Parallel, delayed
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import matplotlib.collections as mpl_collections
import matplotlib.path as mpl_path
import networkx as nx
import os


class StochasticBosque:   
    def __init__(self, minparent=2, minleaf=1, max_depth=None, nvartosample=None, ntrees=1, nsamtosample=None, method=None, oobe=False, weights=None, n_estimators=None, learning_rate=None, boosters=[], n_classes=None):
        self.minparent = minparent
        self.minleaf = minleaf
        self.nvartosample = nvartosample
        self.ntrees = ntrees
        self.nsamtosample = nsamtosample
        self.method = method
        self.oobe = oobe
        self.weights = weights
        self.max_depth = max_depth
        # self.Random_Forest = None
        self.n_estimators=n_estimators; 
        self.learning_rate=learning_rate
        self.boosters=boosters
        self.n_classes=n_classes


    def tree_fit(self, Data, Labels, max_depth, TDindx, random_state, minparent, minleaf, nvartosample, method, weights, negative_gradients, hessians, true_subset):
        
        # Find the positions where true_labels are 1
        positions = true_subset == 1

        # Find the labels that are non-zero at those positions
        non_zero_at_positions = (Labels != 0) & positions

        # Count the number of such labels
        count = np.sum(non_zero_at_positions)

        # n is the number of patients in the training set
        n = len(Labels)
        # L is the total number of leaves required to build a binary tree structure
        # minleaf is the minimum number of samples to form a leaf node in the decision tree
        L = int(2 * np.ceil(n / minleaf) - 1)
        
        # m is the total number of features in the dataset
        m = Data.shape[1]

        # nodeDataIndx will contain the tree structure as a dictionary
        # so the first entry is the root node (0) associated to consecutive integers (# of patients in training set)
        nodeDataIndx = {0: np.arange(n)}

        nodeCutVar = np.zeros(int(L))
        nodeCutValue = np.zeros(int(L))

        # nodeflags keeps track of whether a node has been visted during the tree building process
        # nodeflags has the total number of nodes required to build a binary tree stucture (the +1 is for the root node)
        # At first it will only contain zeros indicating that the node hasn't been visted
        nodeflags = np.zeros(int(L))
        nodelabel = np.full(int(L), np.inf)
        nodelabel = np.zeros(int(L))

        childnode = np.zeros(int(L))
        # giniii starts with an all zeros data set with the shape (m,3)
        giniii = np.zeros((m,3))

        # Need to make the first value 1 to allow passing through the while loop 
        nodeflags[0] = 1

        if method.lower() in ['c', 'g']:
            unique_labels = np.unique(Labels)
            max_label = len(unique_labels)
        else:
            max_label = None

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


        # current_node is used as the index for iterating through nodeflags
        current_node = 0
        # random_state=random_state

        while nodeflags[current_node] == 1:
            # free_node is the index of the first unvisited node (i.e. first occurence of 0),
            # so it is used as an index for the next tree building (in nodeDataIndx)
            #free_node = np.where(nodeflags == 0)[0][0]
            # Check if there are unvisited nodes
            if np.any(nodeflags == 0):
                # free_node is the index of the first unvisited node (i.e., first occurrence of 0)
                free_node = np.where(nodeflags == 0)[0][0]
            else:
                free_node = None

            # currentDataIndx contains the patient row numbers in the current leaf/node as nodeDataIndx is the tree
            # below the code will add nodes to the nodeDataIndx adhead of the current node location
            # when that happens nodeflags[current_node]==1 then you check if the node is pure or not for assigning node label
            currentDataIndx = nodeDataIndx[current_node]# samples in this node 

            # There are seperate free_node and current_node variables because below you need two indexes
            # free_node: finds the next location in nodeDataIndx for adding a node
            # current_node: the next location in nodeDataIndx for checking if pure 
            # (thus free_node will at times be ahead of current node as two nodes can be created in one while loop
            # so two 1 values are assigned to node flags but current node is always equal to the number of while loops)

            if len(np.unique(Labels[currentDataIndx])) == 1:
                # If statement condition: all labels are the same at the current node
                # So code below should make this a leaf and continue
                if method.lower() in ['c', 'g']:
                    #nodelabel[current_node] = unique_labels[Labels[currentDataIndx[0]].astype(np.int64)]
                    negative_gradients_in_leaf = negative_gradients.take(TDindx[currentDataIndx], axis=0)
                    hessians_in_leaf = hessians.take(TDindx[currentDataIndx], axis=0)
                    
                    if np.sum(hessians_in_leaf) == 0:
                        val = 0  # Set a default value or handle division by zero appropriately
                    else:
                        val = np.sum(negative_gradients_in_leaf) / np.sum(hessians_in_leaf)
                    """ if current_node < len(nodelabel):
                        nodelabel[current_node] = val
                    else:
                        break """
                    nodelabel[current_node] = val

                elif method.lower() == 'r':
                    #nodelabel[current_node] = Labels[currentDataIndx[0]]
                    negative_gradients_in_leaf = negative_gradients.take(TDindx[currentDataIndx], axis=0)
                    hessians_in_leaf = hessians.take(TDindx[currentDataIndx], axis=0)
                    
                    if np.sum(hessians_in_leaf) == 0:
                        val = 0  # Set a default value or handle division by zero appropriately
                    else:
                        val = np.sum(negative_gradients_in_leaf) / np.sum(hessians_in_leaf)
                    nodelabel[current_node] = val
                nodeCutVar[current_node] = 0
                nodeCutValue[current_node] = 0
            else:
                # Else statement condition: labels are different at the current node
                # So code below should continue splitting  
                if len(currentDataIndx) >= minparent and free_node != None:
                    # If statement condition: there are enough patients in leaf node to allow further splitting
                    random_instance = np.random.RandomState(random_state)
                    node_var = random_instance.permutation(range(0,m))[:nvartosample]

                    # node_var contains random column numbers of the data (i.e. features), the amount is square root of number of features
                    #node_var = np.random.permutation(range(0,m))[:nvartosample]

                    # at row numbers determined by node var (so finds the corresponding row in giniii to the feature column in node_var)
                    # 1 is added to the first column
                    # so the first column in ginii is used to record the frequency at which specific features are being selected
                    giniii[node_var,0]+=1

                    if weights is not None:
                        Wcd = weights[currentDataIndx]
                    else:
                        Wcd = None

                    bestCutVar, bestCutValue = best_cut_node(method, Data[currentDataIndx][:, node_var], Labels[currentDataIndx], minleaf, max_label)
                    random_state+=1
                    if bestCutVar != -1:
                        #-----------------------------#
                        # Feature importance variable assignment
                        gini_temp_array = np.zeros((1,2))
                        gini_temp_array[0,0] = node_var[bestCutVar]
                        #-----------------------------#

                        # for xgboost I will have to add a condition based on what the gain is to allow for splitting
                        # bestCutVar will equal -1 when method is equal to 'c' (no better cut was found during the iteration of the features)
                        # so there will be no splitting

                        # below is adding the bestCutVar (the best feature chosen for splitting) and bestCutValue 
                        nodeCutVar[current_node] = node_var[bestCutVar]
                        nodeCutValue[current_node] = bestCutValue
                        # is adding a one to value in the column next to the best feature chosen (recording frequencing feature is chosen)
                        giniii[node_var[bestCutVar],1]+= 1

                        # Many tree based algotithms, including decision trees, use even indices for left child node
                        # and odd indices for right child node.
                        # Below updates the left node with data points that have <=bestCut Value
                        nodeDataIndx[free_node] = currentDataIndx[Data[currentDataIndx, node_var[bestCutVar]] <= bestCutValue]
                        # Below updates the right node with data points that have >bestCut Value
                        nodeDataIndx[free_node+1] = currentDataIndx[Data[currentDataIndx, node_var[bestCutVar]] > bestCutValue]
                        # Thus nodeDataIndx ends up as a dictionary with the first node '0' having all data points,
                        # then the child nodes are added as a dictionary
                        # This dictionary is represents the tree


                        y_left=Labels[nodeDataIndx[free_node]]
                        y_right=Labels[nodeDataIndx[free_node+1]]
                        y_parent = list(y_left)+list(y_right)

                        true_left = nodeDataIndx[free_node]
                        true_right = nodeDataIndx[free_node+1]
                        true_parent = list(true_left)+list(true_right)
                        proportion_left = len(true_left) / len(true_parent)
                        proportion_right = len(true_right) / len(true_parent)

                        #proportion_left = len(y_left) / len(y_parent)
                        #proportion_right = len(y_right) / len(y_parent)
                        
                        p_parent = (np.bincount(np.array(true_parent, dtype=np.int64)))/len(true_parent)
                        p_left = (np.bincount(np.array(true_left, dtype=np.int64)))/len(true_left)
                        p_right = (np.bincount(np.array(true_right, dtype=np.int64)))/len(true_right)

                        #p_parent = (np.bincount(np.array(y_parent, dtype=np.int64)))/len(y_parent)
                        #p_left = (np.bincount(np.array(y_left, dtype=np.int64)))/len(y_left)
                        #p_right = (np.bincount(np.array(y_right, dtype=np.int64)))/len(y_right)
                        
                        gini_l = 1-np.sum(p_left**2)
                        gini_r = 1-np.sum(p_right**2)
                        gini_p = 1-np.sum(p_parent**2)
                        
                        gini_gain = gini_p - (proportion_left*gini_l + proportion_right*gini_r)
                        giniii[node_var[bestCutVar], 2] += gini_gain

                        new_row = [node_var[bestCutVar], gini_gain]
                        rows_gini.append(new_row)

                        # below nodeflags is updated to add 1 to both the left and right nodes (i.e. visited)
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
                            class_row = []
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
                                    class_feature_impt[node_var[bestCutVar],j+1] += 0
                                    
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
                                        
                                        class_feature_impt[node_var[bestCutVar],j+1] += normalized_Mi * ancestor_error_subset_sum/ancestor_class_sum
                                        new_row = [node_var[bestCutVar], j, normalized_Mi * ancestor_error_subset_sum/ancestor_class_sum]
                                        class_row.append(new_row)
                                        
                                    else: 
                                        normalized_Mi = 0
                                        class_feature_impt[node_var[bestCutVar],j+1] += normalized_Mi * ancestor_error_subset_sum/ancestor_class_sum
                                        new_row = [node_var[bestCutVar], j, normalized_Mi * ancestor_error_subset_sum/ancestor_class_sum]
                                        class_row.append(new_row)

                            sum_last_values = sum(row[2] for row in class_row)
                            # Create the final row
                            first_value = class_row[0][0]
                            second_value = class_row[0][1]
                            final_row = [first_value, second_value, sum_last_values]
                            rows.append(final_row)
                                        
                    ############################################################################################
                                                        # END #
                    #------------------------------------------------------------------------------------------#                                  

                    else:
                        # No good cut was found because there may only be a couple different classes and the total number of patients is greater than minparent
                        # Or method = 'c'
                        if method.lower() in ['c', 'g']:
                            #leaf_label = np.argmax(np.bincount(Labels[currentDataIndx], minlength=max_label))
                            #nodelabel[current_node] = unique_labels[leaf_label]
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
                            nodelabel[current_node] = val
                        elif method.lower() == 'r':
                            #nodelabel[current_node] = np.mean(Labels[currentDataIndx])
                            negative_gradients_in_leaf = negative_gradients.take(TDindx[currentDataIndx], axis=0)
                            hessians_in_leaf = hessians.take(TDindx[currentDataIndx], axis=0)
                            
                            if np.sum(hessians_in_leaf) == 0:
                                val = 0  # Set a default value or handle division by zero appropriately
                            else:
                                val = np.sum(negative_gradients_in_leaf) / np.sum(hessians_in_leaf)
                            nodelabel[current_node] = val

                else:
                    # Else condition: there are not enough patients for splitting and the leaf node is not pure
                    # So must associate an classification value of the leaf based on the majority classification (i.e. label)
                    if method.lower() in ['c', 'g']:
                        #leaf_label = np.argmax(np.bincount(Labels[currentDataIndx].astype(np.int64), minlength=max_label))
                        #nodelabel[current_node] = unique_labels[leaf_label]
                        negative_gradients_in_leaf = negative_gradients.take(TDindx[currentDataIndx], axis=0)
                        hessians_in_leaf = hessians.take(TDindx[currentDataIndx], axis=0)
                        
                        if np.sum(hessians_in_leaf) == 0:
                            val = 0  # Set a default value or handle division by zero appropriately
                        else:
                            val = np.sum(negative_gradients_in_leaf) / np.sum(hessians_in_leaf)
                        """ if current_node < len(nodelabel):
                            nodelabel[current_node] = val
                        else:
                            break """
                        nodelabel[current_node] = val
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
        #feat_impo = giniii
        feat_impo = pd.DataFrame(rows_gini, columns=['feature', 'gini_gain'])
        class_feature_impt_disbu = pd.DataFrame(rows, columns=['feature', 'class', 'v-value'])
        # Below removing the other classes group '0'
        class_feature_impt_disbu = class_feature_impt_disbu[class_feature_impt_disbu['class'] != 0]

        return nodeCutVar, nodeCutValue, childnode, nodelabel, feat_impo, class_feature_impt, class_feature_impt_disbu
    
    

    def tree_predict(self, Data, cut_var, cut_val, nodechilds, nodelabel):
        n, m = Data.shape
        tree_output = np.zeros(n)
        proximity_matrix = np.zeros(n)

        for i in range(n):
            current_node = 0
            while nodechilds[current_node] != 0:
                # cvar: Retrieves the feature index associated with the current node
                cvar = cut_var[current_node]
                # Condition below: Checks if the feature value for the current sample is less than 
                # or equal to the threshold associated with the current node.
                if Data[i, (cvar).astype(np.int64)] <= cut_val[current_node]:
                    current_node = int(nodechilds[current_node])
                else:
                    current_node = int(nodechilds[current_node]+1)
            tree_output[i] = nodelabel[current_node]
            proximity_matrix[i] = current_node
        
        return tree_output, proximity_matrix

    def build_tree(self, i, random_state, Data, Labels, max_depth, nsamtosample, minparent, minleaf, method, nvartosample, weights, oobe, negative_gradients, hessians, true_label):
        print(f'building tree {i}')
        random_instance = np.random.RandomState(random_state)
        #TDindx = random_instance.choice(len(Labels), nsamtosample, replace=False)
        #nsamtosample = 2*(2 ** max_depth)

        # Get unique values in Labels
        unique_values = np.unique(Labels)

        # Initialize TDindx with one index for each unique value
        #TDindx = np.concatenate([np.random.choice(np.where(Labels == value)[0], 1) for value in unique_values])

        TDindx = np.concatenate([random_instance.choice(np.where(Labels == value)[0], 1) 
                                 for value in unique_values if np.any(Labels == value)])
        # If there are more samples to select, randomly select the rest without replacement
        if nsamtosample > len(unique_values):
            additional_indices = random_instance.choice(len(Labels), nsamtosample - len(unique_values), replace=False)
            TDindx = np.concatenate([TDindx, additional_indices])

        # Shuffle the indices to ensure randomness
        random_instance.shuffle(TDindx)

        Random_ForestT = self.tree_fit(Data[TDindx,:], Labels[TDindx], max_depth=max_depth, TDindx=TDindx, random_state=random_state, minparent=minparent, minleaf=minleaf, method=method, nvartosample=nvartosample, weights=weights, negative_gradients=negative_gradients, hessians=hessians, true_subset=true_label[TDindx])
       
        Random_ForestT_dict = {'tree_cut_var': Random_ForestT[0], 'tree_cut_val': Random_ForestT[1],
                                'tree_nodechilds': Random_ForestT[2], 'tree_nodelabel': Random_ForestT[3], 
                                'feature_importances': Random_ForestT[4], 'class_importance_uncomb_val': Random_ForestT[5],
                                'class_feature_impt_disbu': Random_ForestT[6], 'method': method, 'oobe':oobe}
        if oobe:
            NTD = np.setdiff1d(np.arange(len(Labels)), TDindx)
            tree_output = self.tree_predict(Data[NTD,:], Random_ForestT_dict['tree_cut_var'], Random_ForestT_dict['tree_cut_val'],Random_ForestT_dict['tree_nodechilds'], 
            Random_ForestT_dict['tree_nodelabel'])

            if method in ['c', 'g']:
                # oobe_val = np.mean(tree_output != Labels[NTD])
                oobe_val = np.sum(tree_output - Labels[NTD] == 0) / len(NTD)

            elif method == 'r':
                oobe_val = np.mean(np.square(tree_output - Labels[NTD]))

            Random_ForestT_dict['oobe'] = oobe_val

        return Random_ForestT_dict
    
    def fit_sb(self, Data, Labels, random_state):

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
                
                #tree = StochasticBosque(ntrees=1, nvartosample=None, oobe=True)
                
                if self.ntrees==1:
                    random_forest = self.build_tree(i, random_state+i, Data, negative_gradients, self.max_depth, self.nsamtosample, 
                                                            self.minparent, self.minleaf, self.method, self.nvartosample, 
                                                            self.weights, self.oobe, negative_gradients=negative_gradients, hessians=hessians, true_label=y_ohe[:, k])
                else:
                    random_forest = Parallel(n_jobs=-1, backend='threading')(delayed(self.build_tree)
                                                            (i, random_state+i, Data, negative_gradients, self.max_depth, self.nsamtosample, 
                                                            self.minparent, self.minleaf, self.method, self.nvartosample, 
                                                            self.weights, self.oobe, negative_gradients=negative_gradients, hessians=hessians, true_label=y_ohe[:, k]) 
                                                            for i in range(self.ntrees))
                
                
                y_preds= self.predict(Data, random_forest, training=True, n_classes=self.n_classes)
                
                y_preds = pd.to_numeric(y_preds, errors='coerce')
                
                raw_predictions[:, k] += self.learning_rate * y_preds
                probabilities = self._softmax(raw_predictions)
                class_trees.append(random_forest)
                
            self.boosters.append(class_trees)
        return self.boosters

        """ Random_Forest = []
        Random_Forest = Parallel(n_jobs=-1, backend='threading')(delayed(self.build_tree)(i, random_state+i, Data, Labels, self.nsamtosample, self.minparent, self.minleaf, self.method, self.nvartosample, self.weights, self.oobe) for i in range(self.ntrees))
        return Random_Forest """


    def predict_sb(self, Data, Random_Forest, ntrees, proximity):
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

        if ntrees == 1:
            x, leaf_indices[:, 0] = self.tree_predict(Data, Random_Forest['tree_cut_var'], Random_Forest['tree_cut_val'],
                                                          Random_Forest['tree_nodechilds'], Random_Forest['tree_nodelabel'])
            f_votes[0,:] = x.ravel()
            
            """ f_votes[0,:] = (self.tree_predict(Data, Random_Forest['tree_cut_var'], Random_Forest['tree_cut_val'],Random_Forest['tree_nodechilds'], Random_Forest['tree_nodelabel'])).ravel() """
            oobe_values[0] = Random_Forest['oobe']
            method = Random_Forest['method']
        else:
            i = 0
            while i != ntrees:
                x, leaf_indices[:, i] = self.tree_predict(Data, Random_Forest[i]['tree_cut_var'], Random_Forest[i]['tree_cut_val'],
                                                          Random_Forest[i]['tree_nodechilds'], Random_Forest[i]['tree_nodelabel'])
                f_votes[i,:] = x.ravel()
                """ f_votes[i,:] = (self.tree_predict(Data, Random_Forest[i]['tree_cut_var'], Random_Forest[i]['tree_cut_val'],Random_Forest[i]['tree_nodechilds'], Random_Forest[i]['tree_nodelabel'])).ravel() """
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

    # Extract gini value distribution
    def feature_evaluation (self, Data, Random_Forest):
        all_importances = pd.DataFrame(columns=['feature', 'gini_gain'])

        if (len(Random_Forest[0][0])==1):
                n=0
                while (n != len(Random_Forest)):
                    for k in range(len(Random_Forest[0])):
                        importance_per_tree = pd.DataFrame(Random_Forest[n][k]['feature_importances'])
                        class_impt = pd.DataFrame(importance_per_tree, columns=['feature', 'gini_gain'])
                        all_importances = pd.concat([all_importances, class_impt], ignore_index=True)
                    n+=1
        else:
            if(len(Random_Forest[0][0])==9):
                n=0
                while (n != len(Random_Forest)):
                    for k in range(len(Random_Forest[0])):
                        importance_per_tree = pd.DataFrame(Random_Forest[n][k]['feature_importances'])
                        class_impt = pd.DataFrame(importance_per_tree, columns=['feature', 'gini_gain'])
                        all_importances = pd.concat([all_importances, class_impt], ignore_index=True)
                    n+=1
            else:
                n=0
                while (n != len(Random_Forest)):
                    for k in range(len(Random_Forest[0])):
                        for x in range(len(Random_Forest[0][0])):
                            importance_per_tree = pd.DataFrame(Random_Forest[n][k][x]['feature_importances'])
                            class_impt = pd.DataFrame(importance_per_tree, columns=['feature', 'gini_gain'])
                        all_importances = pd.concat([all_importances, class_impt], ignore_index=True)
                    n+=1
        
        return all_importances
    
    #Extract Summed V-values: 
    def class_feature_evaluation (self, Data, Random_Forest, nclasses):
        all_class_features = np.zeros(((Data.shape[1], nclasses)))

        if (len(Random_Forest[0][0])==1):
                n=0
                while (n != len(Random_Forest)):
                    for k in range(len(Random_Forest[0])):
                        #class_tree_nerrors = np.array(Random_Forest[n][k]['nerrors'])
                        class_impt_per_tree = np.array(Random_Forest[n][k]['class_importance_uncomb_val'][:,2])
                        all_class_features[:,k] += class_impt_per_tree
                    n+=1
        else:
            if(len(Random_Forest[0][0])==9):
                n=0
                while (n != len(Random_Forest)):
                    for k in range(len(Random_Forest[0])):
                        #class_tree_nerrors = np.array(Random_Forest[n][k]['nerrors'])
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
            if(len(Random_Forest[0][0])==9):
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

    def plot_top_features(self, importances, col_names, top_n):
        # Get the total count of feature splitting
        total_feature_splitting = importances[:, 1]
        
        # Get indices of top_n features
        top_feature_indices = np.argsort(total_feature_splitting)[::-1][:top_n]

        # Extract top_n features and their counts
        top_features = total_feature_splitting[top_feature_indices]
        
        # Extract top_n feature names
        names = col_names[top_feature_indices+1]

        # Plotting
        plt.bar(range(top_n), top_features, tick_label=names)
        plt.xlabel('Feature Index')
        plt.ylabel('Total Feature Splitting Count')
        plt.title(f'Top {top_n} Features Selected for Splitting')
        plt.show()

    def _one_hot_encode_labels(self, y):
        if isinstance(y, pd.Series): y = y.values
        ohe = OneHotEncoder()
        y_ohe = ohe.fit_transform(y.reshape(-1, 1)).toarray()
        return y_ohe
        
    def _negative_gradients(self, y_ohe, probabilities):
        # Calculating residuals
        # y_ohe (Observed prediction, which remains constant for each class) - probabilities (Predicted prediction) = Residual
        return y_ohe - probabilities
    
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
    
    def predict_proba(self, X, model, training, nclasses):
        '''Generate probability predictions for the given input data.'''
        #tree = StochasticBosque(ntrees=1, nvartosample=None, oobe=True)
        #raw_predictions =  np.full((X.shape[0], self.n_classes), 0.5)
        #raw_predictions =  np.zeros(shape=(X.shape[0], self.n_classes))
        # k = nclasses
        if training==True:
            if len(model)==9:
                raw_predictions =  np.zeros(shape=(X.shape[0], 1))
                y_preds, votes, oobe = self.predict_sb(X, model, ntrees=1, proximity=False)
                y_preds = pd.to_numeric(y_preds, errors='coerce')
                raw_predictions[:, 0] +=self.learning_rate * y_preds            
            else:   
                raw_predictions =  np.zeros(shape=(X.shape[0], 1))         
                y_preds, votes, oobe = self.predict_sb(X, model, ntrees=len(model), proximity=False)
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
                        y_preds, votes, oobe, leaf_indices = self.predict_sb(X, model[n][k], ntrees=1, proximity=True)
                        # Getting predictions
                        y_preds = pd.to_numeric(y_preds, errors='coerce')
                        raw_predictions[:, k] +=self.learning_rate * y_preds
                        # Getting leaf indices for proximity matrix
                        leaf_indices = pd.DataFrame(leaf_indices)
                        class_indices = pd.concat([class_indices, leaf_indices], axis=1)
                    n+=1
                    proximity_matrix = pd.concat([proximity_matrix, class_indices], axis=1)
            else:
                if(len(model[0][0])==9):
                    raw_predictions =  np.zeros(shape=(X.shape[0], len(model[0])))
                    proximity_matrix = pd.DataFrame()
                    n=0
                    while (n != len(model)):
                        class_indices = pd.DataFrame()
                        for k in range(len(model[0])):
                            y_preds, votes, oobe, leaf_indices = self.predict_sb(X, model[n][k], ntrees=1, proximity=True)
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
                            y_preds, votes, oobe, leaf_indices = self.predict_sb(X, model[n][k], ntrees=len(model[0][0]), proximity=True)
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
