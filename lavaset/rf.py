import lavaset.best_cut_node as best_cut_node
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


class StochasticBosque:   
    def __init__(self, minparent=2, minleaf=1, nvartosample=None, ntrees=100, nsamtosample=None, method='g', oobe=False, weights=None):
        self.minparent = minparent
        self.minleaf = minleaf
        self.nvartosample = nvartosample
        self.ntrees = ntrees
        self.nsamtosample = nsamtosample
        self.method = method
        self.oobe = oobe
        self.weights = weights
        # self.Random_Forest = None


    def tree_fit(self, Data, Labels, random_state, minparent, minleaf, nvartosample, method, weights):
    
        n = len(Labels)
        L = int(2 * np.ceil(n / minleaf) - 1)
        m = Data.shape[1]

        nodeDataIndx = {0: np.arange(n)}

        nodeCutVar = np.zeros(int(L))
        nodeCutValue = np.zeros(int(L))

        nodeflags = np.zeros(int(L+1))

        nodelabel = np.full(int(L), np.inf)
        nodelabel = np.zeros(int(L))

        childnode = np.zeros(int(L))
        giniii = np.zeros((m,3))

        nodeflags[0] = 1

        if method.lower() in ['c', 'g']:
            unique_labels = np.unique(Labels)
            max_label = len(unique_labels)
        else:
            max_label = None

        current_node = 0
        # random_state=random_state

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
                    # node_var = random_instance.permutation(range(0,m))[:nvartosample]


                    node_var = np.random.permutation(range(0,m))[:nvartosample]

                    giniii[node_var,0]+=1
                    if weights is not None:
                        Wcd = weights[currentDataIndx]
                    else:
                        Wcd = None
                    bestCutVar, bestCutValue = best_cut_node.best_cut_node(method, Data[currentDataIndx][:, node_var], Labels[currentDataIndx], minleaf, max_label)
                    random_state+=1
                    if bestCutVar != -1:

                        nodeCutVar[current_node] = node_var[bestCutVar]
                        nodeCutValue[current_node] = bestCutValue
                        giniii[node_var[bestCutVar],1]+= 1

                        nodeDataIndx[free_node] = currentDataIndx[Data[currentDataIndx, node_var[bestCutVar]] <= bestCutValue]
                        nodeDataIndx[free_node+1] = currentDataIndx[Data[currentDataIndx, node_var[bestCutVar]] > bestCutValue]
                        
                        y_left=Labels[nodeDataIndx[free_node]]
                        y_right=Labels[nodeDataIndx[free_node+1]]
                        y_parent = list(y_left)+list(y_right)
        
                        proportion_left = len(y_left) / len(y_parent)
                        proportion_right = len(y_right) / len(y_parent)
                        p_parent = (np.bincount(np.array(y_parent, dtype=np.int64)))/len(y_parent)

                        p_left = (np.bincount(np.array(y_left, dtype=np.int64)))/len(y_left)
                        p_right = (np.bincount(np.array(y_right, dtype=np.int64)))/len(y_right)
                        gini_l = 1-np.sum(p_left**2)
                        gini_r = 1-np.sum(p_right**2)
                        gini_p = 1-np.sum(p_parent**2)
                        
                        gini_gain = gini_p - (proportion_left*gini_l + proportion_right*gini_r)
                        giniii[node_var[bestCutVar], 2] += gini_gain
                        nodeflags[free_node:(free_node + 2)] = 1
                        childnode[current_node] = free_node
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
        feat_impo = giniii

        return nodeCutVar, nodeCutValue, childnode, nodelabel, feat_impo

    def tree_predict(self, Data, cut_var, cut_val, nodechilds, nodelabel):
        n, m = Data.shape
        tree_output = np.zeros(n)

        for i in range(n):
            current_node = 0
            while nodechilds[current_node] != 0:
                cvar = cut_var[current_node]
                if Data[i, (cvar).astype(np.int64)] <= cut_val[current_node]:
                    current_node = int(nodechilds[current_node])
                else:
                    current_node = int(nodechilds[current_node]+1)
            tree_output[i] = nodelabel[current_node]
        
        return tree_output

    def build_tree(self, i, random_state, Data, Labels, nsamtosample, minparent, minleaf, method, nvartosample, weights, oobe):
        print(i)
        random_instance = np.random.RandomState(random_state)
        TDindx = random_instance.choice(len(Labels), nsamtosample, replace=False)
        Random_ForestT = self.tree_fit(Data[TDindx,:], Labels[TDindx], random_state=random_state, minparent=minparent, minleaf=minleaf, method=method, nvartosample=nvartosample, weights=weights)
       
        Random_ForestT_dict = {'tree_cut_var': Random_ForestT[0], 'tree_cut_val': Random_ForestT[1],
                                'tree_nodechilds': Random_ForestT[2], 'tree_nodelabel': Random_ForestT[3], 
                                'feature_importances': Random_ForestT[4],
                                'method': method, 'oobe':oobe}
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
        print('here')
        if self.nsamtosample is None:
            self.nsamtosample = Data.shape[0]
        else:
            self.nsamtosample = self.nsamtosample
        
        if self.nvartosample is None:
            self.nvartosample = Data.shape[1]
        elif self.nvartosample == 'sqrt':
            self.nvartosample = int(np.sqrt(Data.shape[1]))


        Random_Forest = []
        Random_Forest = Parallel(n_jobs=-1, backend='threading')(delayed(self.build_tree)(i, random_state+i, Data, Labels, self.nsamtosample, self.minparent, self.minleaf, self.method, self.nvartosample, self.weights, self.oobe) for i in range(self.ntrees))
        return Random_Forest


    def predict_sb(self, Data, Random_Forest):
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
        f_votes = np.zeros((len(Random_Forest), Data.shape[0]), dtype=object)
        oobe_values = np.zeros((len(Random_Forest),))
        for i, tree in enumerate(Random_Forest):
            f_votes[i,:] = (self.tree_predict(Data, Random_Forest[i]['tree_cut_var'], Random_Forest[i]['tree_cut_val'],Random_Forest[i]['tree_nodechilds'], Random_Forest[i]['tree_nodelabel'])).ravel()
            oobe_values[i] = tree['oobe']

        method = Random_Forest[0]['method']

        if method in ['c', 'g']:
            unique_labels, indices = np.unique(f_votes, return_inverse=True)
            f_votes = indices.reshape((len(Random_Forest), Data.shape[0]))
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
        return f_output, f_votes, oobe_mean
    
    def feature_evaluation (self, Data, Random_Forest):
        all_importances = np.zeros((Data.shape[1], 3))
        for i, tree in enumerate(Random_Forest):
            importance_per_tree = np.array(Random_Forest[i]['feature_importances'])
            all_importances += importance_per_tree
        return all_importances

# results = []
# for i in range(0, 20):
#     print('random_state', i)
#     sb = StochasticBosque(ntrees=100, nvartosample='sqrt', nsamtosample=150)
#     RF = sb.fit_sb(X_train, y_train, random_state=i)   
#     y_pred, votes = sb.predict_sb(X_test, RF, oobe=False)
#     accuracy = accuracy_score(y_test, np.array(y_pred, dtype=int))
#     precision = precision_score(y_test, np.array(y_pred, dtype=int), average='weighted')
#     recall = recall_score(y_test, np.array(y_pred, dtype=int), average='weighted')
#     f1 = f1_score(y_test, np.array(y_pred, dtype=int), average='weighted')

#     result = {'Random State': i, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
#     results.append(result)
#     pd.DataFrame(sb.feature_evaluation(X_train, RF)).to_csv(f'classicRF_feature_impo_100t10nnIBSvsHC_random_state{i}v2.csv')

# print(results)
# fields = ['Random State', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
# en = time.time()
# print(en-st)
# with open('classicRF_metrics_100t10nnIBSvsHCv2.csv', 'w', newline='') as file:
#     writer = csv.DictWriter(file, fieldnames=fields)
#     writer.writeheader()  # Write header
#     writer.writerows(results)  # Write multiple rows

# pd.DataFrame(impo).to_csv('feature_impo_100t10nn_formate_allgini_newcpp.csv')
# print(y_pred)
#minparent=2, minleaf=1, nvartosample=140,ntrees=50, nsamtosample=100, method='g', oobe='n', weights=None))