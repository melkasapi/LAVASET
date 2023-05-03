
from asyncio.staggered import staggered_race
from best_cut_node import best_cut_node
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from sklearn.metrics import accuracy_score
import time
from joblib import Parallel, delayed



st = time.time()


nmr_peaks = pd.read_csv('~/Documents/IBS/NMR_data/IBS_HNMR_data_n267.csv')
X = np.array(nmr_peaks.iloc[:, 3:])
# # y = np.array(nmr_peaks.iloc[:, 1], dtype=int)
y = pd.read_csv('formate-testing/formate_cluster_labels.txt', header=None).iloc[:, 0].to_numpy(dtype=np.double)
# y = np.array(nmr_peaks.iloc[:, 1], dtype=int)
# # y = pd.read_csv('~/Documents/cmr_rf/LAVASET/testing/formate_cluster_labels.txt', header=None).iloc[:, 0].to_numpy(dtype=int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=180)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def cartree(Data, Labels, minparent=2, minleaf=1, nvartosample=None, method='g', weights=None):
    if nvartosample is None:
        nvartosample = Data.shape[1]
   
    n = len(Labels)
    L = int(2 * np.ceil(n / minleaf) - 1)
    m = Data.shape[1]

    # nodeDataIndx = [None] * int(L)
    # nodeDataIndx = [np.array([], dtype=np.int64)] * int(L)
    nodeDataIndx = {0: np.arange(n)}
    nodeDataIndx[0] = np.arange(n)

    nodeCutVar = np.zeros(int(L))
    nodeCutValue = np.zeros(int(L))

    nodeflags = np.zeros(int(L+1))

    nodelabel = np.full(int(L), np.inf)
    nodelabel = np.zeros(int(L))

    childnode = np.zeros(int(L))
    giniii = np.zeros((m,3))

    # gini_dict = {key: 0 for key in range(m)}
    nodeflags[0] = 1

    if method.lower() in ['c', 'g']:
        unique_labels = np.unique(Labels)
        max_label = len(unique_labels)
    else:
        max_label = None

    current_node = 0
    
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
                node_var = np.random.permutation(m)
                node_var = node_var[:nvartosample]
                # node_var = range(m)
                giniii[node_var,0]+=1

                # random_state=10
                # random_instance = np.random.RandomState(random_state)
                # node_var = random_instance.choice(m, nvartosample, replace=False)
                # print(node_var, 'node_var')
                if weights is not None:
                    Wcd = weights[currentDataIndx]
                else:
                    Wcd = None
                
                bestCutVar, bestCutValue = best_cut_node(method, Data[currentDataIndx][:, node_var], Labels[currentDataIndx], minleaf, max_label)
                # bestCutVar = int(bestCutVar_before - 1) # only needed if using cpp implementation
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
                    # gini_dict[node_var[bestCutVar]] += gini_gain
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

        current_node += 1
    # feat_impo = np.array(list(gini_dict.values()), dtype=np.float64)
    # feat_impo /= np.sum(feat_impo)
    feat_impo = giniii
    return nodeCutVar, nodeCutValue, childnode, nodelabel, feat_impo

# cut_var, cut_val, nodechilds, nodelabel, feat_impo = cartree(X_train, y_train, minparent=2, minleaf=1, nvartosample=120, method='g', weights=None)

# print(cut_var, nodechilds, nodelabel)


def eval_cartree(Data, cut_var, cut_val, nodechilds, nodelabel):
    n, m = Data.shape
    tree_output = np.zeros(n)

    # feature_importances = dict(zip(range(m), feat_impo))

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

# y_pred = eval_cartree(X_test,cut_var, cut_val, nodechilds, nodelabel)
# print(y_test)
# print(accuracy_score(y_test, np.array(y_pred, dtype=int)))

# import sys
# sys.exit()


def Stochastic_Bosque(Data, Labels, **kwargs):
    # Set default parameter values
    minparent = kwargs.get('minparent', 2)
    minleaf = kwargs.get('minleaf', 1)
    nvartosample = kwargs.get('nvartosample', int(np.sqrt(Data.shape[1])))
    ntrees = kwargs.get('ntrees', 100)
    nsamtosample = kwargs.get('nsamtosample', Data.shape[0])
    method = kwargs.get('method', 'g')
    oobe = kwargs.get('oobe', False)
    weights = kwargs.get('weights', None)

    Random_Forest = []
    random_state = 0
    def build_tree(i, random_state, Data, Labels, nsamtosample, minparent, minleaf, method, nvartosample, weights, oobe):
        print(i)
        random_instance = np.random.RandomState(random_state)
        TDindx = random_instance.choice(len(Labels), nsamtosample, replace=False)

        Random_ForestT = cartree(Data[TDindx,:], Labels[TDindx], minparent=minparent, minleaf=minleaf, method=method, nvartosample=nvartosample, weights=weights)
        Random_ForestT_dict = {'tree_cut_var': Random_ForestT[0], 'tree_cut_val': Random_ForestT[1],
                                'tree_nodechilds': Random_ForestT[2], 'tree_nodelabel': Random_ForestT[3], 
                                'feature_importances': Random_ForestT[4],
                                'method': method, 'oobe':oobe}
        if oobe:
            NTD = np.setdiff1d(np.arange(len(Labels)), TDindx)
            tree_output = eval_cartree(Data[NTD,:], Random_ForestT)

            if method in ['c', 'g']:
                oobe_val = np.mean(tree_output != Labels[NTD])
            elif method == 'r':
                oobe_val = np.mean(np.square(tree_output - Labels[NTD]))

            Random_ForestT_dict['oobe'] = oobe_val

        return Random_ForestT_dict

    # for i in range(ntrees):
    #     print(i)
    #     random_instance = np.random.RandomState(random_state)
    #     TDindx = random_instance.choice(len(Labels), nsamtosample, replace=False)
    #     # TDindx = np.random.choice(len(Labels), size=nsamtosample, replace=True)
    #     # TDindx = np.unique(TDindx)

    #     Random_ForestT = cartree(Data[TDindx,:], Labels[TDindx], minparent=minparent, minleaf=minleaf, method=method, nvartosample=nvartosample, weights=weights)
    #     Random_ForestT_dict = {'tree_cut_var': Random_ForestT[0], 'tree_cut_val': Random_ForestT[1],
    #                             'tree_nodechilds': Random_ForestT[2], 'tree_nodelabel': Random_ForestT[3], 'feature_importances': Random_ForestT[4],
    #                             'method': method, 'oobe':oobe}
    #     random_state+=1
    #     if oobe == 'y':
    #         NTD = np.setdiff1d(np.arange(len(Labels)), TDindx)
    #         tree_output = eval_cartree(Data[NTD,:], Random_ForestT)

    #         if method in ['c', 'g']:
    #             oobe_val = np.mean(tree_output != Labels[NTD])
    #         elif method == 'r':
    #             oobe_val = np.mean(np.square(tree_output - Labels[NTD]))

    #         Random_ForestT_dict['oobe'] = oobe_val

    #     Random_Forest.append(Random_ForestT_dict)

    # return Random_Forest

    random_state = 0
    Random_Forest = Parallel(n_jobs=-1)(delayed(build_tree)(i, random_state+i, Data, Labels, 
    nsamtosample, minparent, minleaf, method, nvartosample, weights, oobe) for i in range(ntrees))

    return Random_Forest

RF = Stochastic_Bosque(X_train, y_train) 


def eval_Stochastic_Bosque(Data, Random_Forest, oobe=False):
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
    all_importances = np.zeros((Data.shape[1], 3))
    for i, tree in enumerate(Random_Forest):
        f_votes[i,:] = (eval_cartree(Data, Random_Forest[i]['tree_cut_var'], Random_Forest[i]['tree_cut_val'],Random_Forest[i]['tree_nodechilds'], Random_Forest[i]['tree_nodelabel'])).ravel()
        oobe_values[i] = tree['oobe']
        importance_per_tree = np.array(Random_Forest[i]['feature_importances'])
            # all_importances = np.concatenate(all_importances, importance_per_tree)
        # all_importances = np.vstack((all_importances, importance_per_tree))
        all_importances += importance_per_tree

    # all_importances = np.mean(all_importances, axis=0, dtype=np.float64)

    method = Random_Forest[0]['method']

    if method in ['c', 'g']:
        unique_labels, indices = np.unique(f_votes, return_inverse=True)
        f_votes = indices.reshape((len(Random_Forest), Data.shape[0]))
        if oobe:
            weights = ~oobe + oobe * oobe_values
        else:
            weights = None
        f_output = np.apply_along_axis(lambda x: np.bincount(x, weights=weights, minlength=len(unique_labels)).argmax(),
                                       axis=0, arr=f_votes)
        f_output = unique_labels[f_output]
    elif method == 'r':
        f_output = np.mean(f_votes, axis=0)

    return f_output, f_votes, all_importances

# print(eval_cartree(X_test, cut_var, cut_val, nodechilds, nodelabel))

y_pred, votes, impo = eval_Stochastic_Bosque(X_test, RF, oobe=False)
en = time.time()
print(en-st)
# pd.DataFrame(impo).to_csv('feature_impo_100t10nn_formate_allgini_newcpp.csv')

y_pred = np.array(eval_Stochastic_Bosque(X_test, RF, oobe=False)[0], dtype=int)
print(y_pred)
print(accuracy_score(y_test, y_pred))
#minparent=2, minleaf=1, nvartosample=140,ntrees=50, nsamtosample=100, method='g', oobe='n', weights=None))