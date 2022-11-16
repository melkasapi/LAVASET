import numpy as np
import pandas as pd
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from random import Random, random, randrange
from joblib import Parallel, delayed
import time
from sklearn.metrics import accuracy_score
import pyximport
pyximport.install()
from splitting import get_best_split

from cython.parallel import prange, parallel


class Node:
    '''
    Helper class which implements a single tree node.
    '''
    def __init__(self, feature=None, threshold=None, data_left=None, data_right=None, gini=None, value=None, gini_dict=None):
        self.feature = feature
        self.threshold = threshold
        self.data_left = data_left
        self.data_right = data_right
        self.gini = gini
        self.value = value
        self.gini_dict = gini_dict

def _make_estimator(base_estimator, append=True, random_state=None):
    """Make and configure a copy of the `base_estimator_` attribute.

    Warning: This method should be used to properly instantiate new
    sub-estimators.
    """
    estimator = base_estimator
    # estimator.set_params(**{p: getattr(self, p) for p in self.estimator_params})
    
    estimators_= list()
    if append:
        estimators_.append(estimator)
        return estimators_
    else:
        return estimator

class DecisionTree:
    '''
    Class which implements a decision tree classifier algorithm.
    '''
    def __init__(self, min_samples_split=2, max_depth=1000, random_state=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        # self.max_features = max_features
        self.root = None
        self.nodes = []
        self.n_features_ = 0
        self.random_state = random_state
        

    def _best_split(self, X, y):
        '''
        Helper function, calculates the best split for given features and target
        
        :param X: np.array, features
        :param y: np.array or list, target
        :return: dict
        '''
        n, m = X.shape

        max_features = max(1, int(np.sqrt(m)))
        #max_features = int(m*0.05)
        X_pca = np.zeros((X.shape[0], 1))  # initiating nested array with array number = # of samples
        # loadings = np.zeros(((5*2+1), self.max_features))
        loadings = dict()
    
        #features_subset = np.random.choice(X.shape[1], max_features, replace=False)
        random_instance = np.random.RandomState(self.random_state)
        features_subset = random_instance.randint(0, m, max_features)

        indexes_file = pd.read_csv('~/Documents/cmr_rf/nmr_10nn_index2.csv').to_numpy()
        #indexes_file = pd.read_csv('~/Documents/cmr_rf/nmr_knn_index_10k15k.csv').to_numpy()
        #indexes_file = pd.read_csv('~/Documents/cmr_rf/nmr_50nn_index.csv').to_numpy()
        #indexes_file = pd.read_csv('~/Documents/cmr_rf/nmr_20nn_index.csv').to_numpy()


        scaler = StandardScaler()
        node_neighbors = dict()
        for f_idx in features_subset:
            indeces = indexes_file[f_idx].tolist()
            pca_df = scaler.fit_transform(X[:, indeces])
            #pca_df = X[:, indeces] # takes value for relevant feature in all training samples
            #print(pca_df)
            pca = PCA(n_components=1)
            X_pca = np.append(X_pca, pca.fit_transform(pca_df), axis=1)
            loadings[f_idx] = np.ravel(pca.components_.T)
            node_neighbors[f_idx] = indeces

        X_pca = X_pca[:, 1:]  

        df_pca = np.concatenate((X_pca, np.array(y).reshape(1, -1).T), axis=1)

        node_dict = {}
        node_dict = get_best_split(df_pca, features_subset, node_neighbors, loadings)
        return node_dict


    def _build(self, X, y, depth=0):
        '''
        Helper recursive function, used to build a decision tree from the input data.
        
        :param X: np.array, features
        :param y: np.array or list, target
        :param depth: current depth of a tree, used as a stopping criteria
        :return: Node
        '''
        n_rows, n_cols = X.shape
        self.random_state+=1

        # Check to see if a node should be leaf node
        if n_rows >= self.min_samples_split and depth <= self.max_depth:

            # Get the best split
            best = self._best_split(X, y)
            if best == {}:
                print('this is it')
                pass
            #If the split isn't pure
            if best['gini'] > 0:
                #self.random_state+=1
                self.nodes.append(best)
                # Build a tree on the left
                left = self._build(
                    X=X[best['samples_left'], :], 
                    y=best['y_left'], 
                    depth=depth + 1,
                    # random_state = random_state + 1
                )
                right = self._build(
                    X=X[best['samples_right'], :], 
                    y=best['y_right'], 
                    depth=depth + 1,
                    # random_state = random_state + 1
                )
                return Node(
                        feature=best['predictor'], 
                        threshold=best['split_point'], 
                        data_left=left, 
                        data_right=right, 
                        gini=best['gini'],
                        gini_dict=best['gini_latent']
                )
        return Node(
            value=Counter(y).most_common(1)[0][0]
        )
        
    
    def fit(self, X, y, random_state):
        '''
        Function used to train a decision tree classifier model.
        
        :param X: np.array, features
        :param y: np.array or list, target
        :return: None
        '''
        # Call a recursive function to build the tree
        n, m = X.shape
        self.n_features_ = m
        self.random_state = random_state
        self.root = self._build(X, y)

        #self.nodes.append(self.root)

        self.update_feature_importances()
        # f = open("nodes_100t10nn_changedrs.txt", "a")
        # f.write(repr(self.nodes))
        # f.close()
        
    def _predict(self, x, tree):
        '''
        Helper recursive function, used to predict a single instance (tree traversal).
        
        :param x: single observation
        :param tree: built tree
        :return: float, predicted class
        '''
#         # Leaf node

        if tree.value != None:
            return tree.value
        feature_value = x[tree.feature]
        
        # Go to the left
        if feature_value <= tree.threshold:
            return self._predict(x=x, tree=tree.data_left)
        
        # Go to the right
        if feature_value > tree.threshold:
            return self._predict(x=x, tree=tree.data_right)
        
    def predict(self, X):
        '''
        Function used to classify new instances.
        
        :param X: np.array, features
        :return: np.array, predicted classes
        '''
        # Call the _predict() function for every observation
        return [self._predict(x, self.root) for x in X]
    
    def get_feature_importances(self):
        return self.feature_importances_

    def update_feature_importances(self):
        if self.n_features_ == 0:
            return None
        self.feature_importances_ = np.zeros(self.n_features_)
        J = len(self.nodes)
        print(J)
        if J > 0:
            for j, node in enumerate(self.nodes):
                # pd.DataFrame(node).to_csv('leaves_lavaset10v2.csv', mode='a')
                for var, value in node["gini_latent"].items():
                    self.feature_importances_[var] += value
        self.feature_importances_ /= sum(self.feature_importances_)

        return self.feature_importances_ 


class RandomForest:
    '''
    A class that implements Random Forest algorithm from scratch.
    '''
    def __init__(self, num_trees=10, min_samples_split=2, max_depth=1000, max_samples=100):
        self.num_trees = num_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_samples = max_samples
        # Will store individually trained decision trees
        self.decision_trees = []
    

    @staticmethod
    def _generate_sample_indices(random_state, n_samples, max_samples):
        random_instance = np.random.RandomState(random_state)
        sample_indices = random_instance.randint(0, n_samples, max_samples)
        return sample_indices

    def _parallel_build_trees(self, tree, X, y, idx):

        print("building tree %d of %d" % (idx + 1, self.num_trees))
        # Obtain data sample
        # n_rows, n_cols = X.shape
        #     # Sample with replacement
        max_samples = self.max_samples
        indices = self._generate_sample_indices(idx, X.shape[0], max_samples)
        # # this shuffles features for max_features to be different in trees 
        # f_indices = self._generate_feature_indices(idx, X.shape[1])
        # print(f_indices)
        X_sub = X[indices, :]
        # X_sub = X_sub[:, f_indices]
        _y_sub = y[indices]
        df_train = pd.DataFrame(X_sub)
        df_train[-1] = _y_sub
        # Train
        tree.fit(X_sub, _y_sub, random_state=idx)
        
        # Save the classifier
        self.decision_trees.append(tree)
        return tree


    def fit(self, X, y):
        '''
        Trains a Random Forest classifier.

        :param X: np.array, features
        :param y: np.array, target
        :return: None
        '''
        n_more_dts = self.num_trees - len(self.decision_trees)
        # trees = _make_estimator(DecisionTree(), append=True)

        trees = [_make_estimator(DecisionTree(), append=False)
                for i in range(n_more_dts)
                ]

        # print('make estimator results', trees)
        trees = Parallel(
                n_jobs=-3, # all but 2 CPUs used
                verbose=5,
                backend='multiprocessing')(delayed(self._parallel_build_trees)(tree, X, y, i) 
                for i, tree in enumerate(trees)
            )

        self.decision_trees.extend(trees)

    def predict(self, X):
        '''
        Predicts class labels for new data instances.

        :param X: np.array, new instances to predict
        :return: 
        '''
            # Make predictions with every tree in the forest
        y = []
        for tree in self.decision_trees:
            y.append(tree.predict(X))
        
        #np.savetxt('per_tree_preds_100t10nn_changedrs.txt', np.array(y), delimiter=",")
        
        # Reshape so we can find the most common value
        y = np.swapaxes(a=y, axis1=0, axis2=1)

            # Use majority voting for the final prediction
        predictions = []

        for i, preds in enumerate(y):
            counter = Counter(y[i])
            predictions.append(counter.most_common(1)[0][0])
        return predictions
        
    def get_importances(self, X):
        all_importances = np.zeros(X.shape[1])
        for tree in self.decision_trees:
            importance_per_tree = np.array(tree.feature_importances_)
            # all_importances = np.concatenate(all_importances, importance_per_tree)
            all_importances = np.vstack((all_importances, importance_per_tree))

        all_importances = np.mean(all_importances, axis=0, dtype=np.float64)
        return all_importances / np.sum(all_importances)


nmr_peaks = pd.read_csv('~/Documents/IBS/NMR_data/IBS_HNMR_data_n267.csv')

X = np.array(nmr_peaks.iloc[:, 3:])
y = np.array(nmr_peaks.iloc[:, 1])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80)
# print(pd.DataFrame(X_train))

scaler = StandardScaler()
scaler.fit(X_train)
X_test = scaler.transform(X_test)

df_train = pd.DataFrame(X_train)
df_train[-1] = y_train
df_test = pd.DataFrame(X_test)
df_test[-1] = y_test
start = time.time()
tree = RandomForest()
tree.fit(X_train, y_train)
preds = tree.predict(X_test)
print(accuracy_score(preds, y_test))

#print(tree.nodes)
feat = tree.get_importances(X_train)
end = time.time()
print(end-start)

#pd.DataFrame(feat).to_csv('feature_impo_pca_100t10nn_changed_rs.csv')

# simulated_groups = pd.read_excel('simulated_groups.xlsx', sheet_name=0)
# simulated_impo = pd.read_excel('simulated_groups.xlsx', sheet_name=1)
# nmr_peaks = pd.read_csv('~/Documents/IBS/NMR_data/IBS_HNMR_data_n267.csv')
# nmr_peaks.insert(2, 'sim_groups', value=simulated_groups.simulated_class)
# nmr_peaks.sim_groups.replace({2:0}, inplace=True)
# X = np.array(nmr_peaks.iloc[:, 4:])
# y = np.array(nmr_peaks.iloc[:, 2])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
# # print(pd.DataFrame(X_train))

# scaler = StandardScaler()
# scaler.fit(X_train)
# X_test = scaler.transform(X_test)

# df_train = pd.DataFrame(X_train)
# df_train[-1] = y_train
# display(df_train.shape)
# df_test = pd.DataFrame(X_test)
# df_test[-1] = y_test
# start = time.time()
# tree = RandomForest()
# tree.fit(X_train, y_train)
# preds = tree.predict(X_test)
# print(accuracy_score(preds, y_test))

#print(tree.nodes)
# feat = tree.get_importances(X_train)
# end = time.time()
# print(end-start)

# pd.DataFrame(feat).to_csv('feature_impo_pca_100t10nnSIM45.csv')

