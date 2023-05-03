import numpy as np
import pandas as pd
from lavaset import RandomForest, DecisionTree
from calculate_distance import knn_calculation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
import sys

## LOAD DATASET

nmr_peaks = pd.read_csv('~/Documents/IBS/NMR_data/IBS_HNMR_data_n267.csv')
nn = knn_calculation(nmr_peaks.columns[14000:18000], 10)
X = np.array(nmr_peaks.iloc[:, 14000:18000])
#y = np.array(nmr_peaks.iloc[:, 1])
y = pd.read_csv('~/Documents/cmr_rf/LAVASET/testing/formate_cluster_labels.txt', header=None).iloc[:, 0].to_numpy(dtype=int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=180)
# print(pd.DataFrame(X_train))

scaler = StandardScaler(with_std=False)
scaler.fit(X_train)
X_test = scaler.transform(X_test)

start = time.time()
model = RandomForest(num_trees=10, knn=nn)
# model = DecisionTree(knn=nn)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(preds)
print(accuracy_score(preds, y_test))

#print(tree.nodes)
feat = model.get_importances(X_train)
end = time.time()
print(end-start)
print(feat)
pd.DataFrame(feat).to_csv('feature_impo_pca_10t10nn_formate_centered_subset1418k.csv')

feat_count = model.get_feature_counts(X_train)
pd.DataFrame(feat_count).to_csv('feature_count_10T10nn_centered_subset1418k', mode='w')

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

#pd.DataFrame(feat).to_csv('testing/feature_impo_pca_100t10nnFORMATE2.csv')