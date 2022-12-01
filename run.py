import numpy as np
import pandas as pd
from lavaset import RandomForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
import sys

## LOAD DATASET

nmr_peaks = pd.read_csv('~/Documents/IBS/NMR_data/IBS_HNMR_data_n267.csv')

X = np.array(nmr_peaks.iloc[:, 3:])
#y = np.array(nmr_peaks.iloc[:, 1])
y = pd.read_csv('testing/formate_cluster_labels.txt', header=None).iloc[:, 0].to_numpy(dtype=int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80)
# print(pd.DataFrame(X_train))

scaler = StandardScaler()
scaler.fit(X_train)
X_test = scaler.transform(X_test)

start = time.time()
model = RandomForest(num_trees=100)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(preds)
print(accuracy_score(preds, y_test))

#print(tree.nodes)
feat = model.get_importances(X_train)
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

pd.DataFrame(feat).to_csv('testing/feature_impo_pca_100t10nnFORMATE2.csv')