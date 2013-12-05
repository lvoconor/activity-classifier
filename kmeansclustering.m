% use all 9 features for clustering? 
load('rawInertialTrain.mat')
numclusters = 5;
[clusterindices, clustercentroids] = kmeans(raw_X_train, numclusters);
