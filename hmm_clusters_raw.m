%% HMM on raw data

kmeansclustering;

numStates = 6;
[numEmissions, numFeatures] = size(clusterindices);

load('rawInertialTrain.mat');
load('rawIntertialTest.mat');

% Estimate transition and emission probabilities
pseudo_trans = ones(numStates,numStates);
pseudo_emis = ones(numStates, numclusters);
[trans,emis] = hmmestimate(clusterindices, raw_Y_train,'Pseudotransitions',pseudo_trans,'Pseudoemissions',pseudo_emis);

trans;  

% Estimate states of testing data
trainEstimatedStates = hmmviterbi(clusterindices,trans,emis);
trainingAccuracy = sum(raw_Y_train==trainEstimatedStates')/length(raw_Y_train)

% Estimate states of testing data
testNearestClusters = dsearchn(clustercentroids, raw_X_test);
testEstimatedStates = hmmviterbi(testNearestClusters,trans,emis);
testingAccuracy = sum(raw_Y_test==testEstimatedStates')/length(raw_Y_test)
