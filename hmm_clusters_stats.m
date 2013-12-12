%% HMM on raw data

% parameters
numclusters = 16;
% each training size is multiplied by 512
% e.g. min_num_examples = 16 will use 16 * 512 = 8192 raw data points
min_num_examples = 16;
max_num_examples = 39;
outfile = 'stats.txt';

load('rawInertialTrain.mat')
load('rawInertialTest.mat');
Xtest = raw_X_test;
ytest = raw_Y_test;

numStates = 6;

% Estimate transition and emission probabilities
pseudo_trans = ones(numStates,numStates);
pseudo_emis = ones(numStates, numclusters);

results = zeros(max_num_examples-min_num_examples+1,3);

for exnum=min_num_examples:max_num_examples
i = exnum * 512;
Xtrain = raw_X_train(1:i,:);
ytrain = raw_Y_train(1:i);
[numExamples, numFeatures] = size(Xtrain);
[clusterindices, clustercentroids] = kmeans(Xtrain, numclusters);

[numEmissions, numFeatures] = size(clusterindices);
[trans,emis] = hmmestimate(clusterindices, ytrain,'Pseudotransitions',pseudo_trans,'Pseudoemissions',pseudo_emis);

% Estimate states of testing data
trainEstimatedStates = hmmviterbi(clusterindices,trans,emis);
trainingAccuracy = sum(ytrain==trainEstimatedStates')/length(ytrain);

% Estimate states of testing data
testNearestClusters = dsearchn(clustercentroids, Xtest);
testEstimatedStates = hmmviterbi(testNearestClusters,trans,emis);
testingAccuracy = sum(ytest==testEstimatedStates')/length(ytest);

results(exnum-5,:) = [numExamples, trainingAccuracy, testingAccuracy];

fprintf('Number of training examples: %d, training accuracy: %f, testing accuracy: %f', i, trainingAccuracy, testingAccuracy);
end

dlmwrite(outfile,results)
results
