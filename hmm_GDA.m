%% HMM on raw data

GDA;

numStates = 6;

load('CS229training_data.mat');
load('CS229testing_data.mat');

% Estimate transition and emission probabilities
pseudo_trans = ones(numStates,numStates);
pseudo_emis = ones(numStates,numStates);
[trans,emis] = hmmestimate(trainPredictLabel,y_train,'Pseudotransitions',pseudo_trans,'Pseudoemissions',pseudo_emis);

% Estimate states of testing data
trainEstimatedStates = hmmviterbi(trainPredictLabel,trans,emis);
trainingAccuracy = sum(y_train==trainEstimatedStates')/length(y_train)

% Estimate states of testing data
testEstimatedStates = hmmviterbi(testPredictLabel,trans,emis);
testingAccuracy = sum(y_test==testEstimatedStates')/length(y_test)
