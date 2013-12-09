%% HMM on raw data

numBuckets = 200;
numStates = 6;

load('rawInertialTrain.mat');
load('rawInertialTest.mat');

% Discretize data
discreteTrainX = round((raw_X_train + 1) * .5 * (numBuckets - 1) + 1);
discreteTestX = round((raw_X_test + 1) * .5 * (numBuckets - 1) + 1);

% Estimate transition and emission probabilities
pseudo_trans = ones(numStates,numStates);
pseudo_emis = ones(numStates,numBuckets);
size(discreteTrainX)
size(pseudo_trans)
[trans,emis] = hmmestimate(discreteTrainX, raw_Y_train, 'Pseudotransitions',...
                           pseudo_trans, 'Pseudoemissions', pseudo_emis);

% Estimate states of testing data
estimatedStates = hmmviterbi(discreteTestX,trans,emis);
testingAccuracy = sum(raw_Y_test==estimatedStates')/length(raw_Y_test)
