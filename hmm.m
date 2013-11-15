%% HMM on raw data

for numBuckets = 200;
numStates = 6;

load('CS229training_data.mat');
load('CS229testing_data.mat');

% Discretize data
discreteTrainX = round((X_train + 1) * .5 * (numBuckets - 1) + 1);
discreteTestX = round((X_test + 1) * .5 * (numBuckets - 1) + 1);

% Estimate transition and emission probabilities
pseudo_trans = ones(numStates,numStates);
pseudo_emis = ones(numStates,numBuckets);
[trans,emis] = hmmestimate(discreteTrainX, y_train, 'Pseudotransitions',...
                           pseudo_trans, 'Pseudoemissions', pseudo_emis);

% Estimate states of testing data
estimatedStates = hmmviterbi(discreteTestX,trans,emis);
testingAccuracy = sum(y_test==estimatedStates')/length(y_test)

end
