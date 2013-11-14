%% HMM on raw data

numBuckets = 10^4;
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

% Update prior probabilities
trans_hat = zeros(numStates+1,numStates+1);
trans_hat(2:numStates+1,2:numStates+1) = trans;
trans_hat(1,2:numStates+1) = 1/numStates;
emis_hat = [zeros(size(emis(1,:))); emis];

% Estimate states of testing data
estimatedStates = hmmviterbi(discreteTestX,trans_hat,emis_hat);
testingAccuracy = sum(y_test==estimatedStates')/length(y_test)