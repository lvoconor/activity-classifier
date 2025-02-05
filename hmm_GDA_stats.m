%% GDA with built-in functions

%% Train Model
load('CS229training_data.mat');
load('CS229testing_data.mat');

%featuresToUse = [1 2 3 121 122 123]; %preforms terribly, need more than 6
featuresToUse = 1:561;

X = X_train(:,featuresToUse);
cls = ClassificationDiscriminant.fit(X,y_train);

%% Test Model
load CS229testing_data
trainPredictLabel = predict(cls,X);
trainingAccuracy = sum(y_train==trainPredictLabel)/length(y_train);

testPredictLabel = predict(cls,X_test(:,featuresToUse));
testingAccuracy = sum(y_test==testPredictLabel)/length(y_test);

numStates = 6;

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
est = testEstimatedStates';

% Break down accuracy by subject

for i=unique(subject_test)'
    y = y_test(subject_test==i);
    predict = est(subject_test==i);
    acc = sum(y==predict)/length(y);
    num = length(y);
    fprintf('Subject: %d, Data points: %d, Accuracy: %f\n', i, num, acc);
end

% Generate matrix of real activies vs. predicted activities

[~, activities] = textread('activity_labels.txt','%d %s');

predict = est;
vals = zeros(6,6);
for i=unique(y_test)'
  for j=unique(y_test)'
    acc = sum(j==predict & y_test==i)/sum(y_test==i);
    vals(i,j) = acc;
  end
end

activities
vals
