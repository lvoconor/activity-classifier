%% GDA with built-in functions

%% Train Model
load CS229training_data

%featuresToUse = [1 2 3 121 122 123]; %preforms terribly, need more than 6
featuresToUse = 1:561;
X = X_train(:,featuresToUse);

cls = ClassificationDiscriminant.fit(X,y_train);

%% Test Model
load CS229testing_data
trainPredictLabel =  predict(cls,X);
trainingAccuracy = sum(y_train==trainPredictLabel)/length(y_train)

testPredictLabel = predict(cls,X_test(:,featuresToUse));
testingAccuracy = sum(y_test==testPredictLabel)/length(y_test)