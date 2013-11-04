%% GDA with built-in functions
format compact
close all
clear all
clc

%% Train Model
load CS229training_data

%featuresToUse = [1 2 3 121 122 123]; %preforms terribly, need more than 6
featuresToUse = 1:561;
X = X_train(:,featuresToUse);

cls = ClassificationDiscriminant.fit(X,y_train);

%% Test Model
load CS229testing_data
predictLabel =  predict(cls,X);
trainingError = sum(y_train==predictLabel)/length(y_train)

predictLabel = predict(cls,X_test(:,featuresToUse));
testingError = sum(y_test==predictLabel)/length(y_test)