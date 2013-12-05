%% GDA with built-in functions
format compact
close all
clear all
clc

%% Train Model
%load CS229training_data
load rawInertialTrain
y_train = raw_Y_train;
X_train = raw_X_train;
%featuresToUse = 1:561;
featuresToUse = 1:9;
X = X_train(:,featuresToUse);

cls = ClassificationDiscriminant.fit(X,y_train,'discrimType','linear');

%% Test Model
%load CS229testing_data
load rawIntertialTest
y_test = raw_Y_test;
X_test = raw_X_test;
predictLabel =  predict(cls,X);
trainingAccuracy = sum(y_train==predictLabel)/length(y_train)

predictLabel = predict(cls,X_test(:,featuresToUse));
testingAccuracy = sum(y_test==predictLabel)/length(y_test)

%% Backwards Feature Search
close all
clear all
clc

load CS229training_data
load CS229testing_data
feats = textread('features.txt','%s');

firstNFeatures = 15;
sparseTrainFactor = 1;
m = length(X_train(:,1));
X_train = X_train(1:sparseTrainFactor:m,1:firstNFeatures);
y_train = y_train(1:sparseTrainFactor:m);
m = length(X_test(:,1));
X_test = X_test(1:1:m,1:firstNFeatures);
y_test = y_test(1:1:m);

featuresRemoved = zeros(firstNFeatures-1,1);
testAcc = zeros(firstNFeatures-1,1);
trainAcc = zeros(firstNFeatures-1,1);

for i=1:firstNFeatures-1
    featuresLeft = length(X_train(1,:));
    allAccuracy = zeros(featuresLeft,1);
    for j = 1:featuresLeft %number of features left
        featuresToUse = [1:j-1 j+1:featuresLeft]; %use all but jth feature
        X = X_train(:,featuresToUse);
        cls = ClassificationDiscriminant.fit(X,y_train,'discrimType','linear');
        predictLabel = predict(cls,X_test(:,featuresToUse));
        allAccuracy(j) = sum(y_test==predictLabel)/length(y_test);
        fprintf('Removing feature %d \n',j);
    end
    [worstFeatAcc,worstFeatIndex] = max(allAccuracy);
    X_train = X_train(:,[1:worstFeatIndex-1 worstFeatIndex+1:featuresLeft]);
    % find train error
    cls = ClassificationDiscriminant.fit(X_train,y_train,'discrimType','linear');
    predictLabel = predict(cls,X_train);
    trainAcc(i) = sum(y_train==predictLabel)/length(y_train);
    
    featuresRemoved(i) = worstFeatIndex;
    testAcc(i) = worstFeatAcc;
    fprintf('\n\n %d Features Left \n\n\n',featuresLeft-1);
end

hold all
plot(testAcc);
plot(trainAcc);
legend('Test Accuracy','Training Accuracy');
xlabel('Features Removed');
title('Backwards Feature Search (first 100 features)');

%% Forward Feature Search (not done, maybe not necessary)
close all
clear all
clc

load CS229training_data
load CS229testing_data
%feats = textread('features.txt','%s');

sparseTrainFactor = 70;
m = length(X_train(:,1));
X_train = X_train(1:sparseTrainFactor:m,1:15);
y_train = y_train(1:sparseTrainFactor:m);
m = length(X_test(:,1));
X_test = X_test(1:50:m,1:15);
y_test = y_test(1:50:m);

featsAdded = zeros(15,1);
testAcc = zeros(15,1);

X_outer = []; %this is X_train with features added one by one
X_inner = X_train; %this is X_train with features removed one by one

for i=1:15 %adding features
    currentFeatures = length(X_outer(1,:));
    allAccuracy = zeros(15-currentFeatures,1);
    for j=1:15-currentFeatures %loop through all remaining features
        featuresToUse = [1:j-1 j+1:featuresLeft]; %use all but jth feature
        X = X_inner(:,featuresToUse);
        cls = ClassificationDiscriminant.fit(X,y_train);
        predictLabel = predict(cls,X_test(:,featuresToUse));
        allAccuracy(j) = sum(y_test==predictLabel)/length(y_test);
        fprintf('Removing feature %d \n',j);
        
    end
    
end


%%

[allSort,sortInd] = sort(allAccuracy);
plot(allSort);
ylabel('Test Accuracy');
xlabel('Sorted Feature Index');

top5 = zeros(5,1);
for i = 1:5
    top5(i) = find(sortInd==i);
end
feats(top5*2)

bot5 = zeros(5,1);
for i = 1:5
    bot5(i) = find(sortInd==(562-i));
end
feats(bot5*2)

