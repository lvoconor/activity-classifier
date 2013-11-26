%% GDA with built-in functions
format compact
close all
clear all
clc

%% Train Model
%load CS229training_data
load rawIntertialTrain
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

sparseTrainFactor = 1;
m = length(X_train(:,1));
X_train = X_train(1:sparseTrainFactor:m,:);
y_train = y_train(1:sparseTrainFactor:m);
m = length(X_test(:,1));
X_test = X_test(1:1:m,:);
y_test = y_test(1:1:m);

for i=1:1
    allAccuracy = zeros(length(X_train(1,:)),1);
    tic
    for j = 1:length(X_train(1,:)) %number of features left
    %for j = 1:1
        featuresToUse = [1:j-1 j+1:561]; %use all but jth feature
        X = X_train(:,featuresToUse);
        cls = ClassificationDiscriminant.fit(X,y_train);
        predictLabel = predict(cls,X_test(:,featuresToUse));
        allAccuracy(j) = sum(y_test==predictLabel)/length(y_test);
        fprintf('Removing feature %d \n',j);
    end
    toc
    
end

[allSort,sortInd] = sort(allAccuracy);
plot(allSort);
ylabel('Test Accuracy');
xlabel('Sorted Feature Index');
%%
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