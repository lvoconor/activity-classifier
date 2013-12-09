%% GDA with built-in functions

%% Train Model
load CS229training_data

%featuresToUse = [1 2 3 121 122 123]; %preforms terribly, need more than 6
featuresToUse = 1:561;

X = X_train(:,featuresToUse);
cls = ClassificationDiscriminant.fit(X,y_train);

%% Test Model
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


%% PCA (calculate eigenvalue decomp)
close all 
clear all

load CS229training_data
load CS229testing_data
[m,n] = size(X_train);
mtest = size(X_test,1);

% first zero the data
X = X_train-ones(m,n)*diag(mean(X_train));
X_test = X_test-ones(mtest,n)*diag(mean(X_test));
% normalize by variance
sig = var(X);
for i=m
    for j=1:n
        X(i,j) = X(i,j)/sig(j);       
    end
end

sig = var(X_test);
for i=mtest
    for j=1:n
        X_test(i,j) = X_test(i,j)/sig(j);       
    end
end

% create data matrix
DATA = zeros(n,n);
for i = 1:m
    DATA = DATA + X(i,:)'*X(i,:);
end
DATA = DATA/m;

[V,D] = eig(DATA);
%%

clear Xsubspace X_subtest
plot(sort(diag(D),'descend'))

% choose to represent data in k dimenional space
k = 551;
projectionMatrix = V(:,1:k)';
for i=1:m
    Xsubspace(i,:) = projectionMatrix*(X(i,:)');
end

for i=1:size(X_test,1)
    X_subtest(i,:) = projectionMatrix*(X_test(i,:)');
end

% test lower dimensional data

cls = ClassificationDiscriminant.fit(Xsubspace,y_train,'discrimType','linear');
predictLabel = predict(cls,Xsubspace);
trainingAccuracy = sum(y_train==predictLabel)/length(y_train)

predictLabel = predict(cls,X_subtest);
testingAccuracy = sum(y_test==predictLabel)/length(y_test)

