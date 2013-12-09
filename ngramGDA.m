%% Train Model
%load CS229training_data
load rawInertialTrain

n = 3;   

% ngramnumtrainexamples = numexamples -n + 1 % decrease number of examples so don't have to deal with edge conditions;
% ngramnumfeatures = n * numfeatures;
% ngramXtrain = zeros(ngramnumexamples, ngramnumfeatures)
% ngramYtrain = zeros(ngramnumexamples, 1)
% 
% for startidx = n:ngramnumtrainxamples
%     % add the next n example window as features in the current example
%     windowfeatures = zeros(ngramnumfeatures)
%     for windowidx = 1:n+1
%         exampleidx = startidx + windowidx - 1;
%         example = raw_X_train(exampleidx, :)
%         windowfeatures = [windowfeatures example]
%     % get the label of the last example in the window
%     windowlabel = raw_Y_train(startidx + n)    
%     ngramYtrain(startidx) = windowlabel;
%     ngramXtrain(startidx, :) = windowfeatures;
   
[ngramXtrain, ngramYtrain] = ngramConvert(n, raw_X_train, raw_Y_train);
cls = ClassificationDiscriminant.fit(ngramXtrain,ngramYtrain,'discrimType','linear');

%% Test Model
%load CS229testing_data
load rawIntertialTest
[ngramXtest, ngramYtest] = ngramConvert(n, raw_X_test, raw_Y_test);
predictLabel =  predict(cls,ngramXtrain);
trainingAccuracy = sum(ngramYtrain==predictLabel)/length(ngramYtrain)

predictLabel =  predict(cls,ngramXtest);
testingAccuracy = sum(ngramYtest==predictLabel)/length(ngramYtest)
