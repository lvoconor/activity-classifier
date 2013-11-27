% use all 9 features for clustering? 
load('rawInertialTrain.mat')
[numExamples, numFeatures] = size(raw_X_train);
%numclusters = 7;
[clusterindices, clustercentroids] = kmeans(raw_X_train, numclusters);

%% basic consistency check
% ratio of incorrect transitions between clusters within a label 
% numerrors = 0;
% for i = 2:numExamples
%     if clusterindices(i) ~= clusterindices(i-1) & raw_Y_train(i) == raw_Y_train(i-1)
%         numerrors = numerrors + 1;
%     end
% end
% consistency = 1 - numerrors/numExamples
%     