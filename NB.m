load('CS229training_data');
load('CS229testing_data');
load('rawInertialTrain');
load('rawInertialTest');

numTrainDocs = size(X_train,1);
numTestDocs = size(X_test, 1);
numFeatures = size(X_train,2);

nbclassifier = NaiveBayes.fit(X_train, y_train);
output = nbclassifier.predict(X_test);

error=0;
for i=1:numTestDocs
   if (y_test(i) ~= output(i))
     error=error+1;
   end
 end

accuracy = 1 - error/numTestDocs
