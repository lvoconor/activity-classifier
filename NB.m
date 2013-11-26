numTrainDocs = size(X_train,1);
numTestDocs = size(X_test, 1);
numFeatures = size(X_train,2);

nbclassifier = NaiveBayes.fit(X_train, y_train);
output = nbclassifier.predict(X_test);

%%
numTrainDocs = size(raw_X_train,1);
numTestDocs = size(raw_X_test, 1);
numFeatures = size(raw_X_train,2);

nbclassifier = NaiveBayes.fit(raw_X_train, raw_y_train);
output = nbclassifier.predict(raw_X_test);

error=0;
for i=1:numTestDocs
   if (y_test(i) ~= output(i))
     error=error+1;
   end
 end

%Print out the classification error on the test set
classificationerror = error/numTestDocs