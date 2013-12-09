%% GDA with testing on one subject and training on the rest

%% Train Model
load CS229training_data
load CS229testing_data

X = [X_train; X_test];
y = [y_train; y_test];
subj = [subject_train; subject_test];

for i=1:30
    Xtrain = X(subj ~= i);
    ytrain = y(subj ~= i);
    Xtest  = X(subj == i);
    ytest  = y(subj == i);
    cls = ClassificationDiscriminant.fit(Xtrain,ytrain);
    testPredictLabel = predict(cls,Xtest);
    acc = sum(ytest==testPredictLabel)/length(ytest);
    num = length(ytest);
    fprintf('Subject: %d, Data points: %d, Accuracy: %f\n', i, num, acc-1/6);
end

