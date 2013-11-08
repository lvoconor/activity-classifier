load('CS229training_data.mat');
load('CS229testing_data.mat');

numBins = 500;

xIndexes = HMMDiscretizer(X_test, numBins);
[transition_mat, emission_mat, bins, em_pdf] = HMMEstimator(X_train, y_train, numBins);

pstates = hmmdecode(xIndexes,transition_mat,emission_mat);