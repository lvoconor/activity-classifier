clear()
trainAccuracies = [];
testAccuracies = [];
clustercount = [];
for numclusters = 3:10
    hmm_clusters_raw;
    clustercount  = [clustercount ; numclusters];
    trainAccuracies = [trainAccuracies;trainingAccuracy];
    testAccuracies = [testAccuracies;testingAccuracy];
end