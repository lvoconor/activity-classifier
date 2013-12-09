clear()
raw_cluster_trainAccuracies = [];
raw_cluster_testAccuracies = [];
clustercount = [];
for numclusters = 1:25
    hmm_clusters_raw;
    clustercount  = [clustercount ; numclusters];
    raw_cluster_trainAccuracies = [raw_cluster_trainAccuracies;trainingAccuracy];
    raw_cluster_testAccuracies = [raw_cluster_testAccuracies;testingAccuracy];
end