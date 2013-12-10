clear()
raw_cluster_trainAccuracies = [];
raw_cluster_testAccuracies = [];
clustercount = [];
numclustervalues = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,30, 50, 100, 200];
for i = 1:size(numclustervalues)
    numclusters = numclustervalues(i);
    hmm_clusters_raw;
    clustercount  = [clustercount ; numclusters];
    raw_cluster_trainAccuracies = [raw_cluster_trainAccuracies;trainingAccuracy];
    raw_cluster_testAccuracies = [raw_cluster_testAccuracies;testingAccuracy];
end
