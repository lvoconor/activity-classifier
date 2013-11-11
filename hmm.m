samprate = 512;
disp('Loading Data');
load('rawTrain.mat');
disp('Discretizing Data');
discreteX = round(rawTrainX * 10^9);
disp('Sampling Data');
smallX = discreteX(1:samprate:end,:);
smallY = rawTrainY(1:samprate:end,:);
clearvars -except small*;
who
size(smallX)
size(smallY)
disp('Estimating HMM');
[trans,emis] = hmmestimate(smallX, smallY);
disp('Done');
