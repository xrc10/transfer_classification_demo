clc; clear all; close all

%% random seed, make experiment reproducible
rng(315);

%% load simM from extendSimMPath
load('data/ha/simM.mat', 'simM');
tmpSimM = simM;
%% load groundtruth simM from truthSimMPath
load('data/ha/goldSimM.mat', 'simM');
%% merge two simM together
tmpSimM(:,sum(simM,1)>0) = 0;
simM = simM + tmpSimM;
%% evaluate our method, we use french and hausa as a example
% eval = funRcvNBTransLearn(simM, 'french'); % to run french
eval = funBoltNBTransLearn(simM, 'hausa'); % to run hausa

%% print results
tag = {'src', 'tgt', 'trans_val', 'trans_tst'};
aggMacroF1 = [eval.avgSrcMacroF1, eval.avgTgtMacroF1, eval.avgTransMacroF1];
aggMicroF1 = [eval.avgSrcMicroF1, eval.avgTgtMicroF1, eval.avgTransMicroF1];
for j = 1:4
    
    avgMacroF1 = aggMacroF1(:,j);
    avgMicroF1 = aggMicroF1(:,j);
    
    fprintf(['-------', tag{j}, ' Overall--------\n']);
    fprintf('Test: macro F1 is %f, micro F1 is %f\n', mean(avgMacroF1), mean(avgMicroF1));
    
end




