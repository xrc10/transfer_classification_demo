function [ evalObj ] = evaluate( trueLabels, outputLabels )
%EVALUATE Summary of this function goes here
%   get the multi-class performance
%   trueLabels are actual labels
%   outputLabels are system predicted labels
%   evalObj records macro(micro) precision, recall and f1

classSet = sort(unique(trueLabels), 'descend');
confTable = zeros(2,2,size(classSet,1));
precList = zeros(size(classSet,1), 1);
reclList = zeros(size(classSet,1), 1);

for i = 1:size(classSet,1)
    label = classSet(i);
    tp = sum(trueLabels(outputLabels == label) == label);
    fp = sum(trueLabels(outputLabels == label) ~= label);
    fn = sum(trueLabels(outputLabels ~= label) == label);
    tn = sum(trueLabels(outputLabels ~= label) ~= label);
    confTable(1,1,i) = tp;
    confTable(1,2,i) = fp;
    confTable(2,1,i) = fn;
    confTable(2,2,i) = tn;
    if tp > 0
        precList(i) = tp/(tp + fp);
        reclList(i) = tp/(tp + fn);
    end
end

meanConfTable = mean(confTable,3);

evalObj.trueLabels = trueLabels;
evalObj.outputLabels = outputLabels;
evalObj.confTable = confTable;
evalObj.macroPrec = mean(precList);
evalObj.macroRecl = mean(reclList);
if evalObj.macroPrec + evalObj.macroRecl > 0
    evalObj.macroF1 = 2*mean(precList)*mean(reclList)/(mean(precList) + mean(reclList));
else
    evalObj.macroF1 = 0;
end
evalObj.microPrec = meanConfTable(1,1)/(meanConfTable(1,1) + meanConfTable(1,2));
evalObj.microRecl = meanConfTable(1,1)/(meanConfTable(1,1) + meanConfTable(2,1));
evalObj.microF1 = 2*meanConfTable(1,1)/(2*meanConfTable(1,1) + meanConfTable(2,1) + meanConfTable(1,2));

end

