function eval = funRcvNBTransLearn(simM, lang)

useClassPriorTransProb = false;

%% read data

rng(315);

kFold = 5;

fprintf('loading data...\n');

if ~exist('dataSplit', 'dir')
    mkdir('dataSplit');
end

if ~exist('NBmodels', 'dir')
    mkdir('NBmodels');
end

%% load or create src data split
if ~exist(fullfile('dataSplit', ['english_', num2str(kFold), '_tf',  '.mat']), 'file')
    srcFolder = '/clair/ruochenx/LORELEI/data/RCV1/RCV1_tf';
    [XAll, dataSplit] = kFoldRcvSplit(srcFolder, kFold);
    save(fullfile('dataSplit', ['english_', num2str(kFold), '_tf']), 'XAll', 'dataSplit');
else
    load(fullfile('dataSplit', ['english_', num2str(kFold), '_tf', '.mat']));
end

XSrcAll = XAll;
srcDataSplit = dataSplit;

%% load or create tgt data split
if ~exist(fullfile('dataSplit', [lang, '_', num2str(kFold), '_tf',  '.mat']), 'file')
    
    tgtFolder = fullfile('/clair/ruochenx/LORELEI/data/RCV2/RCV2_raw', [lang, '_tf']);
    
    [XAll, dataSplit] = kFoldRcvSplit(tgtFolder, kFold);
    
    save(fullfile('dataSplit', [lang, '_', num2str(kFold), '_tf']), 'XAll', 'dataSplit');
else
    load(fullfile('dataSplit', [lang, '_', num2str(kFold), '_tf', '.mat']));
end

XTgtAll = XAll;
tgtDataSplit = dataSplit;

pSrc = size(simM, 1);
pTgt = size(simM, 2);

%% normalize simM that each row sum up to 1
n =  sum( simM, 2 );
n( n == 0 ) = 1;
fprintf('normalizing similarity matrix...\n');
simM = bsxfun( @rdivide, simM, n );

avgSrcMacroF1 = zeros(kFold, 1);
avgSrcMicroF1 = zeros(kFold, 1);
avgTgtMacroF1 = zeros(kFold, 1);
avgTgtMicroF1 = zeros(kFold, 1);
avgTransMacroF1 = zeros(kFold, 2); % [val,tst]
avgTransMicroF1 = zeros(kFold, 2);


%% first find tgt class set
tgtClassNames = keys(tgtDataSplit);

for i = 1:kFold
    
    fprintf('Fold:%d\n', i);
    
    valFold = i;
    tstFold = mod(i, kFold)+1;
    trnFolds = setdiff((1:kFold)', [valFold; tstFold]);
    
    %%%%%%%%%%%%%%%src%%%%%%%%%%%%%%%%
    %% train on src train set
    if ~exist(fullfile('NBmodels', ['english', '_', num2str(i), '.mat']), 'file')
        srcClassNames = keys(srcDataSplit);
        models = containers.Map;
        for k = 1:size(srcClassNames,2)
            className = srcClassNames{k};
            foldData = srcDataSplit(className);
            posIdx = foldData{1};
            negIdx = foldData{2};
            [yTrn, XTrn] = getFoldData(XSrcAll, posIdx, negIdx, trnFolds);
            model = NaiveBayes.fit(XTrn, yTrn, 'Distribution', 'mn');
            models(className) = model;
        end
        save(fullfile('NBmodels', ['english', '_', num2str(i)]), 'models');
    else
        fprintf('<<Src>>Getting previous model...\n');
        load(fullfile('NBmodels', ['english', '_', num2str(i), '.mat']));
    end

    %% test on src val set and test set
    if ~exist(fullfile('NBmodels', ['english_', lang, '_eval', '.mat']), 'file')
        fprintf('<<Src>>: evaluating...\n');
        valConfTableSum = zeros(2,2);
        tstConfTableSum = zeros(2,2);
        valMacroF1 = 0;
        tstMacroF1 = 0;
        for k = 1:size(tgtClassNames,2)
            className = tgtClassNames{k};
            model = models(className);
            foldData = srcDataSplit(className);
            posIdx = foldData{1};
            negIdx = foldData{2};
            %% val set
            [yVal, XVal] = getFoldData(XSrcAll, posIdx, negIdx, valFold);
            yPred = predict(model, XVal);
            evalObj = evaluate(yVal, yPred);
            confTable = evalObj.confTable(:,:,1);
            valConfTableSum = valConfTableSum + confTable;
            valMacroF1 = valMacroF1 + (2*confTable(1,1))/(2*confTable(1,1) + confTable(2,1) + confTable(1,2));
            %% tst set
            [yTst, XTst] = getFoldData(XSrcAll, posIdx, negIdx, tstFold);
            yPred = predict(model, XTst);
            evalObj = evaluate(yTst, yPred);
            confTable = evalObj.confTable(:,:,1);
            tstConfTableSum = tstConfTableSum + confTable;
            tstMacroF1 = tstMacroF1 + (2*confTable(1,1))/(2*confTable(1,1) + confTable(2,1) + confTable(1,2));
        end
        valMacroF1 = valMacroF1/size(tgtClassNames,2);
        tstMacroF1 = tstMacroF1/size(tgtClassNames,2);
        valMicroF1 = 2*valConfTableSum(1,1)/(2*valConfTableSum(1,1) + valConfTableSum(2,1) + valConfTableSum(1,2));
        tstMicroF1 = 2*tstConfTableSum(1,1)/(2*tstConfTableSum(1,1) + tstConfTableSum(2,1) + tstConfTableSum(1,2));
        fprintf('<<Src>>Validation: macro F1 is %f, micro F1 is %f\n', valMacroF1, valMicroF1);
        
        avgSrcMacroF1(i) = tstMacroF1;
        avgSrcMicroF1(i) = tstMicroF1;
    else
        fprintf('Getting previous evaluation result...\n');
        load(fullfile('NBmodels', ['english_', lang, '_eval', '.mat']));
    end
    fprintf('<<Src>>Test: macro F1 is %f, micro F1 is %f\n', avgSrcMacroF1(i), avgSrcMicroF1(i));
    
    %%%%%%%%%%%%%%%trans%%%%%%%%%%%%%%%%
    if useClassPriorTransProb
        n =  sum( simM, 2 );
        pSrcShrk = sum(n>0);
        simMShrk = simM(n>0, :);
        srcClassNames = keys(srcDataSplit);
        models = containers.Map;
        if useClassPriorTransProb
            yTrnAllClass = [];
            idxAllClass = [];
        end
        valMacroF1 = 0;
        for k = 1:size(srcClassNames,2)
            className = srcClassNames{k};
            foldData = srcDataSplit(className);
            posIdx = foldData{1};
            negIdx = foldData{2};
            [yTrn, XTrn] = getFoldData(XSrcAll, posIdx, negIdx, trnFolds);
            XTrn = extSparseDim(XTrn, 2, pSrc);
            XTrn = XTrn(:,n>0);
            model = NaiveBayes.fit(XTrn, yTrn, 'Distribution', 'mn');
            %% test on validation set for reference
            [yVal, XVal] = getFoldData(XSrcAll, posIdx, negIdx, valFold);
            XVal = extSparseDim(XVal, 2, pSrc);
            XVal = XVal(:,n>0);
            tic;
            yPred = predict(model, XVal);
            toc;
            evalObj = evaluate(yVal, yPred);
            confTable = evalObj.confTable(:,:,1);
            valMacroF1 = valMacroF1 + (2*confTable(1,1))/(2*confTable(1,1) + confTable(2,1) + confTable(1,2));
            %% save model
            models(className) = model;
            if useClassPriorTransProb
                yTrnAllClass = [yTrnAllClass; k*ones(size(cell2mat(posIdx(trnFolds)), 1), 1)];
                idxAllClass = [idxAllClass; cell2mat(posIdx(trnFolds))];
            end
        end
        fprintf('<<Trans>>After shrink, validation: macro F1 is %f\n', valMacroF1/size(srcClassNames,2));
        XTrnAllClass = XSrcAll(idxAllClass,:);
        XTrnAllClass = extSparseDim(XTrnAllClass, 2, pSrc);
        XTrnAllClass = XTrnAllClass(:,n>0);
        modelAllClass = NaiveBayes.fit(XTrnAllClass, yTrnAllClass, 'Distribution', 'mn');
    end
    
    %% test on tgt val set and test set
    valConfTableSum = zeros(2,2);
    tstConfTableSum = zeros(2,2);
    valMacroF1 = 0;
    tstMacroF1 = 0;
    
    %     %% transfer the learned model
    if useClassPriorTransProb
        fprintf('<<Trans>>: learning trans probability...\n');
        paramsSrc = extSparseDim(cell2mat(modelAllClass.Params), 2, pSrcShrk);
        %% get all tgt documents(no label) to do EM
        %% shi's method
        XTgtAllExt = extSparseDim(XTgtAll, 2, pTgt);
        classTransProbs = infClassTransProbs(paramsSrc, modelAllClass.Prior, simMShrk, XTgtAllExt);
        srcClassNames = keys(srcDataSplit);
    end
    
    fprintf('<<Trans>>: evaluating...\n');
    for k = 1:size(tgtClassNames,2)
        className = tgtClassNames{k};
        if useClassPriorTransProb
            [~, srcClassIdx] = ismember(className, srcClassNames);
            classTransProb = classTransProbs{srcClassIdx};
        end
        model = models(className);
        foldData = tgtDataSplit(className);
        posIdx = foldData{1};
        negIdx = foldData{2};
        [yAll, ~] = getFoldData(XTgtAll, posIdx, negIdx, (1:kFold)');
        
        %% transfer the learned model
        labels = unique(yAll);
        modTgt.NClasses = model.NClasses;
        modTgt.NDims = pTgt;
        modTgt.ClassLevels = labels;
        modTgt.CIsNonEmpty = model.CIsNonEmpty;
        modTgt.Dist = model.Dist;
        modTgt.Prior = model.Prior;
        modTgt.NonEmptyClasses = (1:size(labels,1))';
        
        %% translate features
        if useClassPriorTransProb
            paramsSrc = extSparseDim(cell2mat(model.Params), 2, pSrcShrk);
            paramsTgt = transNBModelParamsClassPrior( paramsSrc, {simMShrk, classTransProb} );
        else
            paramsSrc = extSparseDim(cell2mat(model.Params), 2, pSrc);
            paramsTgt = transNBModelParams( paramsSrc, simM );
        end
        modTgt.Params = {paramsTgt};
        %% val set
        [yVal, XVal] = getFoldData(XTgtAll, posIdx, negIdx, valFold);
        XVal = extSparseDim(XVal, 2, pTgt);
        yPred = myNBPredict(modTgt, XVal);
        evalObj = evaluate(yVal, yPred);
        confTable = evalObj.confTable(:,:,1);
        valConfTableSum = valConfTableSum + confTable;
        valMacroF1 = valMacroF1 + (2*confTable(1,1))/(2*confTable(1,1) + confTable(2,1) + confTable(1,2));
        %% tst set
        [yTst, XTst] = getFoldData(XTgtAll, posIdx, negIdx, tstFold);
        XTst = extSparseDim(XTst, 2, pTgt);
        yPred = myNBPredict(modTgt, XTst);
        evalObj = evaluate(yTst, yPred);
        confTable = evalObj.confTable(:,:,1);
        tstConfTableSum = tstConfTableSum + confTable;
        tstMacroF1 = tstMacroF1 + (2*confTable(1,1))/(2*confTable(1,1) + confTable(2,1) + confTable(1,2));
    end
    valMacroF1 = valMacroF1/size(tgtClassNames,2);
    tstMacroF1 = tstMacroF1/size(tgtClassNames,2);
    valMicroF1 = 2*valConfTableSum(1,1)/(2*valConfTableSum(1,1) + valConfTableSum(2,1) + valConfTableSum(1,2));
    tstMicroF1 = 2*tstConfTableSum(1,1)/(2*tstConfTableSum(1,1) + tstConfTableSum(2,1) + tstConfTableSum(1,2));
    fprintf('<<Trans>>Validation: macro F1 is %f, micro F1 is %f\n', valMacroF1, valMicroF1);
    fprintf('<<Trans>>Test: macro F1 is %f, micro F1 is %f\n', tstMacroF1, tstMicroF1);
    
    avgTransMacroF1(i,1) = valMacroF1;
    avgTransMacroF1(i,2) = tstMacroF1;
    avgTransMicroF1(i,1) = valMicroF1;
    avgTransMicroF1(i,2) = tstMicroF1;
    
    %%%%%%%%%%%%%%%tgt%%%%%%%%%%%%%%%%
    
    %% train on tgt train set
    if ~exist(fullfile('NBmodels', [lang, '_', num2str(i), '.mat']), 'file')
        models = containers.Map;
        for k = 1:size(tgtClassNames,2)
            className = tgtClassNames{k};
            foldData = tgtDataSplit(className);
            posIdx = foldData{1};
            negIdx = foldData{2};
            [yTrn, XTrn] = getFoldData(XTgtAll, posIdx, negIdx, trnFolds);
            model = NaiveBayes.fit(XTrn, yTrn, 'Distribution', 'mn');
            models(className) = model;
        end
        save(fullfile('NBmodels', [lang, '_', num2str(i)]), 'models');
    else
        fprintf('Getting previous model...\n');
        load(fullfile('NBmodels', [lang, '_', num2str(i), '.mat']));
    end
    
    if ~exist(fullfile('NBmodels', [lang, '_monoling_eval', '.mat']), 'file')
        
        fprintf('<<Tgt>>: evaluating...\n');
        %% test on tgt val set and test set
        valConfTableSum = zeros(2,2);
        tstConfTableSum = zeros(2,2);
        valMacroF1 = 0;
        tstMacroF1 = 0;
        for k = 1:size(tgtClassNames,2)
            className = tgtClassNames{k};
            model = models(className);
            foldData = tgtDataSplit(className);
            posIdx = foldData{1};
            negIdx = foldData{2};
            %% val set
            [yVal, XVal] = getFoldData(XTgtAll, posIdx, negIdx, valFold);
            yPred = predict(model, XVal);
            evalObj = evaluate(yVal, yPred);
            confTable = evalObj.confTable(:,:,1);
            valConfTableSum = valConfTableSum + confTable;
            valMacroF1 = valMacroF1 + (2*confTable(1,1))/(2*confTable(1,1) + confTable(2,1) + confTable(1,2));
            %% tst set
            [yTst, XTst] = getFoldData(XTgtAll, posIdx, negIdx, tstFold);
            yPred = predict(model, XTst);
            evalObj = evaluate(yTst, yPred);
            confTable = evalObj.confTable(:,:,1);
            tstConfTableSum = tstConfTableSum + confTable;
            tstMacroF1 = tstMacroF1 + (2*confTable(1,1))/(2*confTable(1,1) + confTable(2,1) + confTable(1,2));
        end
        valMacroF1 = valMacroF1/size(tgtClassNames,2);
        tstMacroF1 = tstMacroF1/size(tgtClassNames,2);
        valMicroF1 = 2*valConfTableSum(1,1)/(2*valConfTableSum(1,1) + valConfTableSum(2,1) + valConfTableSum(1,2));
        tstMicroF1 = 2*tstConfTableSum(1,1)/(2*tstConfTableSum(1,1) + tstConfTableSum(2,1) + tstConfTableSum(1,2));
        fprintf('<<Tgt>>Validation: macro F1 is %f, micro F1 is %f\n', valMacroF1, valMicroF1);
        
        avgTgtMacroF1(i) = tstMacroF1;
        avgTgtMicroF1(i) = tstMicroF1;
        
    else
        fprintf('<<Tgt>>Getting previous evaluation result...\n');
        load(fullfile('NBmodels', [lang, '_monoling_eval', '.mat']));
    end
    fprintf('<<Tgt>>Test: macro F1 is %f, micro F1 is %f\n', avgTgtMacroF1(i), avgTgtMicroF1(i));
    
end

if ~exist(fullfile('NBmodels', ['english_', lang, '_eval', '.mat']), 'file')
    save(fullfile('NBmodels', ['english_', lang, '_eval']), 'avgSrcMacroF1', 'avgSrcMicroF1');
end

if ~exist(fullfile('NBmodels', [lang, '_monoling_eval', '.mat']), 'file')
    save(fullfile('NBmodels', [lang, '_monoling_eval']), 'avgTgtMacroF1', 'avgTgtMicroF1');
end

eval.avgSrcMacroF1 = mean(avgSrcMacroF1);
eval.avgTgtMacroF1 = mean(avgTgtMacroF1);
eval.avgTransMacroF1 = mean(avgTransMacroF1);
eval.avgSrcMicroF1 = mean(avgSrcMicroF1);
eval.avgTgtMicroF1 = mean(avgTgtMicroF1, 1);
eval.avgTransMicroF1 = mean(avgTransMicroF1, 1);

end

function [y, X] = getFoldData(XAll, posIdx, negIdx, folds)
X = [XAll(cell2mat(posIdx(folds)),:); XAll(cell2mat(negIdx(folds)),:)];
y = [ones(size(cell2mat(posIdx(folds)),1),1); -ones(size(cell2mat(negIdx(folds)),1),1)];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


