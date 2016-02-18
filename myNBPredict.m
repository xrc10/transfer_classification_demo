function [ gpre ] = myNBPredict( obj, test, varargin )
%MYNBPREDICT predict output for naive bayes model
%   obj is the model
%   test are instances we want to predict
if nargin < 2
    error(message('stats:NaiveBayes:predict:TooFewInputs'));
end

if ~isnumeric(test)
    error(message('stats:NaiveBayes:predict:TestBadType'));
end

if ~isreal(test)
    error(message('stats:NaiveBayes:predict:TestComplexType'));
end

if size(test,2)~= obj.NDims
    error(message('stats:NaiveBayes:BadTestSize', obj.NDims));
end

pnames = {'handlemissing'};
dflts = {'off'};
handleMissing = internal.stats.parseArgs(pnames, dflts, varargin{:});
handleMissing = internal.stats.getParamVal(handleMissing,{'on' 'off'},'HandleMissing');

if strcmp(handleMissing,'off')
    wasInvalid =  any(isnan(test),2);
else
    wasInvalid = all(isnan(test),2);
end

if isscalar(obj.Dist) && strcmp(obj.Dist,'mn')
    testBad = any(test<0 |test ~= round(test),2);
    if any(testBad)
        warning(message('stats:NaiveBayes:predict:BadDataforMN'));
        wasInvalid = wasInvalid | testBad;
    end
end

hadInvalid = any(wasInvalid);
if hadInvalid
    test(wasInvalid,:)= [];
end

log_condPdf = getlogCondPDF(obj,test,handleMissing);
gpre = getClassIdx(obj,log_condPdf);
%convert class index to the corresponding class levels

isGpreNaN= isnan(gpre);
gpre= obj.ClassLevels(gpre(~isGpreNaN),:);
if any(isGpreNaN)
    try
        gpre= dfswitchyard('statinsertnan',isGpreNaN,gpre);
    catch ME
        if ~isequal(ME.identifier,'stats:statinsertnan:InputTypeIncorrect')
            rethrow(ME);
        else
            error(message('stats:NaiveBayes:predict:logicalwithNaN'));
        end
    end
end

if hadInvalid
    try
        gpre = dfswitchyard('statinsertnan', wasInvalid, gpre);
    catch ME
        if ~isequal(ME.identifier,'stats:statinsertnan:InputTypeIncorrect')
            rethrow(ME);
        else
            error(message('stats:NaiveBayes:predict:logicalwithNaN'));
        end
    end
end

end

function   logCondPDF=getlogCondPDF(obj,test, handleNaNs)
nTest= size(test,1);

%log of conditional class density (P(x_i| theta))
%Initialize to NaNs
logCondPDF = NaN(nTest, obj.NClasses);

if  isscalar(obj.Dist) && strcmp(obj.Dist,'mn')
    %The fitted probabilities are guaranteed to be non-zero.
    logpw = log(cell2mat(obj.Params));
    %cell2mat discards empty rows corresponding to empty classes
    if strcmp(handleNaNs,'on')
        test(isnan(test)) = 0;
    end
    len = sum(test,2);
    lnCoe = gammaln(len+1) - sum(gammaln(test+1),2);
    logCondPDF(:,obj.NonEmptyClasses) = bsxfun(@plus,test * logpw', lnCoe);
    
else % 'normal', 'kernel' or 'mvmn'
    if any(obj.MVMNFS)
        mvmnfsidx = find(obj.MVMNFS);
        tempIdx = zeros(nTest,length(mvmnfsidx));
        if strcmp(handleNaNs,'on')
            for j = 1: length(mvmnfsidx)
                [~,tempIdx(:,j)]=ismember(test(:,mvmnfsidx(j)),obj.UniqVal{j});
                isNaNs = isnan(test(:,mvmnfsidx(j)));
                tempIdx(isNaNs,j) = length(obj.UniqVal{j})+1;
            end
        else % handleNaNs is 'off',
            for j = 1: length(mvmnfsidx)
                [~,tempIdx(:,j)]=ismember(test(:,mvmnfsidx(j)),obj.UniqVal{j});
            end
        end
        
        testUnseen = any(tempIdx==0,2); % rows with unseen values
        if any(testUnseen)
            %remove rows with invalid input
            warning(message('stats:NaiveBayes:BadDataforMVMN'));
            test(testUnseen,:)=[];
            tempIdx (testUnseen,:)=[];
        end
    else
        testUnseen = false(nTest,1);
    end
    
    ntestValid = size(test,1);
    
    for k = obj.NonEmptyClasses
        logPdf =zeros(ntestValid,1);
        if any(obj.GaussianFS)
            param_k=cell2mat(obj.Params(k,obj.GaussianFS));
            templogPdf = bsxfun(@plus, -0.5* (bsxfun(@rdivide,...
                bsxfun(@minus,test(:,obj.GaussianFS),param_k(1,:)),param_k(2,:))) .^2,...
                -log(param_k(2,:))) -0.5 *log(2*pi);
            if strcmp(handleNaNs,'off')
                logPdf = logPdf + sum(templogPdf,2);
            else
                logPdf = logPdf + nansum(templogPdf,2);
            end
        end%
        
        if any(obj.KernelFS)
            kdfsIdx = find(obj.KernelFS);
            for j = 1:length(kdfsIdx);
                tempLogPdf = log(obj.Params{k,kdfsIdx(j)}.pdf(test(:,kdfsIdx(j))));
                if strcmp(handleNaNs,'on')
                    tempLogPdf(isnan(tempLogPdf)) = 0;
                end
                logPdf = logPdf + tempLogPdf;
                
            end
        end
        
        if any(obj.MVMNFS)
            for j = 1: length(mvmnfsidx)
                curParams = [obj.Params{k,mvmnfsidx(j)}; 1];
                %log(1)=0;
                tempP = curParams(tempIdx(:,j));
                logPdf = logPdf + log(tempP);
            end
        end
        
        if any(testUnseen)
            % saves the log of class conditional PDF for
            % the kth class
            logCondPDF(~testUnseen,k)= logPdf;
            %set to -inf for unseen test value.
            logCondPDF(testUnseen,k)=-inf;
        else
            logCondPDF(:,k)= logPdf;
        end
        
    end %loop for k
    
end

end

function [cidx, postP, logPdf] = getClassIdx(obj,log_condPdf)

log_condPdf =bsxfun(@plus,log_condPdf, log(obj.Prior));
[maxll, cidx] = max(log_condPdf,[],2);
%set cidx to NaN if it is outlier
cidx(maxll == -inf |isnan(maxll)) = NaN;
%minus maxll to avoid underflow
if nargout >= 2
    postP = exp(bsxfun(@minus, log_condPdf, maxll));
    %density(i) is \sum_j \alpha_j P(x_i| \theta_j)/ exp(maxll(i))
    density = nansum(postP,2); %ignore the empty classes
    %normalize posteriors
    postP = bsxfun(@rdivide, postP, density);
    if nargout >= 3
        logPdf = log(density) + maxll;
    end
    
end

end %function getClassIdx

