function [ paramsTgt ] = transNBModelParams( paramsSrc, simM )
%TRANSMODELPARAMS Summary of this function goes here
% paramsSrc: class_num * src_word_dim matrix
% paramsTgt: class_num * tgt_word_dim matrix
% simM: src_word_dim * tgt_word_dim matrix

%% normalize simM that each row sum up to 1, commented because it is done outside of this function
% n =  sum( simM, 2 );
% n( n == 0 ) = 1;
% fprintf('normalizing similarity matrix...\n');
% simM = bsxfun( @rdivide, simM, n );
%% simple weighted summation
% fprintf('transfering NB model...\n');
paramsTgt = paramsSrc * simM;

%% one way to avoid zeros in paramsTgt
paramsTgt = paramsTgt + 1/size(paramsTgt, 2);

%% another way to avoid zeros in paramsTgt
% pseudoCount = size(paramsTgt, 2)*10;
% paramsTgt = paramsTgt * pseudoCount + 1;

%% normalize paramsTgt
n =  sum( paramsTgt, 2 );
n( n == 0 ) = 1;
% fprintf('normalizing parameter matrix...\n');
paramsTgt = bsxfun( @rdivide, paramsTgt, n );

end

