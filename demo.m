clear all; close all; clc; 
addpath('./Utils');
addpath('./Liblinear/matlab');
addpath('./common');
% TrnSize = 12000; 
TrnSize = 10000; 
ImgSize = 28; 
ImgFormat = 'gray'; %'color' or 'gray'

%% Loading data from MNIST Basic (10000 training, 2000 validation, 50000 testing) 
% load('F:\Data\MNISTdata\mnist_basic');
% load('F:\Data\MNISTdata\mnist_train'); 
% load('F:\Data\MNISTdata\mnist_test'); 

% mnist_train = load('F:\Data\Data Set\mnist_rotation_new\mnist_all_rotation_normalized_float_train_valid.amat');
% mnist_test = load('F:\Data\Data Set\mnist_rotation_new\mnist_all_rotation_normalized_float_test.amat');

% mnist_train = load('F:\Data\Data Set\rectangles\rectangles_train.amat');
% mnist_test = load('F:\Data\Data Set\rectangles\rectangles_test.amat');

mnist_train = loadMNIST_train('train-labels.idx1-ubyte', 'train-images.idx3-ubyte');
mnist_test = loadMNIST_test('t10k-labels.idx1-ubyte', 't10k-images.idx3-ubyte');

%load('./MNISTdata/mnist_train');
%mnist_train = [train_X train_labels];
%clear trainX train_labels;
%load('./MNISTdata/mnist_test');
%mnist_test = [test_X test_labels];
%clear testX test_labels;
% ===== Reshuffle the training data =====
Randnidx = randperm(size(mnist_train,1)); 
mnist_train = mnist_train(Randnidx,:); 
% =======================================

TrnData = mnist_train(1:TrnSize,2:end)';  % partition the data into training set and validation set
TrnLabels = mnist_train(1:TrnSize,1);
ValData = mnist_train(TrnSize+1:end,2:end)';
ValLabels = mnist_train(TrnSize+1:end,1);
clear mnist_train;

TestData = mnist_test(:,2:end)';
TestLabels = mnist_test(:,1);
clear mnist_test;


% ==== Subsampling the Training and Testing sets ============
% (comment out the following four lines for a complete test) 
% TrnData = TrnData(:,1:4:end);  % sample around 2500 training samples
% TrnLabels = TrnLabels(1:4:end); % 
% TestData = TestData(:,1:50:end);  % sample around 1000 test samples  
% TestLabels = TestLabels(1:50:end); 
% ===========================================================

nTestImg = length(TestLabels);

rng('default');
rng(1);

%% Net parameters (they should be funed based on validation set; i.e., ValData & ValLabel)
% We use the parameters in our IEEE TPAMI submission
Net.NumStages = 2;
Net.PatchSize = 15;
Net.NumFilters = [8 8];  %[8 8]
Net.HistBlockSize = [14 14]; % if Net.HistBlockSize < 1, stands for block size is the ratio of the image size,[14 14]
Net.BlkOverLapRatio = 0.5;
Net.Poolingsize = 2;
Net.PoolingStride = 2;    % if stride == poolingsize, then no pad
Net.Pooling = 0;  % 0 stands for no pooling, 1 stands for max pooling, 2 stands for average pooling
Net.Whitten = 1;
Net.SignSqrtNorm = 0;
Net.Type = 'ELM';
Net.NormClassifier = 0;
Net.ResolutionFlag = 3; % 0 stands for standard net type; 1 stands for Laplacian Pyramid; 2 stands for multi scale version; 
%%% 3 stands for concatinates the first layer output to the last to classify.
Net.ResolutionNum = 2; % stands for how many scale used
Net.WPCA = 0; % stands for the dimensions that use wpca recuded, 0 stands for no wpca
Net.SigPara = [1 1 1;1 1 1];
Net.MaxNumIter = 1000;
Net.LRate = 0.1;

if Net.ResolutionFlag == 0
    Net.ResolutionNum = 0;
end

fprintf('\n ====== Net Parameters ======= \n')
Net

%% Net Training with 10000 samples

fprintf('\n ====== Net Training ======= \n')
TrnData_ImgArray = mat2imgarray(TrnData,ImgSize,ImgSize,ImgFormat); % convert columns in TrnData to cells 
clear TrnData; 
tic;
[ftrain, V, M, P, BlkIdx] = Net_train(TrnData_ImgArray,Net,1); % BlkIdx serves the purpose of learning block-wise DR projection matrix; e.g., WPCA
if Net.WPCA ~= 0
    block_dim = 2 ^ Net.NumFilters(2);
    DR_WPCA = cell(size(ftrain,1)/block_dim,1);
    ftrain_DR = zeros(DR_WPCA*Net.WPCA,TrnSize);
    for i = 1 : length(DR_WPCA)
        [wcoeff,~,latent,~,explained] = pca(ftrain((i-1)*block_dim+1:i*block_dim,:)','VariableWeights','variance');
        coefforth = diag(std(ingredients)) \ wcoeff;
        DR_WPCA{i,1} = coefforth(:,1:Net.WPCA);
        ftrain_DR = DR_WPCA{i,1}' * ftrain((i-1)*block_dim+1:i*block_dim,:);
    end
    ftrain = ftrain_DR;
    clear ftrain_DR;
else
    DR_WPCA = 0;
end
Net_TrnTime = toc;
clear TrnData_ImgCell; 

%% SVM
fprintf('\n ====== Training Linear SVM Classifier ======= \n')

% standardize data
ftrain = ftrain';
if Net.NormClassifier == 1
    trainXC_mean = mean(ftrain);
    % Incrementally calculate the var
    ind = ones(500,1); varftrain = zeros(1,size(ftrain,2));
    for i = 1 : 1000 : TrnSize
        if i+1000 >= TrnSize
            ftrain(i:end,:) = ftrain(i:end,:) - trainXC_mean(ones(TrnSize-i+1,1),:);
            varftrain = varftrain + sum(ftrain(i:end,:) .* ftrain(i:end,:));
        else
            ftrain(i:i+1000-1,:) = ftrain(i:i+1000-1,:) - trainXC_mean(ind,:);
            varftrain = varftrain + sum(ftrain(i:i+1000-1,:) .* ftrain(i:i+1000-1,:));
        end
    end
    varftrain = varftrain ./ (TrnSize-1);
    trainXC_sd = sqrt(varftrain+0.01);
    clear varftrain ind;
    ftrain = bsxfun(@rdivide, ftrain, trainXC_sd);
end

tic;
models = train(TrnLabels, ftrain, '-s 1 -B 1'); % we use linear SVM classifier (C = 1), calling libsvm library
LinearSVM_TrnTime = toc;
clear ftrain; 

%% Net Feature Extraction and Testing 

TestData_ImgArray = mat2imgarray(TestData,ImgSize,ImgSize,ImgFormat); % convert columns in TestData to cells 
clear TestData; 

fprintf('\n ====== Net Testing ======= \n')

nCorrRecog = zeros(nTestImg,1);
RecHistory = zeros(nTestImg,1);

tic; 
ftest = Net_FeaExt(TestData_ImgArray,V,M,P,Net); % extract a test feature using trained Net model 
for idx = 1:1:nTestImg
    if Net.WPCA ~= 0
        for i = 1 : length(DR_WPCA)
            ftest_DR = DR_WPCA{i,1}' * ftest{idx}((i-1)*block_dim+1:i*block_dim,:);
        end
        ftest{idx} = ftest_DR;
    end
    t = ftest(:,idx)';
    if Net.NormClassifier == 1
        t = bsxfun(@rdivide, bsxfun(@minus, t, trainXC_mean), trainXC_sd);
    end
    [xLabel_est, accuracy, decision_values] = predict(TestLabels(idx),...
        sparse(t), models, '-q'); % label predictoin by libsvm
   
    RecHistory(idx) = xLabel_est;
    if xLabel_est == TestLabels(idx)
        nCorrRecog(idx) = 1;
    end
   
    
end
Averaged_TimeperTest = toc/nTestImg;
Accuracy = sum(nCorrRecog)/nTestImg; 
ErRate = 1 - Accuracy;

%% Results display
fprintf('\n ===== Results of Net, followed by a linear SVM classifier =====');
fprintf('\n     Net training time: %.2f secs.', Net_TrnTime);
fprintf('\n     Linear SVM training time: %.2f secs.', LinearSVM_TrnTime);
fprintf('\n     Testing error rate: %.2f%%', 100*ErRate);
fprintf('\n     Average testing time %.2f secs per test sample. \n\n',Averaged_TimeperTest);