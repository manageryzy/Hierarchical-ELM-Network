function [f, V, M, P, BlkIdx] = Net_train(InImg,Net,IdtExt)
% =======INPUT=============
% InImg     Input images (cell); each cell can be either a matrix (Gray) or a 3D tensor (RGB)
% Net    Net parameters (struct)
%       .Net.NumStages
%           the number of stages in Net; e.g., 2
%       .PatchSize
%           the patch size (filter size) for square patches; e.g., 3, 5, 7
%           only a odd number allowed
%       .NumFilters
%           the number of filters in each stage; e.g., [16 8] means 16 and
%           8 filters in the first stage and second stage, respectively
%       .HistBlockSize
%           the size of each block for local histogram; e.g., [10 10]
%       .BlkOverLapRatio
%           overlapped block region ratio; e.g., 0 means no overlapped
%           between blocks, and 0.3 means 30% of blocksize is overlapped
% IdtExt    a number in {0,1}; 1 do feature extraction, and 0 otherwise
% =======OUTPUT============
% f         Net features (each column corresponds to feature of each image)
% V         learned Net filter banks (cell)
% BlkIdx    index of local block from which the histogram is compuated


if length(Net.NumFilters)~= Net.NumStages;
    display('Length(Net.NumFilters)~=Net.NumStages')
    return
end

NumImg = length(InImg);


OutImg = InImg;

V = cell(Net.NumStages,1); M = cell(Net.NumStages,1); P = cell(Net.NumStages,1); g = cell(Net.NumStages-1,1);

ImgIdx = (1:NumImg)';
clear InImg;

filternum = 1;
for stage = 1:Net.NumStages
    display(['Computing Net filter bank and its outputs at stage ' num2str(stage) '...'])
    
    [V{stage,1}, M{stage,1}, P{stage,1}] = Net_FilterBank(OutImg, Net.PatchSize, Net.NumFilters(stage), Net.Whitten, Net.SigPara(stage,1), Net.SigPara(stage,3),Net.MaxNumIter,Net.LRate); % compute Net filter banks
    
    if stage ~= Net.NumStages % compute the Net outputs only when it is NOT the last stage
        [OutImg, ImgIdx] = Net_output(OutImg, ImgIdx, ...
            Net.PatchSize, Net.NumFilters(stage), V{stage}, M{stage}, P{stage}, Net.Whitten, Net.SignSqrtNorm, Net.SigPara(stage,2));
        if IdtExt == 1 % enable feature extraction
             [g{stage,1}, BlkIdx] = HashingHist(Net,ImgIdx,OutImg);
        end
    end
    
    filternum = filternum * Net.NumFilters(stage);
end

if IdtExt == 1 % enable feature extraction
    display('Net training feature extraction...')
    
    f = cell(NumImg,1); % compute the Net training feature one by one
    
    OutImgIndex = ImgIdx==1;
    OutImgIndex = ones(sum(OutImgIndex),1);
    
    i = 1;
    work = zeros(ceil(NumImg/512),2);
    data = cell(ceil(NumImg/512),1);
    for idx = 1:512:NumImg
        s = 511;
        if NumImg - idx < 512
            s = NumImg - idx;
        end
        work(i,1) = idx;
        work(i,2) = s;
        data{i,1} = OutImg(:,:,(idx-1)*filternum/Net.NumFilters(end)+1:(idx+s)*filternum/Net.NumFilters(end));
        i=i+1;
    end
    clear OutImg;
    
    PatchSize = Net.PatchSize;
    NumFilters = Net.NumFilters(end);
    Whitten = Net.Whitten;
    SignSqrtNorm = Net.SignSqrtNorm;
    SigPara = Net.SigPara(Net.NumStages,2);
    Vend = V{end};
    Mend = M{end};
    Pend = P{end};
    for idx = 1:size(work,1)
        [OutImg_i, ~] = Net_output(data{idx}, OutImgIndex,...
            PatchSize, NumFilters, Vend, Mend, Pend, Whitten, SignSqrtNorm, SigPara);  % compute the last Net outputs of image "idx"
        
        for i = 0:work(idx,2)
            [f{work(idx,1)+i}, ~] = HashingHist(Net,ones(filternum,1),OutImg_i(:,:,filternum*(i)+1:filternum*(i+1))); 
        end
    end
    
    
    
    for idx = 1:NumImg
        if 0==mod(idx,1000); display(['Extracting Net feasture of the ' num2str(idx) 'th training sample...']); end
        
        if Net.NumStages ~= 1
            f{idx} = [f{idx} ; g{1,1}(:,idx)];
        end
    end
    f = [f{:}];
    
end
end