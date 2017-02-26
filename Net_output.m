function [OutImg, OutImgIdx] = Net_output(InImg, InImgIdx, PatchSize, NumFilters, V, M, P, Whitten, SigSqrtNorm, sigscale)
% Computing PCA filter outputs
% ======== INPUT ============
% InImg         Input images (cell structure); each cell can be either a matrix (Gray) or a 3D tensor (RGB)
% InImgIdx      Image index for InImg (column vector)
% PatchSize     Patch size (or filter size); the patch is set to be sqaure
% NumFilters    Number of filters at the stage right before the output layer
% V             Net filter banks (cell structure); V{i} for filter bank in the ith stage
% ======== OUTPUT ===========
% OutImg           filter output (cell structure)
% OutImgIdx        Image index for OutImg (column vector)



ImgZ = size(InImg,3);
mag = (PatchSize-1)/2;
OutImg = zeros(size(InImg,1),size(InImg,2),NumFilters*ImgZ);
cnt = 1;
V = gpuArray(V);
M = gpuArray(M);
P = gpuArray(P);
for i = 1:512:ImgZ
    s = 512;
    if ImgZ-i<512
        s = ImgZ - i;
    end
    
    [ImgX, ImgY, ~] = size(InImg);
    img = zeros([round(ImgX+PatchSize-1),round(ImgY+PatchSize-1), s+1],'gpuArray');
    img((mag+1):end-mag,(mag+1):end-mag,:) = InImg(:,:,i:i+s);
    im = im2col_cuda(img,PatchSize,PatchSize); % collect all the patches of the ith image in a matrix
    %     im = bsxfun(@minus, im, mean(im)); % patch-mean removal
    % normalize for contrast
    im = gpuArray(im);
    patchestmp = permute(im, [2 1 3]);
    patchestmp = bsxfun(@rdivide, bsxfun(@minus, patchestmp, mean(patchestmp,2)), sqrt(var(patchestmp,[],2)+10));
    if (Whitten == 1)
        t = bsxfun(@minus, patchestmp, M);
        for j=1:size(t,3)
            patchestmp(:,:,j) = t(:,:,j) * P;
        end
        clear t;
    end
    im = permute(patchestmp, [2 1 3]);
    clear patchestmp;
    
    for j = 1:NumFilters
        
        tv = sigscale * V(:,j)';
        for k=1:size(im,3)
            t(:,:,k) = tanh(tv*im(:,:,k));
            
            if SigSqrtNorm == 1
                t(:,:,k) = sign(t(:,:,k)) .* sqrt(abs(t(:,:,k)));
            end
        end
%         OutImg{cnt} = sigscale * reshape(V(:,j)'*im,ImgX,ImgY);  % convolution output
        %OutImg{cnt} = reshape( 1 ./ (1 + exp(-sigscale * V(:,j)'*im)),ImgX,ImgY);  % convolution output
        % Sined square root normalization

        OutImg(:,:,cnt:cnt+s) = reshape(gather(t),ImgX,ImgY,s+1);
        cnt = cnt + s +1;
        clear t;
    end
    
    %            fprintf(1,'Layered Max Val %f Min Val %f\n',max(max(OutImg{:})),min(min(OutImg(:))));
end
OutImgIdx = kron(InImgIdx,ones(NumFilters,1));