function Filters = ELM_AE(data, nHidden, C, sigscale,MaxNumIter,LRate)
%ELM_AE Summary of this function goes here
%   Detailed explanation goes here
%% Run the extreme learning machine auto encoder (ELM_AE)
InputWeight=rand(nHidden,size(data,1))*2 -1;
InputWeight = gpuArray(InputWeight);
data = gpuArray(data);
fprintf(1,'AutoEncorder data %f %f\n',nHidden,size(data,1));
if nHidden > size(data,1)
    InputWeight = orth(InputWeight);
else
    InputWeight = orth(InputWeight')';
end
BiasofHiddenNeurons=rand(nHidden,1)*2 -1;
BiasofHiddenNeurons = gpuArray(BiasofHiddenNeurons);
BiasofHiddenNeurons=orth(BiasofHiddenNeurons);
tempH=InputWeight*data;                                           %   Release input of training data
ind=ones(1,size(data,2));
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;

clear BiasMatrix;
clear BiasofHiddenNeurons;
% fprintf(1,'AutoEncorder Max Val %f Min Val %f\n',max(tempH(:)),min(tempH(:)));
H = tanh(sigscale*tempH);%1 ./ (1 + exp(-sigscale*tempH));
clear tempH sigscale;                                        %   Release the temparary array for calculation of hidden neuron output matrix H
if nHidden == size(data,1)
    [~,Filterstmp,~] = procrustNew( data',H');
else
    if C == 0
        Filterstmp =pinv(H') * data';                        % implementation without regularization factor //refer to 2006 Neurocomputing paper
    else
        rhohats = gather(mean(H,2));
        rho = 0.05;                                                 %%%%%%%%%%
        KLsum = sum(rho * log(rho ./ rhohats) + (1-rho) * log((1-rho) ./ (1-rhohats)));
        
        Hsquare =  H * H';
        HsquareL = diag(max(Hsquare,[],2));
        Filterstmp=( ( eye(size(H,1)).*KLsum +HsquareL )*(1/C)+Hsquare) \ (H * data');
        
        clear Hsquare;
        clear HsquareL;
        
        % USA 
        tic;
        for epoch = 1:(MaxNumIter+1)  
            
            tempH = InputWeight * data;
            H = 2./(1+ exp(-2 * tempH))-1;
            clear tempH;
            
            KLsum = sum(rho * log(rho ./ rhohats) + (1-rho) * log((1-rho) ./ (1-rhohats)));
            Hsquare =  H * H';
            HsquareL = diag(max(Hsquare,[],2));
        
            Filterstmp=((eye(size(H,1)).*KLsum +HsquareL)/C+Hsquare) \ H * data'; 
            
            temp = H' .* (1-H');
            pinvH = pinv(H);
            dataPinvH = data * pinvH;
            Gradient_W = 2 * data * (temp .* ( pinvH * (H * data') * dataPinvH - data' * dataPinvH ));
            InputWeight = InputWeight - LRate*Gradient_W';
            
        end
        toc
    end
end
Filters = gather(Filterstmp');
end
