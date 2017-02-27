function [ result ] = im2col_cuda( A,N1,N2 )
%IM2COL_CUDA calc im2col using cuda
%   A[x,y,channels] 
%   N1 N2 search area 
%  See also IM2COL, COUNTCOVER.
size1 = size(A,1);
size2 = size(A,2);
len = size(A,3);
persistent kernel
if isempty(kernel)
    kernel = parallel.gpu.CUDAKernel('im2col_cuda.ptx','im2col_cuda.cu','im2col_cuda');
end

numElements = N1*N2*(size1-N1+1)*(size2-N2+1);
kernel.ThreadBlockSize = [1,1,1];

para = gpuArray(int32([size1 size2 N1 N2 numElements]));
result = zeros([ N1*N2 (size1-N1+1)*(size2-N2+1) len],'gpuArray');

for i=1:4096:len
    s = 4095;
    if len-i<4096
        s = len - i;
    end
    
    kernel.GridSize = [N1*N2,(size1-N1+1)*(size2-N2+1),s+1];

    in1 = gpuArray(real(A(:,:,i:i+s)));
    out = zeros([ N1*N2 (size1-N1+1)*(size2-N2+1) s+1],'gpuArray');
    result(:,:,i:i+s) = feval(kernel,out,in1,para);
end
end

