function [ result ] = calct( tv,im )
size1x = size(tv,1);
size1y = size(tv,2);
size2x = size(im,1);
size2y = size(im,2);
len = size(im,3);

persistent kernel
if isempty(kernel) 
    kernel = parallel.gpu.CUDAKernel('calct.ptx','calct.cu','calct');
end 

kernel.ThreadBlockSize = [1,1,1];
kernel.GridSize = [size(tv,1),size(im,2),len];

para = gpuArray(int32([size1x size1y size2x size2y len]));
out = zeros([size(tv,1),size(im,2),len],'gpuArray');
result = feval(kernel,out,real(tv), im, para);
end