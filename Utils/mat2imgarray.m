function Img = mat2imgarray(D,height,width,ImgFormat)

N = size(D,2);
if strcmp(ImgFormat,'gray')
    Img = reshape(D,height,width,N);
elseif strcmp(ImgFormat,'color')
    error('color not support')
end


