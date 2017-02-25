function [test_data] = loadMNIST_test(MNIST_test_labels, MNIST_test_images)


image = zeros(10000,784);

fd = fopen(MNIST_test_labels,'rb');
fseek(fd,8,-1);
label = fread(fd,10000);
fclose(fd);

fd = fopen(MNIST_test_images,'rb');
fseek(fd,16,-1);
for i=1:10000
    image(i,:) = fread(fd,784);
end
fclose(fd);

clear fd
clear i

test_data = [label,image];
clear image
clear label

