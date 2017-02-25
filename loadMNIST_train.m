function [train_data] = loadMNIST_train(MNIST_train_labels, MNIST_test_images)


image = zeros(60000,784);

fd = fopen(MNIST_train_labels,'rb');
fseek(fd,8,-1);
label = fread(fd,60000);
fclose(fd);

fd = fopen(MNIST_test_images,'rb');
fseek(fd,16,-1);
for i=1:60000
    image(i,:) = fread(fd,784);
end
fclose(fd);

clear fd
clear i

train_data = [label,image];
clear image
clear label

