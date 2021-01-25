clear all;
clc;
close all;

final_img = [];
len = 8;
for i=1:15
    s = sprintf('C:/Users/marye/Desktop/sac de mots/images/entrainement/avions/%d.jfif',i);
    A = imread(s);
    figure(i);
    imshow(A);
    img = imresize(A, [len len]);
    
    my_img = [];
    for j = 1:len
        my_img = [my_img, img(j, :)];
    end
    
    my_img = double(my_img);
    my_img = my_img ./ sum(my_img);
    my_img = my_img - mean(my_img);
    
    final_img = [final_img; my_img];
    figure(i+15);
    imagesc(final_img);
end
