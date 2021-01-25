clear all;
clc;
close all;

%% Parameters

path_avions = 'entrainement/avions';

final_img = [];
nb_img = 3;
len = 8;

for i = 1:nb_img
    s = sprintf('entrainement/avions/%d.jfif',i);
    A = imread(s);
%     figure(i);
%     imshow(A);
    key_points_res = key_points(A);

%     img = imresize(A, [len len]);
%     
%     my_img = [];
%     for j = 1:len
%         my_img = [my_img, img(j, :)];
%     end
%     
%     my_img = double(my_img);
%     my_img = my_img ./ sum(my_img);
%     my_img = my_img - mean(my_img);
%     
%     final_img = [final_img; my_img];
%     figure(i+nb_img);
%     imagesc(final_img);
    
end


