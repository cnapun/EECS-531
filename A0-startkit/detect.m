close all;clear;clc;

% %% Load things
Io = imread('characters.png');
I = mat2gray(rgb2gray(Io));
load('template-h.mat');
figure; imshow(Io); title('input image');
figure; imshow(T); title('letter h');

[th, tw] = size(T);
[ih, iw] = size(I);

rh = ih-th+1;
rw = iw-tw+1;
%% process with 2norm sliding window
D = zeros(rh, rw);
for i=1:rh
    for j=1:rw
        pat = I(i:i+th-1, j:j+tw-1);
        D(i,j) = norm(pat(:)-T(:)).^2;
    end
end
D = D/max(D(:));
[ay, ax] = find(D<0.1); % indices

%% Plot
boxes = zeros(ih, iw);
for i=1:size(ay)
   boxes(ay(i)+1:ay(i)+th, ax(i)+1) = 1;
   boxes(ay(i)+1:ay(i)+th, ax(i)+tw) = 1;
   boxes(ay(i)+1, ax(i)+1:ax(i)+tw) = 1;
   boxes(ay(i)+th, ax(i)+1:ax(i)+tw) = 1;
end
t = Io;

figure; imshow(Io.*cast(boxes==0, 'uint8')); title('input image');
