% This script load images from CMU faces dataset

clear all, close all, clc

base_path = '/home/changyale/dataset/faces_4/';
file_list = getAllFiles(base_path);
img_name = {};

index = 1;
for i = 1:length(file_list)
    tmp = file_list{i};
    if strcmp(tmp(length(tmp)-3:length(tmp)),'.pgm')
        img_name{index} = tmp;
        index = index+1;
    end
end

% extract identity
identity = dir(base_path);
isub = [identity(:).isdir];
identity = {identity(isub).name}';
identity(ismember(identity,{'.','..'})) = [];
% pose
pose = {'left','right','straight','up'}';
% expression
expression = {'angry','happy','neutral','sad'}';
% eye
eye = {'open','sunglasses'}';

% Groud truth about image
img_identity = zeros(length(img_name),1);
img_pose = zeros(length(img_name),1);
img_expression = zeros(length(img_name),1);
img_eye = zeros(length(img_name),1);
img = {};

% Extract ground truth from image file names
for i = 1:length(img_name)
    tmp_1 = strsplit(img_name{i},'_');
    tmp_2 = strsplit(tmp_1{length(tmp_1)-4},'/');
    % identity
    for j = 1:length(identity)
        if strcmp(identity{j},tmp_2{length(tmp_2)})
            img_identity(i) = j;
            break
        end
    end
    % pose
    for j = 1:length(pose)
        if strcmp(pose{j},tmp_1{length(tmp_1)-3})
            img_pose(i) = j;
            break
        end
    end
    % expression
    for j = 1:length(expression)
        if strcmp(expression{j},tmp_1{length(tmp_1)-2})
            img_expression(i) = j;
            break
        end
    end
    % eye
    for j = 1:length(eye)
        if strcmp(eye{j},tmp_1{length(tmp_1)-1})
            img_eye(i) = j;
            break
        end
    end
    % img data
    tmp = double(imread(img_name{i}));
    tmp_size = size(tmp);
    img{i} = reshape(tmp',1,tmp_size(1)*tmp_size(2));
end

img = cell2mat(img');

img_faces = struct('identity',img_identity,'pose',img_pose,'expression',...
    img_expression,'eye',img_eye,'data',img);

save img_faces.mat img_faces