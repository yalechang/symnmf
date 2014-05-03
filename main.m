% This script implement multi-source alternate clustering
clear all, close all, clc;

% Parameter Settings
lambda = 2.00;
% Number of nearest neighbors in self-tuning spectral clustering
nn = 7;

% Load faces dataset
load img_faces
img_identity = img_faces.identity;
img_pose = img_faces.pose;
img_expression = img_faces.expression;
img_eye = img_faces.eye;
img = img_faces.data;

% Load extracted features
load feat_pca
load feat_gabor
load feat_hog

% Compute identity assignment matrix
identity_unique = unique(img_identity);
Y = zeros(length(img_identity),length(identity_unique));
for i = 1:length(img_identity)
    for j = 1:length(identity_unique)
        if img_identity(i) == identity_unique(j)
            Y(i,j) = 1;
        end
    end
end

% Compute similarity matrix from original data
% Raw data
D = dist2(img,img);
aff_raw = scale_dist3(D,nn);
% PCA features
D = dist2(feat_pca,feat_pca);
aff_pca = scale_dist3(D,nn);
% gabor features
D = dist2(feat_gabor,feat_gabor);
aff_gabor = scale_dist3(D,nn);
% HoG features
D = dist2(feat_hog,feat_hog);
aff_hog = scale_dist3(D,nn);

% Define new kernel matrix
A_new = aff_raw-lambda/2*Y*Y';

run_symnmf = 10;
H_list = {};
iter_list = zeros(run_symnmf,1);
obj_list = zeros(run_symnmf,1);

for i = 1:run_symnmf
    tic
    i
    [H_list{i},iter_list(i),obj_list(i)] = symnmf_newton(A_new,4);
    toc
end
for i = 1:run_symnmf
    if obj_list(i) == min(obj_list)
        H = H_list{i};
        i
        break
    end
end
    
label_pred = zeros(length(img_pose),1);
for i = 1:length(img_pose)
    for j = 1:length(unique(img_pose))
        if H(i,j) == max(H(i,:))
            label_pred(i) = j;
        end
    end
end

nmi_pose = nmi(label_pred,img_pose);