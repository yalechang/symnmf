%clear all, close all, clc

% Parameter Settings
lambda = 1.0;
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

D = dist2(img,img);
aff_raw = scale_dist3(D,nn);
deg_raw = diag(sum(aff_raw).^(-0.5));
aff_raw_norm = deg_raw*aff_raw*deg_raw;
% PCA features
D = dist2(feat_pca,feat_pca);
aff_pca = scale_dist3(D,nn);
deg_pca = diag(sum(aff_pca).^(-0.5));
aff_pca_norm = deg_pca*aff_pca*deg_pca;
% gabor features
D = dist2(feat_gabor,feat_gabor);
aff_gabor = scale_dist3(D,nn);
deg_gabor = diag(sum(aff_gabor).^(-0.5));
aff_gabor_norm = deg_gabor*aff_gabor*deg_gabor;
% HoG features
D = dist2(feat_hog,feat_hog);
aff_hog = scale_dist3(D,nn);
deg_hog = diag(sum(aff_hog).^(-0.5));
aff_hog_norm = deg_hog*aff_hog*deg_hog;

% Compute identity assignment matrix
identity_unique = unique(img_identity);
Y = zeros(length(img_identity),length(identity_unique));
for i = 1:length(img_identity)
   Y(i,img_identity(i)) = 1;
end
aff_Y = Y*Y';
deg_Y = diag(sum(aff_Y).^(-0.5));
aff_Y_norm = deg_Y*aff_Y*deg_Y;

% Define new kernel matrix
A_new = aff_hog_norm-lambda/2*aff_Y_norm;

run_symnmf = 1;
H_list = {};
iter_list = zeros(run_symnmf,1);
obj_list = zeros(run_symnmf,1);

for i = 1:run_symnmf
    tic
    [H_list{i},iter_list(i),obj_list(i)] = symnmf_newton(A_new,4);
    toc
end
for i = 1:run_symnmf
    if obj_list(i) == min(obj_list)
        H = H_list{i};
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