%clear all, close all, clc

% Parameter Settings
lambda = 2.00;

% Load faces dataset
load img_faces
img_identity = img_faces.identity;
img_pose = img_faces.pose;
img_expression = img_faces.expression;
img_eye = img_faces.eye;
img = img_faces.data;

% Choose dataset
X = feat_hog;

% Compute Distance Matrix
D = dist2(X,X);

% Compute Affinity matrix
% Number of nearest neighbors in self-tuning spectral
nn = 7;
kk = floor(log2(length(img_eye)))+1;
% Case 1: dense Gaussian
A = scale_dist3(D,nn);
% Case 2: sparse Gaussian
%A = scale_dist3_knn(D,nn,kk,true);

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

% Define new kernel matrix
A_new = A-lambda/2*Y*Y';

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