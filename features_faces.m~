% This script is used to extract different features from faces dataset

%clear all, close all, clc

%% Load faces dataset
load img_faces
img_identity = img_faces.identity;
img_pose = img_faces.pose;
img_expression = img_faces.expression;
img_eye = img_faces.eye;
img = img_faces.data;

%% Extract PCA Features
[coeff,score,latent,tsquared,explained,mu] = pca(img);
n_components = 0;
ratio = zeros(length(latent),1);
for i = 1:length(latent)
    ratio(i) = sum(latent(1:i))/sum(latent);
    if ratio(i) > 0.9
        n_components = i;
        break
    end
end
feat_pca = score(:,1:n_components);

%% Extract Gabor Features
feat_gabor = {};
gaborArray = gaborFilterBank(5,8,39,39);
for i = 1:length(img_identity)
    tmp = reshape(img(i,:),32,30)';
    featureVector = gaborFeatures(tmp,gaborArray,10,8);
    feat_gabor{i} = featureVector';
end
feat_gabor = cell2mat(feat_gabor');

% Extract HoG Features
feat_hog = {};
for i = 1:length(img_identity)
    featureVector = extractHOGFeatures(reshape(img(i,:),32,30)');
    feat_hog{i} = featureVector;
end
feat_hog = double(cell2mat(feat_hog'));

%% Extract LBP Features

