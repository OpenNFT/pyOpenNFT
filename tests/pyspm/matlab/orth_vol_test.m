clear; clc;

currDir  = pwd;
idcs   = strfind(currDir,'\');
dataPath = currDir(1:idcs(end-1));
dataPath = dataPath + "\data\";

templateFileName = dataPath + "fanon-0007-00006-000006-01.nii";
strctFileName = dataPath + "structScan_PSC.nii";
infoVolTempl = niftiinfo(templateFileName);
structVolTempl = niftiinfo(strctFileName);
mat     = infoVolTempl.Transform.T';
dim     = infoVolTempl.ImageSize;
vol     = double(niftiread(templateFileName));

% mat: a 12-parameter affine transform (from sform0)
%      Note that the mapping is from voxels (where the first
%      is considered to be at [1,1,1], to millimetres
mat = mat * [eye(4,3) [-1 -1 -1 1]'];
prepareOrthView(mat, dim);

[imgt, imgc, imgs] = updateOrthView(vol, mat);

mat    = structVolTempl.Transform.T';
vol     = double(niftiread(structVolTempl));

[imgt_struct, imgc_struct, imgs_struct] = updateOrthView(vol, mat);

% figure;
% subplot(2,2,1);
% imshow(imgc);
% subplot(2,2,2);
% imshow(imgs);
% subplot(2,2,3);
% imshow(imgt);
