clear; clc;

currDir  = pwd;
idcs   = strfind(currDir,'\');
dataPath = currDir(1:idcs(end-1));
dataPath = dataPath + "\data\";

reslVol = load(dataPath+"reslVol.mat").reslVol_python;
templateFileName = dataPath + "fanon-0007-00006-000006-01.nii";

infoVolTempl = niftiinfo(templateFileName);
matTemplMotCorr     = infoVolTempl.Transform.T';
dimVol = infoVolTempl.ImageSize;

dicomInfoVox   = sqrt(sum(matTemplMotCorr(1:3,1:3).^2));

gKernel = [5 5 5] ./ dicomInfoVox;
smReslVol = zeros(dimVol);
spm_smooth(reslVol, smReslVol, gKernel);