clear; clc;

currDir  = pwd;
idcs   = strfind(currDir,'\');
dataPath = currDir(1:idcs(end-1));
dataPath = dataPath + "\data\";

templateFileName = dataPath + "fanon-0007-00006-000006-01.nii";
infoVolTempl = niftiinfo(templateFileName);
mat     = infoVolTempl.Transform.T';
dim     = infoVolTempl.ImageSize;

load(dataPath + "mainLoopData.mat");
load(dataPath + "P.mat");
load(dataPath + "sumVols_python_dcm.mat");
load(dataPath + "mc_python_dcm.mat");
P.motCorrParam = mc_python;    


dicomInfoVox   = sqrt(sum(mat(1:3,1:3).^2));
gKernel = [5 5 5] ./ dicomInfoVox;

statMap2D_all = zeros(150,444,444);

for i=6:155

    reslVol = squeeze(sumVols(i,:,:,:));

    indVolNorm = i - 5;
    smReslVol = zeros(dim);
    spm_smooth(reslVol, smReslVol, gKernel);
    
    if P.iglmAR1
        if indVolNorm == 1
            % initalize first AR(1) volume
            mainLoopData.smReslVolAR1_1 = (1 - P.aAR1) * smReslVol;
        else
            mainLoopData.smReslVolAR1_1 = smReslVol - ...
                P.aAR1 * mainLoopData.smReslVolAR1_1;
        end
        smReslVol = mainLoopData.smReslVolAR1_1;
    end
    
    [mainLoopData, statMap2D] = prepVol_test(mainLoopData, P, smReslVol, indVolNorm);
    
    statMap2D_all(indVolNorm,:,:) = statMap2D;
    
end

% imshow(statMap2D);
