clear; clc;
load('C:\pyOpenNFT\tests\data\mainLoopData.mat')
load('C:\pyOpenNFT\tests\data\P.mat')

inpFileName = "C:\pyOpenNFT\tests\data\001_000007_000006.dcm";
templateFileName = 'C:\pyOpenNFT\tests\data\fanon-0007-00006-000006-01.nii';
niiFileName = 'C:\pyOpenNFT\tests\data\structScan_PSC.nii';

% nii1FileName = 'C:\pyOpenNFT\tests\data\first_test\6.nii';
% nii2FileName = 'C:\pyOpenNFT\tests\data\first_test\7.nii';

infoVolTempl = spm_vol(templateFileName);
tmp_imgVolTempl  = spm_read_vols(infoVolTempl);
matTemplMotCorr     = infoVolTempl.mat;
dimVol = [74, 74, 36];
R(1,1).dim = dimVol;

[slNrImg2DdimX, slNrImg2DdimY, img2DdimX, img2DdimY] = getMosaicDim(dimVol);
% tmpVol = img2Dvol3D(tmp_imgVolTempl, slNrImg2DdimX, slNrImg2DdimY, dimVol);

isZeroPadVol = 1;
if isZeroPadVol
    nrZeroPadVol = 3;
    zeroPadVol = zeros(R(1,1).dim(1),R(1,1).dim(2),nrZeroPadVol);
    R(1,1).dim(3) = R(1,1).dim (3)+nrZeroPadVol*2;
    R(1,1).Vol = cat(3, cat(3, zeroPadVol, tmp_imgVolTempl), zeroPadVol);
end

A0=[];x1=[];x2=[];x3=[];wt=[];deg=[];b=[];
R(1,1).mat = matTemplMotCorr;

% volTempl = spm_vol(nii2FileName);
% tmpVol  = spm_read_vols(volTempl);
% R(2,1).dim     = dimVol;
% tmpVol = img2Dvol3D(tmpVol, slNrImg2DdimX, slNrImg2DdimY, R(2,1).dim);

indVol = 6;

dcmData = double(dicomread(inpFileName));
tmpVol = img2Dvol3D(dcmData, slNrImg2DdimX, slNrImg2DdimY, dimVol);

% niftiwrite(tmpVol,'C:\pyOpenNFT\tests\data\dcmVol.nii')
% volTempl = spm_vol('C:\pyOpenNFT\tests\data\dcmVol.nii');
% tmpVol  = spm_read_vols(volTempl);
R(2,1).dim     = dimVol;
R(2,1).mat     = matTemplMotCorr;
% dimVol = R(2,1).dim;

if P.isZeroPadding
    zeroPadVol = zeros(R(2,1).dim(1),R(2,1).dim(2),P.nrZeroPadVol);
    R(2,1).dim(3) = R(2,1).dim (3)+nrZeroPadVol*2;
    R(2,1).Vol = cat(3, cat(3, zeroPadVol, tmpVol), zeroPadVol);
else
    R(2,1).Vol = tmpVol;
end
R(2,1).mat = matTemplMotCorr;

% r_python = load('C:\pyOpenNFT\tests\data\r_python.mat').r_python';
% 
% R(1,1).mat = r_python{1}.mat;
% R(1,1).Vol = r_python{1}.Vol;
% R(2,1).mat = r_python{2}.mat;
% R(2,1).Vol = r_python{2}.Vol;

flagsSpmRealign = struct('quality',.9,'fwhm',5,'sep',4,...
    'interp',4,'wrap',[0 0 0],'rtm',0,'PW','','lkp',1:6);
flagsSpmReslice = struct('quality',.9,'fwhm',5,'sep',4,...
    'interp',4,'wrap',[0 0 0],'mask',1,'mean',0,'which', 2);

%% realign
[R, A0, x1, x2, x3, wt, deg, b, nrIter] = ...
    spm_realign_rt(R, flagsSpmRealign, indVol,  ...
    P.nrSkipVol + 1, A0, x1, x2, x3, wt, deg, b);

%% reslice
if P.isZeroPadding
    tmp_reslVol = spm_reslice_rt(R, flagsSpmReslice);
    reslVol = tmp_reslVol(:,:,P.nrZeroPadVol+1:end-P.nrZeroPadVol);
else
    reslVol = spm_reslice_rt(R, flagsSpmReslice);
end

