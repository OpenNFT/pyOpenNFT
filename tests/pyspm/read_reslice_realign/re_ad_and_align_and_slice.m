load('C:\pyOpenNFT\tests\data\mainLoopData.mat')
load('C:\pyOpenNFT\tests\data\P.mat')

inpFileName = "C:\pyOpenNFT\tests\data\001_000007_000006.dcm";

A0=[];x1=[];x2=[];x3=[];wt=[];deg=[];b=[];
R(1,1).mat = mainLoopData.matTemplMotCorr;
R(1,1).dim = mainLoopData.dimTemplMotCorr;
R(1,1).Vol = mainLoopData.imgVolTempl;

matVol = mainLoopData.matVol;
%matTemplMotCorr = mainLoopData.matTemplMotCorr;
%dicomInfoVox = mainLoopData.dicomInfoVox;
%dimTemplMotCorr = mainLoopData.dimTemplMotCorr;
dimVol = mainLoopData.dimVol;
slNrImg2DdimX = mainLoopData.slNrImg2DdimX;
slNrImg2DdimY = mainLoopData.slNrImg2DdimY;
%img2DdimX = mainLoopData.img2DdimX;
%img2DdimY = mainLoopData.img2DdimY;

indVol = 6;

dcmData = double(dicomread(inpFileName));

% dcmData = load('C:\pyOpenNFT\tests\data\dcmData_python.mat').dcmData_python;

R(2,1).mat = matVol;
tmpVol = img2Dvol3D(dcmData, slNrImg2DdimX, slNrImg2DdimY, dimVol);

% tmpVol = load('C:\pyOpenNFT\tests\data\tmpVol_python.mat').tmpVol_python;

if P.isZeroPadding
    zeroPadVol = zeros(dimVol(1),dimVol(2),P.nrZeroPadVol);
    dimVol(3) = dimVol(3)+P.nrZeroPadVol*2;
    R(2,1).Vol = cat(3, cat(3, zeroPadVol, tmpVol), zeroPadVol);
else
    R(2,1).Vol = tmpVol;
end
R(2,1).dim = dimVol;

% R_py = load('C:\pyOpenNFT\tests\data\R_python.mat').R_python';

% R(1,1).mat = R_py{1}.mat;
% R(1,1).Vol = R_py{1}.Vol;
% R(2,1).mat = R_py{2}.mat;
% R(2,1).Vol = R_py{2}.Vol;

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
    dimVol(3) = dimVol(3) - P.nrZeroPadVol*2;
else
    reslVol = spm_reslice_rt(R, flagsSpmReslice);
end

