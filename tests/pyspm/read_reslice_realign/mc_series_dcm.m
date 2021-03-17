clear; clc;
load('C:\pyOpenNFT\tests\data\mainLoopData.mat')
load('C:\pyOpenNFT\tests\data\P.mat')

timeStamps_matlab = zeros(10,1);

templateFileName = 'C:\pyOpenNFT\tests\data\fanon-0007-00006-000006-01.nii';

% for jj=1:10
    
%     tic
    infoVolTempl = niftiinfo(templateFileName);
    tmp_imgVolTempl  = niftiread(templateFileName);
    matTemplMotCorr     = infoVolTempl.Transform.T';
    dimVol = infoVolTempl.ImageSize;
    R(1,1).dim = dimVol;
    R(1,1).mat = matTemplMotCorr;

    [slNrImg2DdimX, slNrImg2DdimY, img2DdimX, img2DdimY] = getMosaicDim(dimVol);

    isZeroPadVol = 1;
    if isZeroPadVol
        nrZeroPadVol = 3;
        zeroPadVol = zeros(R(1,1).dim(1),R(1,1).dim(2),nrZeroPadVol);
        R(1,1).dim(3) = R(1,1).dim (3)+nrZeroPadVol*2;
        R(1,1).Vol = cat(3, cat(3, zeroPadVol, tmp_imgVolTempl), zeroPadVol);
    end

    A0=[];x1=[];x2=[];x3=[];wt=[];deg=[];b=[];
    sumVols = zeros(155,74,74,36);

    % timeStamps_matlab(3) = toc;tic

    for i=1:155
        fileName = strcat(strcat('C:\pyOpenNFT\tests\data\third test\',int2str(i)),'.dcm');
        tmpVol  = dicomread(fileName);
        R(2,1).dim     = dimVol;
        tmpVol = img2Dvol3D(tmpVol, slNrImg2DdimX, slNrImg2DdimY, R(2,1).dim);
        indVol = i;

        if P.isZeroPadding
            zeroPadVol = zeros(R(2,1).dim(1),R(2,1).dim(2),P.nrZeroPadVol);
            R(2,1).dim(3) = R(2,1).dim (3)+nrZeroPadVol*2;
            R(2,1).Vol = cat(3, cat(3, zeroPadVol, tmpVol), zeroPadVol);
        else
            R(2,1).Vol = tmpVol;
        end
        R(2,1).mat = matTemplMotCorr;

        flagsSpmRealign = struct('quality',.9,'fwhm',5,'sep',4,...
            'interp',4,'wrap',[0 0 0],'rtm',0,'PW','','lkp',1:6);
        flagsSpmReslice = struct('quality',.9,'fwhm',5,'sep',4,...
            'interp',4,'wrap',[0 0 0],'mask',1,'mean',0,'which', 2);

        %% realign
        [R, A0, x1, x2, x3, wt, deg, b, nrIter] = ...
            spm_realign_rt(R, flagsSpmRealign, indVol, 1, A0, x1, x2, x3, wt, deg, b);

        tmpMCParam = spm_imatrix(R(2,1).mat / R(1,1).mat);
        if (indVol == 1)
            P.offsetMCParam = tmpMCParam(1:6);
        end
        P.motCorrParam(i,:) = tmpMCParam(1:6)-P.offsetMCParam;
        %P.motCorrParam(indVolNorm,:) = tmpMCParam(1:6);

        %% reslice
        if P.isZeroPadding
            tmp_reslVol = spm_reslice_rt(R, flagsSpmReslice);
            reslVol = tmp_reslVol(:,:,P.nrZeroPadVol+1:end-P.nrZeroPadVol);
        else
            reslVol = spm_reslice_rt(R, flagsSpmReslice);
        end

        sumVols(i,:,:,:) = reslVol;

    %     timeStamps_matlab(i+3) = toc;tic

    end
% timeStamps_matlab(jj) = toc;
% end
