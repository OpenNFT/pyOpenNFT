function [mainLoopData, statMap2D_pos] = prepVol_test(mainLoopData, P, smReslVol, indIglm)

    dimVol = mainLoopData.dimVol;
    img2DdimX = mainLoopData.img2DdimX;
    img2DdimY = mainLoopData.img2DdimY;
    slNrImg2DdimX = mainLoopData.slNrImg2DdimX;
    slNrImg2DdimY = mainLoopData.slNrImg2DdimY;
    statMap2D_pos = zeros(img2DdimX, img2DdimY);
    
    nrVoxInVol = mainLoopData.nrVoxInVol;
    nrBasFct = mainLoopData.nrBasFct;
    numscan = mainLoopData.numscan;
    spmMaskTh = mainLoopData.spmMaskTh;
    basFct = mainLoopData.basFct;

    if indIglm ~= 1
        
        pVal = mainLoopData.pVal;
        tContr = mainLoopData.tContr;
        nrBasFctRegr = mainLoopData.nrBasFctRegr;
        Cn = mainLoopData.Cn{indIglm-1};
        Dn = mainLoopData.Dn;
        s2n = mainLoopData.s2n;
        tn = mainLoopData.tn;
        tTh = mainLoopData.tTh;
        dyntTh = mainLoopData.dyntTh;
        statMap3D_pos = zeros(dimVol);
                   
    else
        
        pVal = mainLoopData.pVal;
        tContr = mainLoopData.tContr;

        nrHighPassRegr = size(mainLoopData.K.X0,2);
        nrMotRegr = 6;
        nrBasFctRegr = nrMotRegr+nrHighPassRegr+2;
            
        Cn = zeros(nrBasFct + nrBasFctRegr);
        Dn = zeros(nrVoxInVol, nrBasFct + nrBasFctRegr);
        s2n = zeros(nrVoxInVol, 1);
        tn.pos = zeros(nrVoxInVol, 1);
        tn.neg = zeros(nrVoxInVol, 1);
        tTh = zeros(numscan, 1);
        dyntTh = 0;
        
        statMapVect = zeros(nrVoxInVol, 1);
        statMap3D_pos = zeros(dimVol);
        statMap3D_neg = zeros(dimVol);
        mainLoopData.statMapVect = statMapVect;
        mainLoopData.statMap3D_pos = statMap3D_pos;
        mainLoopData.statMap3D_neg = statMap3D_neg;
        tempStatMap2D = zeros(img2DdimY,img2DdimX);
        mainLoopData.statMap2D = tempStatMap2D;
    end

    tmpRegr = [zscore(P.motCorrParam(1:indIglm,:)), P.linRegr(1:indIglm), ...
        mainLoopData.K.X0(1:indIglm,:), ones(indIglm,1)];
    
    % AR(1) for regressors of no interest
    if P.iglmAR1
        tmpRegr = arRegr(P.aAR1,tmpRegr);
    end

    basFctRegr = [basFct(1:indIglm,:), tmpRegr];
    % account for contrast term in contrast vector (+1)
    tContr.pos = [tContr.pos; zeros(nrBasFctRegr,1)];
    tContr.neg = [tContr.neg; zeros(nrBasFctRegr,1)];
    
    % estimate iGLM
    [idxActVoxIGLM, dyntTh, tTh, Cn, Dn, s2n, tn, neg_e2n, Bn, e2n, Fn, Nn] = ...
        iGlmVol( Cn, Dn, s2n, tn, smReslVol(:), indIglm, ...
        (nrBasFct+nrBasFctRegr), tContr, basFctRegr, pVal, ...
        dyntTh, tTh, spmMaskTh);
    
    % catch negative iGLM estimation error message for log
    mainLoopData.neg_e2n{indIglm} = neg_e2n;

    mainLoopData.basFctRegr{indIglm} = basFctRegr;
    mainLoopData.nrBasFctRegr = nrBasFctRegr;
    mainLoopData.Cn{indIglm} = Cn;
    mainLoopData.Fn{indIglm} = Fn;
    mainLoopData.Nn{indIglm} = Nn;
    mainLoopData.Dn = Dn;
    mainLoopData.s2n = s2n;
    mainLoopData.tn = tn;
    mainLoopData.tTh = tTh;
    mainLoopData.dyntTh = dyntTh;
    mainLoopData.idxActVoxIGLM.pos{indIglm} = idxActVoxIGLM.pos;
    mainLoopData.idxActVoxIGLM.neg{indIglm} = idxActVoxIGLM.neg;
    
    if ~isempty(idxActVoxIGLM.pos) && max(tn.pos) > 0 % handle empty activation map
        % and division by 0
        maskedStatMapVect_pos = tn.pos(idxActVoxIGLM.pos);
        maxTval_pos = max(maskedStatMapVect_pos);
        statMapVect = maskedStatMapVect_pos;
        statMap3D_pos(idxActVoxIGLM.pos) = statMapVect;

        statMap2D_pos = vol3Dimg2D(statMap3D_pos, slNrImg2DdimX, slNrImg2DdimY, ...
            img2DdimX, img2DdimY, dimVol) / maxTval_pos;
        statMap2D_pos = statMap2D_pos * 255;

    end

end

