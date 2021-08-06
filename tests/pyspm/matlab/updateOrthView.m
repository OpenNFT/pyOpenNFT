function [imgt, imgc, imgs] = updateOrthView(vol, mat)
% update orth view
global strParam;
% Calculate Background
bb   = strParam.bb;
Dims = round(diff(bb)'+1);
is   = inv(strParam.Space);
cent = is(1:3,1:3) * strParam.centre(:) + is(1:3,4);

M = strParam.Space\strParam.premul*mat;
TM0 = [ 1 0 0 -bb(1,1)+1
        0 1 0 -bb(1,2)+1
        0 0 1  -cent(3)
        0 0 0     1     ];
TM = inv(TM0 * M);
TD = Dims([1 2]);

CM0 = [ 1 0 0 -bb(1,1)+1
        0 0 1 -bb(1,3)+1
        0 1 0  -cent(2)
        0 0 0     1     ];
CM = inv(CM0 * M);
CD = Dims([1 3]);

if strParam.mode == 0
    SM0 = [ 0 0 1 -bb(1,3)+1
            0 1 0 -bb(1,2)+1
            1 0 0  -cent(1)
            0 0 0     1     ];
    SM = inv(SM0 * M);
    SD = Dims([3 2]);
else
    SM0 = [ 0 -1 0 +bb(2,2)+1
            0  0 1 -bb(1,3)+1
            1  0 0 -cent(1)
            0  0 0    1];
    SM = inv(SM0 * M);
    SD = Dims([2 3]);
end;

% Template parameters, used for ROIs and Stat map
coordParam.TM0 = TM0; coordParam.CM0 = CM0; coordParam.SM0 = SM0; 
coordParam.TD  = TD;  coordParam.CD  = CD;  coordParam.SD = SD;

M    = strParam.Space \ strParam.premul * mat;
imgt =        spm_slice_vol(vol, inv(coordParam.TM0*M), coordParam.TD, [0 NaN])';
imgc =        spm_slice_vol(vol, inv(coordParam.CM0*M), coordParam.CD, [0 NaN])';
imgs = fliplr(spm_slice_vol(vol, inv(coordParam.SM0*M), coordParam.SD, [0 NaN])');

% get min/max threshold
mn = -Inf;
mx = Inf;
% threshold images
imgt = max(imgt, mn); imgt = min(imgt, mx);
imgc = max(imgc, mn); imgc = min(imgc, mx);
imgs = max(imgs, mn); imgs = min(imgs, mx);

% recompute min/max for display
mx = -inf; mn = inf;

if ~isempty(imgt)
    tmp = imgt(isfinite(imgt));
    mx = max([mx max(max(tmp))]);
    mn = min([mn min(min(tmp))]);
end;
if ~isempty(imgc)
    tmp = imgc(isfinite(imgc));
    mx = max([mx max(max(tmp))]);
    mn = min([mn min(min(tmp))]);
end;
if ~isempty(imgs)
    tmp = imgs(isfinite(imgs));
    mx = max([mx max(max(tmp))]);
    mn = min([mn min(min(tmp))]);
end;
if mx == mn, mx = mn + eps; end;

imgt = uint8(imgt / max(imgt(:)) * 255);
imgc = uint8(double(imgc) / max(imgc(:)) * 255);
imgs = uint8(double(imgs) / max(imgs(:)) * 255);

return