clear; clc;
load('orth_matlab.mat')
load('orth_python.mat')

% % get min/max threshold
% mn = -Inf;
% mx = Inf;
% % threshold images
% tst = max(tst, mn); imgt = min(tst, mx);
% % recompute min/max for display
% mx = -inf; mn = inf;
% if ~isempty(tst)
%     tmp = tst(isfinite(tst));
%     mx = max([mx max(max(tmp))]);
%     mn = min([mn min(min(tmp))]);
% end;
% if mx == mn, mx = mn + eps; end;
% tst = uint8(tst / max(tst(:)) * 255);
% imshow(tst)

figure;
subplot(2,2,1);
imshow(imgc);
subplot(2,2,2);
imshow(imgs);
subplot(2,2,3);
imshow(imgt);

figure;
subplot(2,2,1);
imshow(imgc_p);
subplot(2,2,2);
imshow(imgs_p);
subplot(2,2,3);
imshow(imgt_p);

figure;
subplot(2,2,1);
imshowpair(imgc, imgc_p, 'diff');
subplot(2,2,2);
imshowpair(imgs, imgs_p, 'diff');
subplot(2,2,3);
imshowpair(imgt, imgt_p, 'diff');