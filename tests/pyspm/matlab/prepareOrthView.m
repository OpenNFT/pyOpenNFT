function prepareOrthView(mat, dim)
% set structure for Display and draw a first overlay
global strParam
strParam = struct('n', 0, 'bb',[],'Space',eye(4),'centre',[0 0 0],'mode',1,...
    'area',[0 0 1 1],'premul',eye(4),'hld',1,'modeDispl',[0, 0, 0]);
strParam.Space = spm_matrix([0 0 0  0 pi -pi/2])*strParam.Space;    

% get bounding box and resolution
if isempty(strParam.bb) 
     strParam.maxBB = maxbb(mat, dim); 
     strParam.bb = strParam.maxBB;    
end
resolution(mat);

% Draw at initial location, center of bounding box
mmcentre     = mean(strParam.Space*[strParam.maxBB';1 1],2)';
strParam.centre    = mmcentre(1:3);
% Display modes: [Background+Stat+ROIs, Background+Stat, Background+ROIs]
strParam.modeDispl = [0 0 1];

return;

function bb = maxbb(mat, dim)
global strParam
mn = [Inf Inf Inf];
mx = -mn;
premul = strParam.Space \ strParam.premul;

d = dim;
corners = [
    1    1    1    1
    1    1    d(3) 1
    1    d(2) 1    1
    1    d(2) d(3) 1
    d(1) 1    1    1
    d(1) 1    d(3) 1
    d(1) d(2) 1    1
    d(1) d(2) d(3) 1
    ]';
XYZ = mat(1:3, :) * corners;

XYZ = premul(1:3, :) * [XYZ; ones(1, size(XYZ, 2))];
bb = [
    min(XYZ, [], 2)'
    max(XYZ, [], 2)'
    ];

mx = max([bb ; mx]);
mn = min([bb ; mn]);
bb = [mn ; mx];
return;


function resolution(mat)
global strParam
resDefault = 1; % Default minimum resolution 1mm
res = min([resDefault,sqrt(sum((mat(1:3,1:3)).^2))]);
res      = res / mean(svd(strParam.Space(1:3,1:3)));
Mat      = diag([res res res 1]);
strParam.Space = strParam.Space * Mat;
strParam.bb    = strParam.bb / res;
return;