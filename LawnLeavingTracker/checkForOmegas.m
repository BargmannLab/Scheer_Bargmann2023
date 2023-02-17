function omega = checkForOmegas(img)

%Based on segWorm_Elias.m in the same folder
%01/03/2019

omega = false;

% Find the worm.

cc = bwconncomp(img);
wormPixels = [];
if ~isempty(cc.PixelIdxList)
    maxCCIdx = 0;
    maxCCSize = 0;
    for i = 1:length(cc.PixelIdxList)
        ccSize = length(cc.PixelIdxList{i});
        if ccSize > maxCCSize
            maxCCSize = ccSize;
            maxCCIdx = i;
        end
    end
    wormPixels = cc.PixelIdxList{maxCCIdx};
end

% Find a point on the contour.
[y, x] = ind2sub(size(img), min(wormPixels));

% Trace the contour.
contour = bwtraceboundary(img, [y x], 'NE');

%Smooth the contour with a moving mean filter. %ELIAS ADDITION
windowsize = 5;
contour = movmean(contour,windowsize,1);

% The worm is roughly divided into 24 segments of musculature (i.e., hinges
% that represent degrees of freedom) on each side. Therefore, 48 segments
% around a 2-D contour.
% Note: "In C. elegans the 95 rhomboid-shaped body wall muscle cells are
% arranged as staggered pairs in four longitudinal bundles located in four
% quadrants. Three of these bundles (DL, DR, VR) contain 24 cells each,
% whereas VL bundle contains 23 cells." - www.wormatlas.org
sWormSegs = 24;
cWormSegs = 2 * sWormSegs;

% Clean up the worm's contour.
% contour = cleanWorm(contour, size(contour, 1) / cWormSegs);

% Compute the contour's local high/low-frequency curvature.
% Note: worm body muscles are arranged and innervated as staggered pairs.
% Therefore, 2 segments have one theoretical degree of freedom (i.e. one
% approximation of a hinge). In the head, muscles are innervated
% individually. Therefore, we sample the worm head's curvature at twice the
% frequency of its body.
% Note 2: we ignore Nyquist sampling theorem (sampling at twice the
% frequency) since the worm's cuticle constrains its mobility and practical
% degrees of freedom.

cCCLengths = circComputeChainCodeLengths(contour);
wormSegLength = (cCCLengths(1) + cCCLengths(end)) / cWormSegs;
hfAngleEdgeLength = wormSegLength;
hfCAngles = circCurvature(contour, hfAngleEdgeLength, cCCLengths);

% Blur the contour's local high-frequency curvature.
% Note: on a small scale, noise causes contour imperfections that shift an
% angle from its correct location. Therefore, blurring angles by averaging
% them with their neighbors can localize them better.
wormSegSize = size(contour, 1) / cWormSegs;
hfAngleEdgeSize = wormSegSize;
hfBlurSize = ceil(hfAngleEdgeSize / 2);
hfBlurWin(1:hfBlurSize) = 1 / hfBlurSize;
mhfCAngles = circConv(hfCAngles, hfBlurWin);

MinDist =  hfAngleEdgeLength*6; %multiply by 6 to increase the minimum distance of head to tail
% Compute the contour's local high/low-frequency curvature maxima.
[mhfCMaxP, ~] = maxPeaksCircDist(mhfCAngles, MinDist, ...
    cCCLengths);

% Are the head and tail on the outer contour?
mhfHT = mhfCMaxP > 50; 
%ELIAS CHANGED THIS THRESHOLD 11/08/2017 to 45
%ELIAS CHANGED THIS THRESHOLD 01/04/2019 to 50
mhfHTSize = sum(mhfHT);
if mhfHTSize < 2
    errNum = 105;
    errMsg = ['The worm contour has less than 2 high-frequency sampled '...
        'convexities sharper than 60 degrees (the head and tail). ' ...
        'Therefore, the worm is coiled or obscured and cannot be segmented. NOT USABLE'];
    omega = true;
    return;
end

end
