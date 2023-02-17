function [skel, pt1, pt2, omega] = segWormOnly_Elias2(img, x_offset, y_offset, pixpermm)
skel = NaN; %defaults
pt1 = [NaN NaN];
pt2 = [NaN NaN];
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

% No worm found.
if isempty(wormPixels)
    error('No worm was found.');
end

% Find a point on the contour.
[y, x] = ind2sub(size(img), min(wormPixels));

% Trace the contour clockwise.
contour = bwClockTrace(img, [x y], true);

% The worm is roughly divided into 24 segments of musculature (i.e., hinges
% that represent degrees of freedom) on each side. Therefore, 48 segments
% around a 2-D contour.
% Note: "In C. elegans the 95 rhomboid-shaped body wall muscle cells are
% arranged as staggered pairs in four longitudinal bundles located in four
% quadrants. Three of these bundles (DL, DR, VR) contain 24 cells each,
% whereas VL bundle contains 23 cells." - www.wormatlas.org
sWormSegs = 24;
cWormSegs = 2 * sWormSegs;

%Smooth the contour with a moving mean filter. %ELIAS ADDITION
windowsize = 5;
smthContour = movmean(contour,windowsize,1,'Endpoints','discard');

%Now shift this contour around to avoid the discontinuity at the ends
Contour_shift = circshift(contour,round(size(contour,1)/2),1);
smthContour_shift = movmean(Contour_shift,windowsize,1,'Endpoints','discard');

missingContour = setdiff(smthContour_shift,smthContour,'rows');
newContour = [smthContour ; zeros(size(missingContour))];
contourEndpoint = smthContour(end,:);
newidx = size(smthContour,1)+1;
while size(missingContour,1)>0
    [~,closeridx] = min(sqrt(sum(bsxfun(@minus, missingContour, contourEndpoint).^2,2))); %find the closest point in the missing contour to add
    newEndpoint = missingContour(closeridx,:);
    newContour(newidx,:) = newEndpoint;
    contourEndpoint = newEndpoint;
    missingContour(closeridx,:)=[];
    newidx = newidx+1;
end

%now make the new img from the smoothed newContour, this helps with
%morphological method.
img = poly2mask(newContour(:,2),newContour(:,1),size(img,1),size(img,2));


% Compute the contour's local high/low-frequency curvature.
% Note: worm body muscles are arranged and innervated as staggered pairs.
% Therefore, 2 segments have one theoretical degree of freedom (i.e. one
% approximation of a hinge). In the head, muscles are innervated
% individually. Therefore, we sample the worm head's curvature at twice the
% frequency of its body.
% Note 2: we ignore Nyquist sampling theorem (sampling at twice the
% frequency) since the worm's cuticle constrains its mobility and practical
% degrees of freedom.
cCCLengths = circComputeChainCodeLengths(newContour);
wormSegLength = (cCCLengths(1) + cCCLengths(end)) / cWormSegs;
hfAngleEdgeLength = wormSegLength;
hfCAngles = circCurvature(newContour, hfAngleEdgeLength, cCCLengths);
lfAngleEdgeLength = 2 * hfAngleEdgeLength;
lfCAngles = circCurvature(newContour, lfAngleEdgeLength, cCCLengths);

% Blur the contour's local high-frequency curvature.
% Note: on a small scale, noise causes contour imperfections that shift an
% angle from its correct location. Therefore, blurring angles by averaging
% them with their neighbors can localize them better.
wormSegSize = size(newContour, 1) / cWormSegs;
hfAngleEdgeSize = wormSegSize;
hfBlurSize = ceil(hfAngleEdgeSize / 2);
hfBlurWin(1:hfBlurSize) = 1 / hfBlurSize;
mhfCAngles = circConv(hfCAngles, hfBlurWin);

% Compute the contour's local high/low-frequency curvature maxima.
[mhfCMaxP, mhfCMaxI] = maxPeaksCircDist(mhfCAngles, hfAngleEdgeLength, cCCLengths);
[lfCMaxP, lfCMaxI] = maxPeaksCircDist(lfCAngles, lfAngleEdgeLength, cCCLengths);

% Check if the worm is doing an omega; this is an unsegmentable posture
% Are the head and tail on the outer contour?
mhfHT = mhfCMaxP > 50;
%ELIAS CHANGED THIS THRESHOLD 11/08/2017 to 45
%ELIAS CHANGED THIS THRESHOLD 01/04/2019 to 50
mhfHTSize = sum(mhfHT);
if mhfHTSize < 2
    disp('Omega! Not segmented.');
    omega = true;
    return;
end

% The head and tail are on the outer contour.
% The low-frequency sampling identified the head and tail.

lfHT = lfCMaxP > 90;
lfHTSize = sum(lfHT);

% Are there too many possible head/tail points? Use morphological methods
% (Sagi method) to segment the animal instead
if lfHTSize > 2 %errNum 104 (The worm has 3 or more low-frequency sample convexities sharper than 90 degrees)
    MorphMethod();
    return;
end

if lfHTSize > 1
    
    % Find the head and tail convexities in the low-frequency sampling.
    % Note: the tail should have a sharper angle.
    lfHTI = lfCMaxI(lfHT);
    lfHTP = lfCMaxP(lfHT);
    if lfHTP(1) <= lfHTP(2)
        headI = lfHTI(1);
        tailI = lfHTI(2);
    else
        headI = lfHTI(2);
        tailI = lfHTI(1);
    end
    
    % Localize the head by finding its nearest, sharpest (but blurred),
    % high-frequency convexity.
    mhfHTI = mhfCMaxI(mhfHT);
    dhfHeadI = abs(cCCLengths(headI) - cCCLengths(mhfHTI));
    dhfHeadI = min(dhfHeadI, cCCLengths(end) - dhfHeadI);
    [~, hfHeadI] = min(dhfHeadI);
    headI = mhfHTI(hfHeadI);
    
    % Localize the tail by finding its nearest, sharpest (but blurred),
    % high-frequency convexity.
    dhfTailI = abs(cCCLengths(tailI) - cCCLengths(mhfHTI));
    dhfTailI = min(dhfTailI, cCCLengths(end) - dhfTailI);
    [~, hfTailI] = min(dhfTailI);
    tailI = mhfHTI(hfTailI);
    
    % The high-frequency sampling identifies the head and tail.
elseif mhfHTSize < 3
    
    % Find the head and tail convexities in the high-frequency sampling.
    % Note: the tail should have a sharper angle.
    mhfHTI = mhfCMaxI(mhfHT);
    mhfHTP = mhfCMaxP(mhfHT);
    if mhfHTP(1) <= mhfHTP(2)
        headI = mhfHTI(1);
        tailI = mhfHTI(2);
    else
        headI = mhfHTI(2);
        tailI = mhfHTI(1);
    end
    % The high-frequency sampling identifies several, potential heads/tails.
else
    % Initialize our head and tail choicse.
    mhfHTI = mhfCMaxI(mhfHT);
    mhfHTI1 = mhfHTI(1);
    mhfHTI2 = mhfHTI(2);
    
    % How far apart are the head and tail?
    dmhfHTI12 = abs(cCCLengths(mhfHTI(1)) - cCCLengths(mhfHTI(2)));
    dmhfHTI12 = min(dmhfHTI12, cCCLengths(end) - dmhfHTI12);
    
    % Search for the 2 sharp convexities that are furthest apart.
    for i = 1:(mhfHTSize - 1)
        for j = (i + 1):mhfHTSize
            
            % How far apart are these 2 convexities?
            dmhfHTIij = abs(cCCLengths(mhfHTI(i)) - ...
                cCCLengths(mhfHTI(j)));
            dmhfHTIij = min(dmhfHTIij, cCCLengths(end) - dmhfHTIij);
            
            % These 2 convexities are better head and tail choices.
            if dmhfHTIij > dmhfHTI12
                mhfHTI1 = mhfHTI(i);
                mhfHTI2 = mhfHTI(j);
                dmhfHTI12 = dmhfHTIij;
            end
        end
    end
    
    % Which convexity is the head and which is the tail?
    % Note: the tail should have a sharper angle.
    if mhfCAngles(mhfHTI1) < mhfCAngles(mhfHTI2)
        headI = mhfHTI1;
        tailI = mhfHTI2;
    else
        headI = mhfHTI2;
        tailI = mhfHTI1;
    end
end

% Orient the contour and angles at the maximum curvature (the head or tail).
if headI > 1
    newContour = [newContour(headI:end,:); newContour(1:(headI - 1),:)];
    cCCLengths = [cCCLengths(headI:end) - cCCLengths(headI - 1); ...
        cCCLengths(1:(headI - 1)) + ...
        (cCCLengths(end) - cCCLengths(headI - 1))];
    %hfCAngles = [hfCAngles(headI:end); hfCAngles(1:(headI - 1))];
    lfCAngles = [lfCAngles(headI:end); lfCAngles(1:(headI - 1))];
    lfCMaxI = lfCMaxI - headI + 1;
    wrap = lfCMaxI < 1;
    lfCMaxI(wrap) = lfCMaxI(wrap) + length(lfCAngles);
    tailI = tailI - headI + 1;
    headI = 1;
    if tailI < 1
        tailI = tailI + size(newContour, 1);
    end
end


% Compute the contour's local low-frequency curvature minima.
[lfCMinP, lfCMinI] = minPeaksCircDist(lfCAngles, lfAngleEdgeLength,cCCLengths);

% Compute the worm's skeleton.
try
    [skeleton , ~] = linearSkeleton(headI, tailI, lfCMinP, lfCMinI, ...
        lfCMaxP, lfCMaxI, newContour, wormSegLength, cCCLengths);
catch
    %Ev Method failed for some reason, go back to Morph Method
    MorphMethod();
    return;
end

skel = [skeleton(:,2)+x_offset skeleton(:,1)+y_offset]; %flip x and y to go from image to plotting
pt1 = skel(1,:);
pt2 = skel(end,:);
%check that the skeleton is long enough (if not, it may be an omega that
%missed detection by the other methods)
[arclen,~] = arclength(skel(:,1),skel(:,2));
tooShortScaleFactor = 75/112;
if arclen<tooShortScaleFactor*pixpermm
    disp('ArcLength is too short! Omega!');
    skel = NaN;
    pt1 = [NaN NaN];
    pt2 = [NaN NaN];
    omega = true;
end

    function MorphMethod()
        disp('Use Morphological Method.');
        % from Sagi's FragmentTracker_SL_PlateFluorescence_v03
        %%% Find properties...
        BW_thin             = bwmorph(img,'thin',inf);
        BW_endpoints        = bwmorph(BW_thin,'endpoints',inf);
        % End points
        [I_all ,~]      = find(BW_endpoints==1);
        NumOfEndPoints      = length(I_all);
        if NumOfEndPoints == 0       % e.g. for a circular worm object
            return;
        end
        
        branchpoints = bwmorph(BW_thin,'branchpoints');
        [I_all, J_all] = find(branchpoints==1);
        NumBranchPoints = length(I_all);
        % Correct Skeleton if more than 2 endpoints were found: (this method
        % chooses the line segment which results in less curvature
        % discontinuity) METHOD 1 Elias
        BW_pruned = BW_thin;
        totalNumPixelsBeforePruning = sum(sum(BW_thin));
        counter = 0;
        while NumBranchPoints>0
            counter = counter+1;
            bpoint = [I_all(1) J_all(1)];
            %remove branchpoint and all of its 4-connected neighbors to reveal
            %line segments emanating from it.
            north = [bpoint(1)-1 bpoint(2)]; south = [bpoint(1)+1 bpoint(2)];
            east = [bpoint(1) bpoint(2)+1]; west = [bpoint(1) bpoint(2)-1];
            pixelstoremove = [bpoint;north;south;east;west];
            BW_pruned(sub2ind(size(img),pixelstoremove(:,1),pixelstoremove(:,2)))=0; %remove this branchpoint to reveal line segments emanating from it.
            segments = bwconncomp(BW_pruned);
            arclengths = zeros(segments.NumObjects,1);
            for k = 1:segments.NumObjects
                arclengths(k) = length(segments.PixelIdxList{k});
            end
            [~,longestSegIdx] = max(arclengths); %get the longest segment
            [longestSegPixelsI,longestSegPixelsJ] = ind2sub(size(img),segments.PixelIdxList{longestSegIdx});
            %check which concatenated spline results in the least curvature
            %discontinuity
            otherSegs = setdiff(1:segments.NumObjects,longestSegIdx);
            maxProm = Inf;
            for k = 1:length(otherSegs)
                otherSegPixels = segments.PixelIdxList{otherSegs(k)};
                [otherSegI,otherSegJ] = ind2sub(size(img),otherSegPixels);
                allPixelstoTest = [longestSegPixelsI longestSegPixelsJ; bpoint; otherSegI otherSegJ]; %concatenate
                tmpImg = zeros(size(img));
                tmpImg(sub2ind(size(img),allPixelstoTest(:,1),allPixelstoTest(:,2)))=1; %set these pixels to 1
                tmpImg = bwmorph(tmpImg,'bridge');
                tmpImg = bwmorph(tmpImg,'thin',inf);
                %to trace boundary of this skeleton in order, get endpoints first
                tmpEndpoints = bwmorph(tmpImg,'endpoints');
                endpts = find(tmpEndpoints==1);
                [eI, eJ] = ind2sub(size(img),endpts);
                sortedConcatSegs = bwtraceboundary(tmpImg, [eI(1) eJ(1)], 'NE'); %get these pixels in order
                Vertices = sortedConcatSegs(1:ceil((end+1)/2),:);
                if length(Vertices)< 0.6*totalNumPixelsBeforePruning %if the current spline greatly reduces the length of spline, it's wrong
                    continue;
                end
                BPindex = dsearchn(Vertices,bpoint); %the index of the closest point in spline to the branchpoint
                smthVerts = [movmean(Vertices(:,1),3) movmean(Vertices(:,2),3)];
                curv = CurvatureWindow_Elias(smthVerts, hfAngleEdgeLength*2.5);
                [~,locs,~,proms] = findpeaks(abs(curv));
                peakdist = sqrt((locs-BPindex).^2);
                [closestpeakdist,closestpeakidx] = min(peakdist);
                prom=0;
                if closestpeakdist<5
                    prom = proms(closestpeakidx);
                end
                if prom<maxProm
                    maxProm = prom;
                    BW_pruned = tmpImg;
                end
            end
            %delete the line segment that resulted in more discontinuous
            %curvature, then check for more branchpoints
            branchpoints = bwmorph(BW_pruned,'branchpoints');
            [I_all, J_all] = find(branchpoints==1);
            NumBranchPoints = length(I_all);
            if counter>100
                disp(['Stopping while loop for skeleton correction', char(10)]);
                break;
            end
        end
        cc = bwconncomp(BW_pruned);
        if cc.NumObjects>1 %sometimes this procedure ends up with disconnected skeletons, this is unsegmentable
            return;
        end
        if sum(sum(BW_pruned))< 0.6*totalNumPixelsBeforePruning %as before, if the segmentation is too short, discard.
            return;
        end
        BW_thin = BW_pruned;
        
        % from Hums 2016 code in WormAngles\steps\FindAllWormSkeletonsFromTracks.m
        Endpoints = bwmorph(BW_thin, 'endpoints');
        [EndpontIndcsX, EndpontIndcsY] = find(Endpoints==1);
        if length(EndpontIndcsX)>2 %if you still have a branchpoint after doing the routine
            return;
        end
        if EndpontIndcsX
            wrm = bwtraceboundary(BW_thin, [EndpontIndcsX(1) EndpontIndcsY(1)], 'NE');
            %partition into subsegments
            if (wrm)
                wrmtr=wrm(1:ceil((end+1)/2),:);
                smoothwormX = wrmtr(:,2)';
                smoothwormY = wrmtr(:,1)';
                if numel(smoothwormY) > 45 %commands below will cause a crash when skeleton shrinks too much
                    skel = [smoothwormX+x_offset; smoothwormY+y_offset]'; %check that you're adding the right one to each.
                    pt1 = round(skel(1,:));
                    pt2 = round(skel(end,:));
                    return;
                else
                    disp('the resulting skeleton is too short!');
                    return;
                end
            else
                disp('could not trace the worm!');
                return;
            end
        else
            disp('the resulting image has no endpoints!');
            return;
        end
        
    end

end


