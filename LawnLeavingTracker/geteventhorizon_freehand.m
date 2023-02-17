function [ ev_ho_x, ev_ho_y, lawn_limit_line] = geteventhorizon_freehand(background, outer_boundary_mask, lawn_limit_line, pixpermm, th)
%GETEVENTHORIZON_ROUGH.M this function takes in the background image (not
%blurred) and extracts the convex hull -- the limits of the lawn boundary,
%not taking into consideration all of the minor variations along the line.

diffThresh = (25/117.5)*pixpermm; 
blurbg = imgaussfilt(background,2);

BW = edge(blurbg,'sobel',th,'nothinning');
BWnobord = BW.*outer_boundary_mask;
BWclean = bwareaopen(BWnobord,200); %get rid of schmutz

cleanval = 200;
while sum(sum(BWclean))<1.5e4 %if the cleaning operation deletes too many pixels, decrement bwareaopen function iteratively
    cleanval = cleanval - 50;
    if cleanval<1
        BWclean = BWnobord;
        break;
    end
    BWclean = bwareaopen(BWnobord,cleanval);
end

if isempty(lawn_limit_line) %this is the first time we are doing this, so select the in-bounds region for event horizon detection
    se = strel('disk',5);
    BWclose = imclose(BWclean,se);
    BWnoholes = imfill(BWclose,'holes');
    cc = bwconncomp(BWnoholes);
    lawnPixels = [];
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
        lawnPixels = cc.PixelIdxList{maxCCIdx};
    end
    BWdetect = zeros(size(BWnoholes)); %make a mask for the largest connected component
    BWdetect(lawnPixels)=1;
    [contour_x, contour_y] = get_lawn_boundary_from_pixels(BWdetect,diffThresh);
    lawn_limit_mask = poly2mask(contour_x,contour_y,size(blurbg,1),size(blurbg,2));
    se = strel('disk',10);
    lawn_limit_mask = imdilate(lawn_limit_mask,se); %get the lawn-limit line by image dilation
    [points_y,points_x] = find(lawn_limit_mask);
    K1 = boundary(points_x,points_y);
    lawn_limit_line = [points_x(K1) points_y(K1)];
    
    figure(); imshow(background); hold on;
    plot(contour_x,contour_y,'LineWidth',2);
    plot(lawn_limit_line(:,1),lawn_limit_line(:,2),'LineWidth',2);
    good = input('LAWN BOUNDARY LOOKS GOOD? (1/0)');
    while ~(good==0 || good == 1)
        good = input('OUTER BOUNDARY LOOKS GOOD? (1/0)');
    end
    close all;
    while ~good %if not good do it the manual way
        disp('MANUAL WAY: SELECT LIMITS ON LAWN BOUNDARY...');
        clf;
        ax2 = axes();
        imshow(BWclean); hold on;
        lawn_limits = imfreehand(ax2); wait(lawn_limits);
        lawn_limit_mask = createMask(lawn_limits);
        close all;
        BWdetect = BWclean.*lawn_limit_mask;
        [contour_x, contour_y] = get_lawn_boundary_from_pixels(BWdetect,diffThresh);
        lawn_limit_mask = poly2mask(contour_x,contour_y,size(blurbg,1),size(blurbg,2));
        se = strel('disk',10);
        lawn_limit_mask = imdilate(lawn_limit_mask,se); %get the lawn-limit line by image dilation
        [points_y,points_x] = find(lawn_limit_mask);
        K1 = boundary(points_x,points_y);
        lawn_limit_line = [points_x(K1) points_y(K1)];
        figure(); imshow(background); hold on;
        plot(contour_x,contour_y,'LineWidth',2);
        plot(lawn_limit_line(:,1),lawn_limit_line(:,2),'LineWidth',2);
        good = input('LAWN BOUNDARY LOOKS GOOD? (1/0)');
        while ~(good==0 || good == 1)
            good = input('OUTER BOUNDARY LOOKS GOOD? (1/0)');
        end
    end
    close all;
    ev_ho_x = contour_x;
    ev_ho_y = contour_y;
    
else %upon successive iterations, UI is not required
    lawn_limit_mask = poly2mask(lawn_limit_line(:,1),lawn_limit_line(:,2),size(blurbg,1),size(blurbg,2));
    BWdetect = BWclean.*lawn_limit_mask;
    [ev_ho_x, ev_ho_y] = get_lawn_boundary_from_pixels(BWdetect,diffThresh);
end

end

function [contour_x, contour_y] = get_lawn_boundary_from_pixels(BW,diffThresh) %take in an image with the pixels only
[lawn_y,lawn_x] = find(BW);
K1 = boundary(lawn_x,lawn_y);
lawnBoundary = [lawn_x(K1) lawn_y(K1)];
% check that this lawn boundary doesn't have any points that are MUCH more
% concave than the convex hull
K2 = convhull(lawn_x,lawn_y);
lawnCHull = [lawn_x(K2) lawn_y(K2)];
% get centroid of pixels and of lawn_limit_line
x_cent = nanmean(lawn_x);
y_cent = nanmean(lawn_y);
%extend a radius from the centroid to every point on the lawnBoundary and
%the convex hull -- if the radius to the lawnBoundary and the radius to the
%convex hull are within a threshold distance of each other, choose the
%closer one. if not, use the convex hull.
theta = (1/180)*pi:(1/180)*pi:2*pi; %define angle range
r = 1000; %fixed radius
radius_endpoint = [r*cos(theta)'+x_cent r*sin(theta)'+y_cent];
contourPoints = NaN(size(radius_endpoint));
for i = 1:size(radius_endpoint,1)
    currRad = [x_cent radius_endpoint(i,1); y_cent radius_endpoint(i,2)];
    boundaryCrossing = InterX(lawnBoundary',currRad)';
    if size(boundaryCrossing,1)>1
        [~,furtheridx] = max(sqrt(sum(bsxfun(@minus, boundaryCrossing, [x_cent y_cent]).^2,2)));
        boundaryCrossing = boundaryCrossing(furtheridx,:);
    end
    cHullCrossing = InterX(lawnCHull',currRad)';
    if isempty(cHullCrossing) || size(cHullCrossing,1)>1 %this should never happen, but if it does, continue
        error('There should always be a single intersection point of a radius from centroid and the convex hull!');
    elseif isempty(boundaryCrossing) && isequal([1 2],size(cHullCrossing)) %if there is no boundaryCrossing perhaps there is a gap, use the cHullCrossing
        contourPoints(i,:) = cHullCrossing;
    elseif isequal([1 2],size(boundaryCrossing)) && isequal([1 2],size(cHullCrossing)) %if both intersections exist (and there should now only be one of them)
        radiiLengths = sqrt(sum(bsxfun(@minus, [boundaryCrossing;cHullCrossing], [x_cent y_cent]).^2,2)); %get the length of the radii to these points
        if abs(radiiLengths(1)-radiiLengths(2))<diffThresh % if the difference in these lengths is acceptable (based on diffThresh), use the boundaryCrossing
            contourPoints(i,:) = boundaryCrossing;
        else %otherwise use the convex hull crossing
            contourPoints(i,:) = cHullCrossing;
        end
    end
end
contourPoints(isnan(contourPoints(:,1)),:)=[]; %this should never need to happen, but get rid of NaNs just in case.

% Smooth lawn contour
smthWindow = 15;
smthContour = movmean(contourPoints,smthWindow,1,'Endpoints','discard');

% Shift the lawn boundary, and re-smooth, to get rid of gap
contourPoints_shift = circshift(contourPoints,round(size(contourPoints,1)/2),1);
smthContour_shift = movmean(contourPoints_shift,smthWindow,1,'Endpoints','discard');

% Merge these two smoothed contours to compute the centroid.
smoothContourBoth = [smthContour;smthContour_shift];

x_cent = mean(smoothContourBoth(:,1));
y_cent = mean(smoothContourBoth(:,2));
orderedContour = NaN(size(radius_endpoint));
for i = 1:size(radius_endpoint,1)
    currRad = [x_cent radius_endpoint(i,1); y_cent radius_endpoint(i,2)];
    contourCrossing = InterX(smthContour',currRad)';
    if isempty(contourCrossing) %if you are in the gap, get the intersection from the shifted smoothed version.
        contourCrossing = InterX(smthContour_shift',currRad)';
    end
    orderedContour(i,:) = contourCrossing;
end  
orderedContour(isnan(orderedContour(:,1)),:)=[];
contour_x = orderedContour(:,1);
contour_y = orderedContour(:,2);

end