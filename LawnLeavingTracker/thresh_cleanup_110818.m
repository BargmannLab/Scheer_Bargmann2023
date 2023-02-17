function BWfinal = thresh_cleanup_110818( binImg )
%thresh_cleanup.m This function takes in a binarized image of a worm and
%performs image processing functions on it to produce a cleaned-up image.

% clean up the image a little more with erosion and dilation
    BWdfill = imfill(binImg, 'holes');
    BWnobord = imclearborder(BWdfill, 4);
    se = strel('disk',1);
    BWerode = imerode(BWnobord,se); %erode once

    BWerodeclean = bwareaopen(BWerode,100); %sometimes little specks come out of the erosion process, which you don't want to re-expand upon dilation
    BWdilate = imdilate(BWerodeclean,se);%dilate once
    BWfinal = BWdilate;
%     % now smooth the contours of this image and make a new mask
%     [boundaries,~,~,~] = bwboundaries(BWdilate); %this is a very slow step
% 
%     if size(boundaries,1)==1
%         firstBoundary = boundaries{1};
%         x = firstBoundary(:, 2);
%         y = firstBoundary(:, 1);
%         smoothX = sgolayfilt(x, polynomialOrder, windowWidth);
%         smoothY = sgolayfilt(y, polynomialOrder, windowWidth);
%         BWfinal = poly2mask(smoothX,smoothY,size(BWdilate,1), size(BWdilate,2));
%     else %usually this just means that the worm is out of the frame, will proceed as usual
%         BWfinal = BWdilate;
%     end

end