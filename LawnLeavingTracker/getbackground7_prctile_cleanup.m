function [orig_background, clean_background] = getbackground7_prctile_cleanup(v, level, p, num_gaussian, pixpermm, outer_boundary_mask)
%GETBACKGROUND.M Summary of this function goes here
%   Detailed explanation goes here
number_of_frames = v.NumberOfFrames;
bg_frames = floor(linspace(1,number_of_frames,250)); %was 90
nf = length(bg_frames);

% Fill the background matrix with zeros
stack_back = zeros(v.Height, v.Width, nf);
% Stack all the frames together
for i = 1:nf
    frame = imcomplement(im2double(rgb2gray(read(v,bg_frames(i))))); % read in frame ITS DOUBLE
    stack_back(:,:,i) = frame;
end

% Percentile of the stack to obtain the background
orig_background  = prctile(stack_back,p,3);
clean_background = imgaussfilt(orig_background,num_gaussian);

%04/16/19 temporarily comment this out!
% Check if the background needs cleaning (simulate what will be done in
% tracking to find dwelling pixels)
bg_bgsub = imcomplement(orig_background.*outer_boundary_mask-clean_background.*outer_boundary_mask);
bg_bgthresh = imcomplement(im2bw(bg_bgsub,level));
thresh_cleaned = bwareaopen(bg_bgthresh,round(pixpermm/2));
cc = bwconncomp(thresh_cleaned);
blobPixels = []; %find the largest connected component in orig_background and delete it
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
    blobPixels = cc.PixelIdxList{maxCCIdx};
    tmpImg = zeros(size(thresh_cleaned));
    tmpImg(blobPixels) = 1;
    se = strel('disk',5);
    PixelsToBlur = imdilate(tmpImg,se);
    orig_background = regionfill(orig_background,PixelsToBlur);
    clean_background = imgaussfilt(orig_background,num_gaussian);
end


end

