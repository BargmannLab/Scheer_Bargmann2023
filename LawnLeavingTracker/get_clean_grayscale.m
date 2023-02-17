function cropped_gs_detect = get_clean_grayscale( frame, region_rounded, outer_boundary_mask, BW, bboxes, fullyblurred_background )
%GET_CLEAN_GRAYSCALE.M This function takes in the frame and gets a grayscale, cleaned, blurred version of it for extracting the relative bacterial thickness in each frame.

cropped_gs = imcrop(frame,region_rounded);

% worm_mask = zeros(size(cropped_gs));
% for j = 1:size(bboxes,1) %go through all detected blobs and add them to the mask to blur out
%     bbx = [bboxes(j,1)-5 bboxes(j,2)-5 bboxes(j,3)+10 bboxes(j,4)+10];
%     topleft = [bbx(1) bbx(2)]; topright = [bbx(1)+bbx(3) bbx(2)];
%     bottomleft = [bbx(1) bbx(2)+bbx(4)]; bottomright = [bbx(1)+bbx(3) bbx(2)+bbx(4)];
%     x = [topleft(1) topright(1) bottomright(1) bottomleft(1)];
%     y = [topleft(2) topright(2) bottomright(2) bottomleft(2)];
%     curr_mask = poly2mask(x,y,size(cropped_gs,1),size(cropped_gs,2));
%     worm_mask = worm_mask+BW.*curr_mask;
% end

worm_mask = BW; %why not just use the image directly

se = strel('disk',2); %dilate the mask
to_blur_mask = imdilate(worm_mask,se);
to_blur_mask = imdilate(to_blur_mask,se);
to_blur_mask = imdilate(to_blur_mask,se);

cropped_gs_blur_filled = regionfill(cropped_gs,to_blur_mask); %blur out worms (this is a slow step)
cropped_mask = uint8(imcrop(outer_boundary_mask,region_rounded));
cropped_gs_double = im2double(cropped_gs_blur_filled.*cropped_mask);

%now normalize everything based on the background illumination -- now
%output will be double between 0 and 1 instead of uint8
invfilledbg = imcomplement(fullyblurred_background);
invfilledbg_crop = imcrop(invfilledbg.*outer_boundary_mask,region_rounded);
cropped_gs_detect = cropped_gs_double./invfilledbg_crop;%don't normalize, in case a new object appears in the video, which would offset the scales
end
