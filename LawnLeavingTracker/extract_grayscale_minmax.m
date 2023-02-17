function [min_gs,max_gs,mean_interiorLawn_gs,mean_gs_profile_aligned,mean_norm_gs_profile_aligned,LawnBoundary_AlignIdx,LBD_alignedTo_gs_profile,orig_background_afterBlur] = extract_grayscale_minmax(bg_struct,BGVI,pixpermm)
%extract_grayscale_minmax.m
%   This function takes in a bg_struct from a tracked video and extracts
%   the grayscale profile associated with it. This is useful to know so we
%   can get the minimum and maximum grayscale values associated,
%   respectively with the most and least bacterial density. These values
%   will be used to linearly rescale grayscale values into a "relative bacterial
%   density" value that can be compared across individual video recordings.
%
%   If the data comes from a normal small lawn with a dense boundary, we
%   should use the radius method to extend a transect radius from the
%   center of the lawn to the edge to identify the grayscale values along
%   the radius. The center point is the lawn center. We will set the
%   minimum grayscale value to average of the points in the dense lawn
%   boundary and the maximum grayscale value to an average of points
%   outside the lawn boundary.
%
%   If the data comes from a uniform lawn, we expect the grayscale values
%   to be uniform, so we identify minimum and maximum grayscale values as
%   an averaging of the top darkest and brightest 1000 pixels respectively.
%   Hopefully there isn't that much difference between these numbers.
%
%   If the data comes from a complex lawn such as the overlapping lawn
%   geometry of DandN or SandN, we can instead use the lateral method to
%   generate a grayscale profile in horizontal slices through the image
%   from top to bottom. We identify the extreme grayscale values inside the
%   lawn boundary by averaging pixel values close to the lawn boundary and
%   the dense boundary seam. TBD how this goes -- it might prove more
%   difficult.

lawnBoundary = bg_struct(BGVI).ev_ho;
lawnBoundaryMask = bg_struct(BGVI).lawn_limit_mask_wf;
outerBoundary = bg_struct(BGVI).outer_boundary_line;

% Subtract background illumination and blur out pixels above threshold
orig_bg = bg_struct(BGVI).orig_background;
clean_background = bg_struct(BGVI).clean_background;
outer_boundary_mask = bg_struct(BGVI).outer_boundary_mask;
level = bg_struct(BGVI).level;

bg_bgsub = imcomplement(orig_bg.*outer_boundary_mask-clean_background.*outer_boundary_mask);
bg_bgthresh = imcomplement(im2bw(bg_bgsub,level));
PixelsToBlur = imdilate(bg_bgthresh,strel('disk',5));
orig_background_afterBlur = regionfill(orig_bg,PixelsToBlur);
% clean_background_afterBlur = imgaussfilt(orig_background_afterBlur,10); %10 is normal num_gaussians
% T = adaptthresh(clean_background_afterBlur,1,'ForegroundPolarity','bright');
% curr_bg = imcomplement(clean_background_afterBlur - T); %imadjust could normalize image to same range, but better to leave this out for now, can always do more normalization later.


%get lawn center
x_cent = nanmean(lawnBoundary(:,1));
y_cent = nanmean(lawnBoundary(:,2));
lawnBoundary = [lawnBoundary;lawnBoundary(1,:)]; %add back the first element so the outline is closed.
outerBoundary = [outerBoundary;outerBoundary(1,:)];

%extend a radius from the centroid to every point on the lawnBoundary and
%beyond.
theta = linspace(0,2*pi,360);%define angle range
%get average radial distance
rad_dist = sqrt((lawnBoundary(:,1)-x_cent).^2 + (lawnBoundary(:,2)-y_cent).^2);
r = 1.25*max(rad_dist); %we use just a small distance outside the lawn to get the outside lawn gs values.
r_mm = r/pixpermm; %radial distance in mm
radius_endpoint = [r*cos(theta)'+x_cent r*sin(theta)'+y_cent];

gs_profile = NaN(size(radius_endpoint,1),1000);
interiorLawn_gs = NaN(size(radius_endpoint,1),1);
lawnBoundary_gs = NaN(size(radius_endpoint,1),1);
outsideLawn_gs = NaN(size(radius_endpoint,1),1);
lawnBoundaryIdxs = NaN(size(radius_endpoint,1),1);
boundaryWidthPix = NaN(size(radius_endpoint,1),1);
for i = 1:size(radius_endpoint,1)
%     disp(i);
%     if i==217
%         disp('debug');
%     end
    idxOutof1000 = 1:1000;
    currRad = [x_cent radius_endpoint(i,1); y_cent radius_endpoint(i,2)];
    currRad_line_x = linspace(currRad(1,1),currRad(1,2),1000);
    currRad_line_y = linspace(currRad(2,1),currRad(2,2),1000);
    boundaryCrossing = InterX(lawnBoundary',currRad)';
    if size(boundaryCrossing,1)==1
        %find closest point on the radius to the boundary crossing point (get
        %the index of this point in the line)
        distances = sqrt((currRad_line_x-boundaryCrossing(1)).^2 + (currRad_line_y-boundaryCrossing(2)).^2);
        [~,lawnBoundaryIdx] = min(distances);
        lawnBoundaryIdxs(i) = lawnBoundaryIdx;
    else
        continue;
    end
    
    outerBoundaryCrossing = InterX(outerBoundary',currRad)';
    if size(outerBoundaryCrossing,1)==1
        distances = sqrt((currRad_line_x-outerBoundaryCrossing(1)).^2 + (currRad_line_y-outerBoundaryCrossing(2)).^2);
        [~,outerBoundaryIdx] = min(distances);
        idxOutof1000 = 1:outerBoundaryIdx-2;
        currRad_line_x = currRad_line_x(idxOutof1000);
        currRad_line_y = currRad_line_y(idxOutof1000);
    end
    
    %extract grayscale values
    indToExtract = sub2ind(size(orig_background_afterBlur),round(currRad_line_y),round(currRad_line_x));
    gs_values = orig_background_afterBlur(indToExtract);
    gs_minmax = (gs_values-min(gs_values))./(max(gs_values)-min(gs_values));
    gs_minmax_smooth = movmean(gs_minmax,20);
    [Ypk,Xpk,Wpk,~] = findpeaks(gs_minmax_smooth,'MinPeakProminence',0.25, 'MaxPeakWidth',200); %could this ever be too stringent?
    if isempty(Ypk)
        continue;
    end
    [~,topPeakIdx] = max(Ypk);
    if abs(lawnBoundaryIdx - Xpk(topPeakIdx))>200 %make sure that the lawn boundary peak isn't too far away from the lawn boundary -- this can happen when the lawn gets mussed up alot.
        continue;
    end
    interiorPeakBoundaryIdx = round(Xpk(topPeakIdx)-Wpk(topPeakIdx));
    if interiorPeakBoundaryIdx<200 %if the peak is so wide that it is effectively the entire lawn, skip it
        continue;
    end
    %get the width of boundary dense region
    boundaryWidthPix(i) = Wpk(topPeakIdx);
    
%         figure();hold on;
%         plot(gs_values);
%         plot(interiorPeakBoundaryIdx:lawnBoundaryIdx,gs_values(interiorPeakBoundaryIdx:lawnBoundaryIdx));
    %     pause(); close();
    
    if lawnBoundaryIdx <= interiorPeakBoundaryIdx %sometimes this happens, don't trust these radii.
        continue;
    end
    if lawnBoundaryIdx >= size(gs_values,2) %this might happen if the outer boundary has been drawn too close to the lawn boundary
        continue;
    end
    lawnBoundaryValue = max(gs_values(interiorPeakBoundaryIdx:lawnBoundaryIdx));
    interiorLawnValue = mean(gs_values(1:interiorPeakBoundaryIdx-1));
    outsideLawnValue = mean(gs_values(lawnBoundaryIdx+1:end));
    
    %align gs_values to the lawnBoundaryIdx
    gs_profile(i,idxOutof1000) = gs_values;
    lawnBoundary_gs(i) = lawnBoundaryValue;
    interiorLawn_gs(i) = interiorLawnValue;
    outsideLawn_gs(i) = outsideLawnValue;
    
end
%now define the grayscale values at min and max as outside the lawn and
%at the lawn boundary
min_gs = nanmean(outsideLawn_gs);
max_gs = nanmean(lawnBoundary_gs);
mean_interiorLawn_gs = nanmean(interiorLawn_gs);

%     mean_lawnBoundary_width_pix = nanmean(boundaryWidthPix);
%     mean_lawnBoundary_width_mm = mean_lawnBoundary_width_pix/pixpermm;
max_lawnBoundary_width_pix = nanmax(boundaryWidthPix);
%     max_lawnBoundary_width_mm = max_lawnBoundary_width_pix/pixpermm;

%align the gs_profile to the lawnBoundaryIdx
LawnBoundary_AlignIdx = nanmax(lawnBoundaryIdxs);
LBIdiff = LawnBoundary_AlignIdx-lawnBoundaryIdxs; %value by which to shift in order to align
gs_profile_aligned = NaN(size(gs_profile));

for j = 1:size(gs_profile,1)
    if ~isnan(LBIdiff(j))
        gs_profile_aligned(j,:) = [NaN(1,LBIdiff(j)) gs_profile(j,1:end-LBIdiff(j))]; %pad with NaNs to align
    else
        gs_profile_aligned(j,:) = NaN(1,1000); %pretty sure this is already the case
    end
end
mean_gs_profile_aligned = nanmean(gs_profile_aligned,1);

%convert x axis to lawn boundary distance
r_dist_unit = r_mm/1000; %units along the line in mm
LBDpixelunits = -1*[-1*LawnBoundary_AlignIdx+1:0 1:(1000-LawnBoundary_AlignIdx)];
LBD_alignedTo_gs_profile = r_dist_unit*LBDpixelunits;

%now standardize the grayscale profile between min and max grayscale
%values. And then set any positive grayscale values outside the lawn to 0.
norm_gs_profile_aligned = gs_profile_aligned;
curr_min = prctile(reshape(norm_gs_profile_aligned.',1,[]),2); %use percentiles just in case theres any weird dust or bright pixels
curr_max = prctile(reshape(norm_gs_profile_aligned.',1,[]),98);

norm_gs_profile_aligned(norm_gs_profile_aligned<curr_min) = curr_min;
norm_gs_profile_aligned(norm_gs_profile_aligned>curr_max) = curr_max;
norm_gs_profile_aligned = (norm_gs_profile_aligned-curr_min)/(curr_max-curr_min);
norm_gs_profile_aligned(:,LawnBoundary_AlignIdx+1:end)=0; %set values outside the lawn to 0
mean_norm_gs_profile_aligned = nanmean(norm_gs_profile_aligned,1);

% % Make sure everything looks right with these plots
% figure
% hold on;
% for i = 1:size(gs_profile_aligned,1)
%     plot(LBD_alignedTo_gs_profile,gs_profile_aligned(i,:),'Color',[0.6, 0.6, 0.6, 0.3])
% end
% plot(LBD_alignedTo_gs_profile,mean_gs_profile_aligned,'b','LineWidth',2);
% % pause();
% 
% figure
% hold on;
% for i = 1:size(norm_gs_profile_aligned,1)
%     plot(LBD_alignedTo_gs_profile,norm_gs_profile_aligned(i,:),'Color',[0.6, 0.6, 0.6, 0.3])
% end
% plot(LBD_alignedTo_gs_profile,mean_norm_gs_profile_aligned,'b','LineWidth',2);
% pause();
% 
% norm_bg = orig_background_afterBlur;
% norm_bg(norm_bg>max_gs) = max_gs;
% norm_bg(norm_bg<min_gs) = min_gs;
% norm_bg = (norm_bg-min_gs)./(max_gs-min_gs);
% %set the values outside the lawn to 0
% norm_bg = norm_bg.*lawnBoundaryMask;
% 
% figure;
% imshow(norm_bg);
% pause();
% close all;


end

