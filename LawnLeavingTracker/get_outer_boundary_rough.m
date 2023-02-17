function [ outer_boundary_mask, outer_boundary_line, region_rounded ] = get_outer_boundary_rough( frame, rad_to_subtract, numgaussblur )
%GET_OUTER_BOUNDARY.M This function estimates the outer boundary line from
%the last frame of the first movie -- this is held as a constant throughout
%the movie

blurbg = imgaussfilt(frame,numgaussblur);
th = 0.015;
outer = (edge(blurbg,'sobel',th,'nothinning'));
outer = imclearborder(outer,8);
clean = bwareaopen(outer,5000);
[out_y,out_x] = find(clean);
Par = CircleFitByPratt([out_x out_y]);
c_x = Par(1); c_y = Par(2); radius = Par(3)-rad_to_subtract; %subtract some to avoid messiness on the boundary (was hardcoded at 45 before)
outer_boundary_line = circlepoints( c_x, c_y, radius );
figure(); imshow(frame); hold on;
plot(outer_boundary_line(:,1),outer_boundary_line(:,2),'LineWidth',5);
good = input('OUTER BOUNDARY LOOKS GOOD? (1/0)');
while ~(good==0 || good == 1)
    good = input('OUTER BOUNDARY LOOKS GOOD? (1/0)');
end 
if ~good %if not, choose manually
    clf;
    ax2 = axes();
    imshow(frame); hold on;
    outer_boundary = imellipse(ax2); wait(outer_boundary);
    outer_boundary_line = getVertices(outer_boundary);
    plot(outer_boundary_line(:,1),outer_boundary_line(:,2),'LineWidth',5);
    Par = CircleFitByPratt(outer_boundary_line);
    c_x = Par(1); c_y = Par(2); radius = Par(3)-45; %subtract some to avoid messiness on the boundary
    outer_boundary_line = circlepoints( c_x, c_y, radius );
    plot(outer_boundary_line(:,1),outer_boundary_line(:,2),'LineWidth',5);
    pause();
end
outer_boundary_mask = poly2mask(outer_boundary_line(:,1),outer_boundary_line(:,2),size(frame,1),size(frame,2));
RP = regionprops(outer_boundary_mask);
region_rounded = round(RP.BoundingBox);

close all;
end