function [spline, curvature, posture_angle] = getWormSpline010219( wormskel )
%GETWORMSPLINE2.m this function calculates the smoothed spline and
%curvature based on the midline pixels identified during segmentation. then
%comes up with curvature

rough_x = wormskel(:,1);
rough_y = wormskel(:,2);
xx = linspace(1,length(rough_x),49);
p = 0.6;
pp_x = csaps(1:length(rough_x),rough_x,p,xx,[5 ones(1,length(rough_x)-2) 5]);
pp_y = csaps(1:length(rough_y),rough_y,p,xx,[5 ones(1,length(rough_y)-2) 5]);

spline = [pp_x' pp_y'];
[posture_angle,~] = angleFromXY(pp_x, pp_y); %AEX code

Vertices = spline;
curvature=LineCurvature2D(Vertices);%gets the radius of the tangent circle at every point
end

