function points = circlepoints( c_x, c_y, radius )
%CIRCLEPOINTS.m Given a circle center and a radius, generate 360 points
%along it
th = (0:pi/180:2*pi)';
xunit = radius * cos(th) + c_x;
yunit = radius * sin(th) + c_y;

points = [xunit yunit];
end

