function angles = CurvatureWindow_Elias(points, edgeLength)
%CIRCCURVATURE Compute the curvature for a clockwise, circularly-connected
%vector of points.
%
%   ANGLES = CIRCCURVATURE(POINTS, EDGELENGTH)
%
%   ANGLES = CIRCCURVATURE(POINTS, EDGELENGTH, CHAINCODELENGTHS)
%
%   Inputs:
%       points           - the vector of clockwise, circularly-connected
%                          points ((x,y) pairs).
%       edgeLength       - the length of edges from the angle vertex.
%       chainCodeLengths - the chain-code length at each point;
%                          if empty, the array indices are used instead
%   Output:
%       angles - the angles of curvature per point (0 = none to +-180 =
%                maximum curvature). The sign represents whether the angle
%                is convex (+) or concave (-).
%

% Are the the points 2 dimensional?
if ndims(points) ~=2 || (size(points, 1) ~= 2 && size(points, 2) ~= 2)
    error('circCurvature:PointsNot2D', ...
        'The matrix of points must be 2 dimensional');
end

% Orient the points as a N-by-2 matrix.
isTransposed = false;
if size(points, 2) ~= 2
    points = points';
    isTransposed = true;
end

% Pre-allocate memory.
angles(1:size(points,1),1) = NaN; % orient the vector as rows

% Compute the curvature using the array indices for length.

% Initialize the edges.
edgeLength = round(edgeLength);
p1 = [NaN(edgeLength,2); points(1:(end - edgeLength),:)]; % padding NaNs first
p2 = [points((edgeLength + 1):end,:); NaN(edgeLength,2)]; % padding NaNs last

% Use the difference in tangents to measure the angle.
for i = 1:length(angles)
    currPt = points(i,:);
    prevPt = p1(i,:);
    if sum(isnan(prevPt))==2
        prevPt = points(1,:);
    end
    nextPt = p2(i,:);
    if sum(isnan(nextPt))==2
        nextPt = points(end,:);
    end
    angle = atan2(currPt(1) - nextPt(1), currPt(2) - nextPt(2)) - ...
        atan2(prevPt(1) - currPt(1), prevPt(2) - currPt(2));
    if angle > pi
        angle = angle - 2 * pi;
    elseif angle < -pi
        angle = angle + 2 * pi;
    end
    angles(i) = angle * 180 / pi;
end
%set the first and last entries to the second and second to last entries,
%respectively. this gets rid of jump discontinuities that come from lack of
%data
angles(1)=angles(2);
angles(end)=angles(end-1);

% Transpose the angles.
if isTransposed
    angles = angles';
end
end
