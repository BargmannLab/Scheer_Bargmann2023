function angles = circCurvature(points, edgeLength, varargin)
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
% See also CURVATURE, CIRCCOMPUTECHAINCODELENGTHS
%
%
% ? Medical Research Council 2012
% You will not remove any copyright or other notices from the Software; 
% you must reproduce all copyright notices and other proprietary 
% notices on any copies of the Software.

% Are there chain-code lengths?
if length(varargin) == 1
    chainCodeLengths = varargin{1};
else
    chainCodeLengths = [];
end

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

% Are there enough points?
if (isempty(chainCodeLengths) && size(points,1) < 2 * edgeLength + 1) || ...
        (~isempty(chainCodeLengths) && ...
        chainCodeLengths(size(points,1)) < 2 * edgeLength + 1)
    warning('circCurvature:EdgesTooLong', ...
        'The length of the edges from the vertex exceeds the number of points');
    angles(1:size(points, 1),1) = nan;
    return;
end

% Pre-allocate memory.
angles(1:size(points,1),1) = NaN; % orient the vector as rows

% Compute the curvature using the array indices for length.
if isempty(chainCodeLengths)

    % Initialize the edges.
    edgeLength = round(edgeLength);
    p1 = [points((end - edgeLength + 1):end,:); ...
        points(1:(end - edgeLength),:)];
    p2 = [points((edgeLength + 1):end,:); points(1:edgeLength,:)];
    
    % Use the difference in tangents to measure the angle.
    angles = atan2(points(:,1) - p2(:,1), points(:,2) - p2(:,2)) - ...
        atan2(p1(:,1) - points(:,1), p1(:,2) - points(:,2));
    for i = 1:length(angles)
        if angles(i) > pi
            angles(i) = angles(i) - 2 * pi;
        elseif angles(i) < -pi
            angles(i) = angles(i) + 2 * pi;
        end
        angles(i) = angles(i) * 180 / pi;
    end
    
% Compute the curvature using the chain-code lengths.
else
    
    % Initialize the first edge.
    p1I = size(points,1);
    pvI = 1;
    pvLength = chainCodeLengths(size(points, 1)) + chainCodeLengths(pvI);
    e1 = pvLength - chainCodeLengths(p1I);
    while p1I > 0 && e1 < edgeLength
        p1I = p1I - 1;
        e1 = pvLength - chainCodeLengths(p1I);
    end
    
    % Compute the angles.
    sqrt2 = sqrt(2);
    p2I = pvI;
    while pvI <= size(points, 1)
        
        % Compute the second edge length.
        if p2I >= pvI
            e2 = chainCodeLengths(p2I) - chainCodeLengths(pvI);
            
        % Compute the wrapped, second edge length.
        else
            e2 = chainCodeLengths(size(points, 1)) + ...
                chainCodeLengths(p2I) - chainCodeLengths(pvI);
        end
        
        % Find the second edge.
        while e2 < edgeLength
            p2I = p2I + 1;
            
            % Wrap.
            if p2I > size(points, 1);
                p2I = p2I - size(points, 1);
            end
            
            % Compute the second edge length.
            if p2I >= pvI
                e2 = chainCodeLengths(p2I) - chainCodeLengths(pvI);
                
            % Compute the wrapped, second edge length.
            else
                e2 = chainCodeLengths(size(points, 1)) + ...
                    chainCodeLengths(p2I) - chainCodeLengths(pvI);
            end
        end
        
        % Compute fractional pixels for the first edge.
        % Note: the first edge is equal to or just over the requested edge
        % length. Therefore, the fractional pixels for the requested length
        % lie on the line separating point 1 (index = p1I) from the next
        % closest point to the vertex (index = p1I + 1). Now, we need to
        % add the difference between the requested and real distance (de1)
        % to point p1I, going in a line towards p1I + 1. Therefore, we need
        % to solve the differences between the requested and real x & y
        % (dx1 & dy1). Remember the requested x & y lie on the slope
        % between point p1I and p1I + 1. Therefore, dy1 = m * dx1 where m
        % is the slope. We want to solve de1 = sqrt(dx1^2 + dy1^2).
        % Plugging in m, we get de1 = sqrt(dx1^2 + (m*dx1)^2). Then
        % re-arrange the equality to solve:
        %
        % dx1 = de1/sqrt(1 + m^2) and dy1 = de1/sqrt(1 + (1/m)^2)
        %
        % But, Matlab uses (r,c) = (y,x), so x & y are reversed.
        de1 = e1 - edgeLength;
        if p1I < size(points, 1)
            nextP1I = p1I + 1;
        else
            nextP1I = p1I + 1 - size(points, 1);
        end
        dp1 = points(nextP1I,:) - points(p1I,:);
        if any(dp1 == 0)
            p1 = de1 .* sign(dp1) + points(p1I,:);
        elseif all(abs(dp1) == 1)
            p1 = (de1 / sqrt2) .* dp1 + points(p1I,:);
        else
            dy1 = de1 / sqrt(1 + (dp1(2) / dp1(1)) ^ 2);
            dx1 = de1 / sqrt(1 + (dp1(1) / dp1(2)) ^ 2);
            p1 = [dy1 dx1] .* sign(dp1) + points(p1I,:);
        end
        
        % Compute fractional pixels for the second edge.
        de2 = e2 - edgeLength;
        if p2I > 1
            prevP2I = p2I - 1;
        else
            prevP2I = p2I - 1 + size(points, 1);
        end
        dp2 = points(prevP2I,:) - points(p2I,:);
        if any(dp2 == 0)
            p2 = de2 .* sign(dp2) + points(p2I,:);
        elseif all(abs(dp2) == 1)
            p2 = (de2 / sqrt2) .* dp2 + points(p2I,:);
        else
            dy2 = de2 / sqrt(1 + (dp2(2) / dp2(1)) ^ 2);
            dx2 = de2 / sqrt(1 + (dp2(1) / dp2(2)) ^ 2);
            p2 = [dy2 dx2] .* sign(dp2) + points(p2I,:);
        end
        
        % Use the difference in tangents to measure the angle.
        angles(pvI) = ...
            atan2(points(pvI,1) - p2(1), points(pvI,2) - p2(2)) - ...
            atan2(p1(1) - points(pvI,1), p1(2) - points(pvI,2));
        if angles(pvI) > pi
            angles(pvI) = angles(pvI) - 2 * pi;
        elseif angles(pvI) < -pi
            angles(pvI) = angles(pvI) + 2 * pi;
        end
        angles(pvI) = angles(pvI) * 180 / pi;
        
        % Advance.
        pvI = pvI + 1;
        
        % Compute the first edge length.
        if pvI <= size(points, 1)
            if p1I <= pvI
                e1 = chainCodeLengths(pvI) - chainCodeLengths(p1I);
                
            % Compute the wrapped, second edge length.
            else
                e1 = chainCodeLengths(size(points, 1)) + ...
                    chainCodeLengths(pvI) - chainCodeLengths(p1I);
            end
            
            % Find the first edge.
            nextE1 = e1;
            nextP1I = p1I;
            while nextE1 > edgeLength
                
                % Advance.
                e1 = nextE1;
                p1I = nextP1I;
                nextP1I = p1I + 1;
                
                % Wrap.
                if nextP1I > size(points, 1);
                    nextP1I = nextP1I - size(points, 1);
                end
                
                % Compute the first edge length.
                if nextP1I <= pvI
                    nextE1 = chainCodeLengths(pvI) - ...
                        chainCodeLengths(nextP1I);
                    
                % Compute the wrapped, second edge length.
                else
                    nextE1 = chainCodeLengths(size(points, 1)) + ...
                        chainCodeLengths(pvI) - chainCodeLengths(nextP1I);
                end
            end
        end
    end
end

% Transpose the angles.
if isTransposed
    angles = angles';
end
end
