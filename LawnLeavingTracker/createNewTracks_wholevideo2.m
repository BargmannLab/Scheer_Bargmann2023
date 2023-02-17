%% Create New Tracks
% Create new tracks from unassigned detections. Assume that any unassigned
% detection is a start of a new track. In practice, you can use other cues
% to eliminate noisy detections, such as size, location, or appearance.

% Elias Scheer
% 1-12-17

% borrowed from MultiObjectTracking example

function [tracks, nextId] = createNewTracks_wholevideo2(tracks,nextId,x_shift,y_shift,unassignedDetections,centroids,bboxes,cropworms,cropworms_orig,splines,end1s, end2s, end1s_g, end2s_g, omegas,curvatures, posture_angles, curr_frame,videoframe,videoname)
centroids = centroids(unassignedDetections, :);
bboxes = bboxes(unassignedDetections, :);
cropworms = cropworms(unassignedDetections);
cropworms_orig = cropworms_orig(unassignedDetections);
splines = splines(unassignedDetections);
omegas = omegas(unassignedDetections);
end1s = end1s(unassignedDetections,:);
end2s = end2s(unassignedDetections,:);
end1s_g = end1s_g(unassignedDetections);
end2s_g = end2s_g(unassignedDetections);
curvatures = curvatures(unassignedDetections);
posture_angles = posture_angles(unassignedDetections);
videoname = {cellstr(videoname)};

for i = 1:size(centroids, 1)
    
    centroid = centroids(i,:);
    bbox = bboxes(i, :);
    cropworm = {cropworms(i)};
    cropworm_orig = {cropworms_orig(i)};
    spline = {splines(i)};
    end1 = end1s(i,:);
    end2 = end2s(i,:);
    end1_g = end1s_g(i);
    end2_g = end2s_g(i);
    omega = omegas(i);
    curvature = {curvatures(i)};
    posture_angle = {posture_angles(i)};
    
    
    % Create a Kalman filter object.
    kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
        centroid, [200, 50], [100, 25], 100);
    
    % Create a new track.
    %put NaNs in for speed and angular speed, head and tail
    newTrack = struct(...
        'id', nextId, ...
        'x_shift',x_shift,...
        'y_shift',y_shift,...
        'centroid', centroid, ...
        'bbox', bbox, ...
        'cropworm',cropworm,...
        'cropworm_orig',cropworm_orig,...
        'spline',spline,...
        'omega',omega,...
        'end1',end1,...
        'end2',end2,...
        'end1_g',end1_g,...
        'end2_g',end2_g,...
        'curvature',curvature,...
        'posture_angle',posture_angle,...
        'kalmanFilter', kalmanFilter, ...
        'age', 1, ...
        'totalVisibleCount', 1, ...
        'consecutiveInvisibleCount', 0,...
        'framesActive', curr_frame,...
        'videoname', videoname, ...
        'videoframe', videoframe);
    
    % Add it to the array of tracks.
    tracks(end + 1) = newTrack;
    
    % Increment the next id.
    nextId = nextId + 1;
end
end
