%% Update Assigned Tracks
% The |updateAssignedTracks| function updates each assigned track with the
% corresponding detection. It calls the |correct| method of
% |vision.KalmanFilter| to correct the location estimate. Next, it stores
% the new bounding box, and increases the age of the track and the total
% visible count by 1. Finally, the function sets the invisible count to 0.

% Elias Scheer
% 1-12-17
% borrowed from MultiObjectTracking example

function tracks = updateAssignedTracks_wholevideo2(tracks,assignments,x_shift,y_shift,centroids, bboxes, cropworms, cropworms_orig, splines, end1s, end2s, end1s_g, end2s_g, omegas, curvatures, posture_angles, curr_frame, videoframe, videoname)
numAssignedTracks = size(assignments, 1);
for i = 1:numAssignedTracks
    trackIdx = assignments(i, 1);
    detectionIdx = assignments(i, 2);
    centroid = centroids(detectionIdx, :);
    bbox = bboxes(detectionIdx, :);
    cropworm = cropworms{detectionIdx};
    cropworm_orig = cropworms_orig{detectionIdx};
    spline = splines{detectionIdx};
    end1 = end1s(detectionIdx,:);
    end2 = end2s(detectionIdx,:);
    end1_g = end1s_g(detectionIdx);
    end2_g = end2s_g(detectionIdx);
    curvature = curvatures{detectionIdx};
    posture_angle = posture_angles{detectionIdx};
    omega = omegas(detectionIdx);
    
    x_shift_upToNow = tracks(trackIdx).x_shift;
    tracks(trackIdx).x_shift = [x_shift_upToNow; x_shift];
    
    y_shift_upToNow = tracks(trackIdx).y_shift;
    tracks(trackIdx).y_shift = [y_shift_upToNow; y_shift];
    
    % Correct the estimate of the object's location
    % using the new detection.
    correct(tracks(trackIdx).kalmanFilter, centroid);
    
    % Replace predicted bounding box with detected
    % bounding box & append the new detection to the past observations
    bbox_upToNow = tracks(trackIdx).bbox;
    bbox_upToNow(size(bbox_upToNow,1),:) = bbox; %remove predicted one and correct it with new one
    tracks(trackIdx).bbox = bbox_upToNow;
    
    centroid_upToNow = tracks(trackIdx).centroid;
    centroid_upToNow(size(bbox_upToNow,1),:) = centroid; %remove predicted one and correct it with new one
    tracks(trackIdx).centroid = centroid_upToNow;
    
    %Append cropped worm image, spline, and curvature that defines its shape
    cropworm_upToNow = tracks(trackIdx).cropworm;
    sizethusfar = size(cropworm_upToNow,2);
    cropworm_upToNow{sizethusfar+1} = cropworm; %append latest crop worm
    tracks(trackIdx).cropworm = cropworm_upToNow;
    
    cropworm_orig_upToNow = tracks(trackIdx).cropworm_orig;
    sizethusfar = size(cropworm_orig_upToNow,2);
    cropworm_orig_upToNow{sizethusfar+1} = cropworm_orig; %append latest crop worm
    tracks(trackIdx).cropworm_orig = cropworm_orig_upToNow;
    
    splines_upToNow = tracks(trackIdx).spline;
    splines_upToNow{sizethusfar+1} = spline; %append the latest spline
    tracks(trackIdx).spline = splines_upToNow;
    
    tracks(trackIdx).end1 = [tracks(trackIdx).end1; end1]; %append the endpoints
    tracks(trackIdx).end2 = [tracks(trackIdx).end2; end2];
    
    tracks(trackIdx).end1_g = [tracks(trackIdx).end1_g; end1_g]; %append the grayscale values associated with the head and tail
    tracks(trackIdx).end2_g = [tracks(trackIdx).end2_g; end2_g];
    
    tracks(trackIdx).omega = [tracks(trackIdx).omega; omega]; %append the grayscale values associated with the head and tail
    
    curvatures_upToNow = tracks(trackIdx).curvature;
    curvatures_upToNow{sizethusfar+1} = curvature; %append the latest curvature
    tracks(trackIdx).curvature = curvatures_upToNow;
    
    posture_angles_upToNow = tracks(trackIdx).posture_angle;
    posture_angles_upToNow{sizethusfar+1} = posture_angle; %append the latest posture_angle
    tracks(trackIdx).posture_angle = posture_angles_upToNow;
    
    % Update track's age.
    tracks(trackIdx).age = tracks(trackIdx).age + 1;
    
    % Update visibility.
    tracks(trackIdx).totalVisibleCount = tracks(trackIdx).totalVisibleCount + 1;
    tracks(trackIdx).consecutiveInvisibleCount = 0; %reset this so that previous invisibility does not count against future trackstillin calls
    
    %Update frames active.
    tracks(trackIdx).framesActive = [tracks(trackIdx).framesActive curr_frame];
    
    %Update video name
    name_upToNow = tracks(trackIdx).videoname;
    name_upToNow{sizethusfar+1} = videoname; %append the latest video file name
    tracks(trackIdx).videoname = name_upToNow;
    
    %Update video frame.
    tracks(trackIdx).videoframe = [tracks(trackIdx).videoframe videoframe];
    
    
end
end