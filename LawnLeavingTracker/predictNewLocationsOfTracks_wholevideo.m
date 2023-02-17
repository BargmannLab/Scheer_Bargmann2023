%% Predict New Locations of Existing Tracks
% Use the Kalman filter to predict the centroid of each track in the
% current frame, and update its bounding box accordingly.
% Elias Scheer
% 1-12-16
% borrowed from MultipleObjectTracking example

function tracks = predictNewLocationsOfTracks_wholevideo(tracks)
for i = 1:length(tracks)
    bbox = tracks(i).bbox;
    bbox = bbox(size(bbox,1),:); %just from the last frame
    
    % Predict the current location of the track.
    predictedCentroid = predict(tracks(i).kalmanFilter); %naive bayes
    
    % Shift the bounding box so that its center is at
    % the predicted location.
%     predictedCentroid = int32(predictedCentroid) - bbox(3:4) / 2;
    predictedCentroid = predictedCentroid - bbox(3:4)/2;
    tracks(i).centroid = [tracks(i).centroid ; predictedCentroid]; %not sure why these predictions even need to go into the log...
    bbox_upToNow = tracks(i).bbox;
    predictedBbox = [predictedCentroid, bbox(3:4)];
    tracks(i).bbox =  [bbox_upToNow ; predictedBbox]; %append
end
end