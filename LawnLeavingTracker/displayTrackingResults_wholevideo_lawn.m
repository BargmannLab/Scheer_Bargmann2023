%% Display Tracking Results
% The |displayTrackingResults| function draws a bounding box and label ID
% for each track on the video frame and the foreground mask. It then
% displays the frame and the mask in their respective video players.

% Elias Scheer
% 1-12-17
% borrowed from MultiObjectTracking example

function displayTrackingResults_wholevideo_lawn(trackPlayer, tracks, img, end1s, end2s, ev_ho_x, ev_ho_y)
%blob player
% blobs = insertShape(double(thresh_cleaned), 'Rectangle', bboxes, 'Color', 'green','LineWidth',2);
% blobPlayer.step(blobs);

% masked_worms = uint8(repmat(img, [1, 1, 3])) .* 255;
masked_worms = img;

if ~isempty(tracks)
    % Get centroids and bounding boxes for existing tracks
    curr_centroids = NaN(length(tracks),2);
    bboxes = NaN(length(tracks),4);
    curr_splines = cell(length(tracks));

    NaN(length(tracks),104); %51 line segments for 52 vertices x2 for x,y = 104
    for j = 1:length(tracks)
        tmp_centroid = tracks(j).centroid;
        curr_centroids(j,:) = tmp_centroid(size(tmp_centroid,1),:);
        curr_bbox = tracks(j).bbox;
        curr_bbox = [curr_bbox(:,1)-5 curr_bbox(:,2)-5 curr_bbox(:,3)+10 curr_bbox(:,4)+10]; %put back the expanded bbox
        bboxes(j,:) = curr_bbox(size(curr_bbox,1),:);
        %get splines matrix in format for plotting
        tmp_spline = tracks(j).spline;
        tmp_spline = tmp_spline{size(tmp_spline,2)}';
        if ~isempty(tmp_spline) && sum(sum(isnan(tmp_spline)))<1
            %express splines as an M-by-2L matrix, where each row is a vector representing a polyline with L number of vertices.
            t(1:2:length(tmp_spline)*2) = tmp_spline(1,:);
            t(2:2:length(tmp_spline)*2) = tmp_spline(2,:);
            curr_splines{j} = t;
        end
       
    end
    % Get ids.
    ids = int32([tracks(:).id]);
    
    % Create labels for objects indicating the ones for
    % which we display the predicted rather than the actual
    % location.
    labels = cellstr(int2str(ids'));
    predictedTrackInds = [tracks(:).consecutiveInvisibleCount] > 0;
    isPredicted = cell(size(labels));
    isPredicted(predictedTrackInds) = {' predicted'};
    labels = strcat(labels, isPredicted);
    
    % Draw the objects on the mask.
    centroids_to_draw = [curr_centroids 3*ones(size(curr_centroids,1),1)];
    masked_worms = insertShape(masked_worms,'FilledCircle',centroids_to_draw,'LineWidth',1,'Color','magenta');
    
    % Draw the detected endpoints (not from spline)
    for k = 1:size(end1s,1)
        pt1 = end1s(k,:); pt2 = end2s(k,:);
        if sum(isnan(pt1))==0 && sum(isnan(pt2))==0
            points = [pt1; pt2];
            points_to_draw = [points 2*ones(size(points,1),1)];
            masked_worms = insertShape(masked_worms,'FilledCircle',points_to_draw,'LineWidth',1,'Color','green');
        end
    end
    
    % Draw splines on the mask. & Draw spline endpoints
    for k = 1:length(curr_splines)
        curr_spline = curr_splines{k};
        if ~isempty(curr_spline) && sum(isnan(curr_spline))==0
            masked_worms = insertShape(masked_worms,'Line',curr_spline,'LineWidth',1,'Color','cyan');
        end
    end

        
%     %plots the entire trailing centroid path before now 
%     for j = 1:length(tracks) 
%         masked_worms = insertMarker(masked_worms,tracks(j).centroid,'*','color','blue','size',1);
%     end

    masked_worms = insertObjectAnnotation(masked_worms, 'rectangle', bboxes, labels,'Color','magenta');
    
    masked_worms = insertText(masked_worms, [10 10], length(tracks), 'BoxOpacity', 1, 'FontSize', 45);
    
    masked_worms = insertMarker(masked_worms,[ev_ho_x ev_ho_y],'+','color','red','size',1);
end

% Display the mask and the frame.
trackPlayer.step(masked_worms);

end