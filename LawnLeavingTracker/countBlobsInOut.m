% countBlobsInOut.m
% 04/26/2017
% This function takes in the tracks which have been saved over the course
% of the video and outputs a count of the number of blobs found inside and
% outside the event horizon at every frame. Note that the blobs outside the
% event horizon will be much more reliable than those inside since there
% will be many multiblobs inside. The number of blobs outside can be used
% to calculate an aversion ratio if the total number of animals in the
% video is known.

function [headinlawn, centroidinlawn, tailinlawn, fullyinlawn] = countBlobsInOut(track, bg_struct)
bgvidindex = track.bgvidindex;  %this is the index key to knowing which event horizon was active at a particular frame

bbox = track.bbox;
bbox_verts =   [[bbox(:,1) bbox(:,2)]...          % top left
    [bbox(:,1) bbox(:,2)+bbox(:,4)]...            % bottom left
    [bbox(:,1)+bbox(:,3) bbox(:,2)]...            % top right
    [bbox(:,1)+bbox(:,3) bbox(:,2)+bbox(:,4)]];   % bottom right

topleft_in = zeros(track.age,1);
bottomleft_in = zeros(track.age,1);
topright_in = zeros(track.age,1);
bottomright_in = zeros(track.age,1);

headinlawn = zeros(track.age,1);
centroidinlawn = zeros(track.age,1);
tailinlawn = zeros(track.age,1);

for i = 1:track.age
    ev_ho = bg_struct(bgvidindex(i)).ev_ho_crp_rel; %get the event horizon that corresponds to this frame of the video
    x = ev_ho(:,1); y = ev_ho(:,2);
    
    topleft_in(i) = inpolygon(double(bbox_verts(i,1)),double(bbox_verts(i,2)),x,y);
    bottomleft_in(i) = inpolygon(double(bbox_verts(i,3)),double(bbox_verts(i,4)),x,y);
    topright_in(i) = inpolygon(double(bbox_verts(i,5)),double(bbox_verts(i,6)),x,y);
    bottomright_in(i) = inpolygon(double(bbox_verts(i,7)),double(bbox_verts(i,8)),x,y);

    
    headinlawn(i) = inpolygon(track.head(i,1),track.head(i,2),x,y);
    centroidinlawn(i) = inpolygon(track.centroid(i,1),track.centroid(i,2),x,y);
    tailinlawn(i) = inpolygon(track.tail(i,1),track.tail(i,2),x,y);
end
%wherever the worm's bounding box is fully inside the lawn, set to 1
fullyinlawn = topleft_in&bottomleft_in&topright_in&bottomright_in;

headinlawn = logical(headinlawn);
centroidinlawn = logical(centroidinlawn);
tailinlawn = logical(tailinlawn);
fullyinlawn = logical(fullyinlawn);

end




























