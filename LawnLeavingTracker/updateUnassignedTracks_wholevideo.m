
%% Update Unassigned Tracks
% Mark each unassigned track as invisible, and increase its age by 1.

% Elias Scheer
% 1-12-17
% borrowed from MultiObjectTracking example

function [tracks, tracksThatLeft] = updateUnassignedTracks_wholevideo(tracks,tracksThatLeft,unassignedTracks)
for i = 1:length(unassignedTracks)
    ind = unassignedTracks(i);
    tracks(ind).consecutiveInvisibleCount = tracks(ind).consecutiveInvisibleCount + 1;
end

if length(unassignedTracks)>=1
    [tracks, tracksThatLeft] = identifyTracksStillIn_wholevideo(tracks, tracksThatLeft);
end

end

