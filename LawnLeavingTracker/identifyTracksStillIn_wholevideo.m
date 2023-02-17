function [tracksStillGood, tracksThatLeft] = identifyTracksStillIn_wholevideo(tracks, tracksThatLeft)
%
%identifyTracksStillIn.m


still_visible = NaN(length(tracks),1);

for i = 1:length(tracks)
    still_visible(i) = ~(tracks(i).consecutiveInvisibleCount > 0); %was 2 before
end

stillGoodInds = find(still_visible);
tracksStillGood = tracks(stillGoodInds); %these should be tracked


newTracksThatLeft = tracks(setdiff(1:length(tracks),stillGoodInds));

WorthSavingTracksThatLeft_idx = [newTracksThatLeft(:).age]'>9; %only save tracks that left which were at least 10 frames long

newTracksThatLeft = newTracksThatLeft(WorthSavingTracksThatLeft_idx);
tracksThatLeft = [tracksThatLeft, newTracksThatLeft];
end

