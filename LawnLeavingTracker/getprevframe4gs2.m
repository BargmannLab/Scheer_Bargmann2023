function cent_lookback = getprevframe4gs2(tracks,tracksThatLeft,index,curr_cent,numlookback,frame,pixpermm)
%GETPREVDIST.M Given a track, get the distance of the current position
%from the previous locations in order to figure out how far back to look
%for a frame without the worm for grayscale value extraction.

mindist = (50/112)*pixpermm;
allTracks = [tracksThatLeft,tracks];
range = index-numlookback:index;
%figure out which frames in which tracks this is made up of
cent = NaN(size(range,2),2);
for i = 1:length(allTracks)
    fa = allTracks(i).framesActive;
    [~,IA,IB] = intersect(fa,range);
    cent(IB,:) = allTracks(i).centroid(IA,:);
end

if sum(isnan(cent(:,1)))>0.5*size(cent,1) %if a track just started or if the lookback is mostly frames when there were no tracks, just go back as far as possible
    cent_lookback = 360;
    return;
end

cent_dist = diag(pdist2(cent,repmat(curr_cent,length(cent),1)));
%find peaks
[c_pks,c_locs] = findpeaks(cent_dist,'MinPeakHeight',mindist,'MinPeakProminence',5);
c_locs_rel = c_locs+(index-numlookback);

if isempty(c_pks) %either all the peaks are too low or its just smoothly decreasing i.e. when the worm is moving fast
    tmp = (abs(cent_dist-(mindist+3))<=1); %this is to select points that are very close to our mindist if its a smoothly decreasing distance
    cent_closest = max(find(tmp));
    if isempty(cent_closest)
        [~, cent_closest] = nanmax(cent_dist);
    elseif cent_dist(cent_closest)<mindist %if the best you can do is less than mindist+5, just take the max distance
        [~, cent_closest] = nanmax(cent_dist);
    end
    cent_closest_rel = cent_closest+(index-numlookback);
    cent_lookback = index-cent_closest_rel+1;
else
    cent_lookback = index-c_locs_rel(end)+1;
end

% h = figure(1);
% imshow(frame); hold on;
% scatter(curr_cent(1),curr_cent(2),'g+');
% plot(cent(:,1),cent(:,2));
% scatter(cent(size(range,2)-cent_lookback,1),cent(size(range,2)-cent_lookback,2),'ro');
% % scatter(tracks(1).end1(:,1),tracks(1).end1(:,2))
% debug = input('debug?');
% if debug==1
%     disp('debug');
% end



end

