function [TRACKS, POKE_STRUCT] = get_head_pokes_PERMISSIVE( TRACKS, bg_struct, INTS_OUT_BY_TRACK, OUT_INT_TRK_KEY, pixpermm )
%GET_HEAD_POKES3.m This function identifies times that the worm pokes its
%head out of the lawn while the rest of the body remains in the lawn.
%This method looks specifically at peaks in radial distance to define all
%headpokes and also demarcates a subcategory, headpoke-reversals, in which
%the headpoke is coupled to a backwards movement.

%thresholds
min_rd = (-5/112); %minimum distance from the lawn boundary = 5/112*pixpermm
min_prom = (5/112); %minimum peak prominence in the radial distance (default = 5/112)
min_wid = 0;
time_tol = 8; %tolerance for peak overlap (2.67 seconds)

speed_thresh = 0.02;

POKE_INTS_GLOBAL = []; %keeps track of the head poke intervals in the whole video, indexed by global frame numbers
POKE_INTS_BY_TRACK = []; %same but indices are from start of track = 1
POKE_TRK_KEY = []; %corresponding to the last two vectors -- which track do they refer to among TRACKS?
POKE_PEAKS_GLOBAL = [];
POKE_PEAK_IDX_BY_TRACK = [];
POKE_PEAK_IDX_BY_INT = [];
POKE_DIST_MINSUBTRACT = []; %will be in mm
POKE_IS_REV = [];
POKE_IS_FWD = [];
POKE_IS_PAUSE = [];
POKE_APPROACH_ANGLE = [];
POKE_SPEED = struct();
AVG_POKE_SPEED = [];
POKE_RAD_DIST = struct(); %will be in mm
POKE_EVHO_DIST = struct(); %will be in mm
POKE_GS = struct();
struct_counter = 1;

for trk = 1:length(TRACKS)
    donewithtrack = false;
    
    track = TRACKS(trk);
    banned_intervals = INTS_OUT_BY_TRACK(OUT_INT_TRK_KEY==trk,:);
    if isempty(banned_intervals)
        banned_intervals = [-10 -10]; %an interval that will never be a problem
    end
        
    EV_HO_DIST = NaN(track.age,1);
    RAD_DIST = NaN(track.age,1);
    HEAD_POKES = false(track.age,1);
    HEAD_POKE_FWD = false(track.age,1);
    HEAD_POKE_REV = false(track.age,1);
    HEAD_POKE_PAUSE = false(track.age,1);
    HEAD_POKE_ANGLE = NaN(track.age,1); %NaNs everywhere, and angle of approach only at head pokes
    
    for j = 1:track.age
        eh = bg_struct(track.bgvidindex(j)).ev_ho_crp_rel;
        [eh_cent_x, eh_cent_y, ~] = centroid(eh(:,1),eh(:,2));
        head = track.head(j,:);
        RAD_DIST(j) = pdist2(head,[eh_cent_x eh_cent_y]); %find head distance from the center of the lawn
        [~,dist,~] = distance2curve(eh,head); %find head distance to the nearest point on event horizon
        EV_HO_DIST(j) = dist;
    end
    tmp = EV_HO_DIST(~track.headinlawn);%make distance negative if its outside the event horizon
    EV_HO_DIST(~track.headinlawn) = -1*tmp;
    HEAD_GS = track.head_gs;
    %fill in missing data, this improves analysis
    RAD_DIST = fillmissing(RAD_DIST,'linear'); %should use 'nearest' instead! 05/02/19
    EV_HO_DIST = fillmissing(EV_HO_DIST,'linear');
    
    %smooth data a little and convert to mm
    RAD_DIST = movmean(RAD_DIST,3)./pixpermm;
    EV_HO_DIST = movmean(EV_HO_DIST,3)./pixpermm;
    
    min_ED_neg = min(-1*EV_HO_DIST);
    ED_neg_shifted = (-1*EV_HO_DIST)-(min(-1*EV_HO_DIST));
    min_rd_shifted = min_rd-min_ED_neg;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % EVENT HORIZON DISTANCE METHOD - look for negative peaks in event
    % horizon distance
    [peak_height,peak_centers,~,peak_prom, borders] = findpeaks_Elias(ED_neg_shifted,'MinPeakHeight',min_rd_shifted,'MinPeakProminence',min_prom,'MinPeakWidth',min_wid,'WidthReference','halfheight');
%     [rd_peaks_ALL, rd_peak_intervals_ALL, rd_peak_height_ALL] = refine_peak_borders(ED_neg_shifted, peak_centers, borders, peak_height); %you can only refine peaks using this method if the values are > 0.
    [rd_peaks_ALL, rd_peak_intervals_ALL, rd_peak_height_ALL] = refine_peak_borders(ED_neg_shifted, peak_centers, borders, peak_prom, peak_height); 
    rd_peak_height_ALL = rd_peak_height_ALL+min_ED_neg; %shift peak heights back to negative inside and positive outside the lawn.
    
    % REMOVE POTENTIAL POKE INTERVALS THAT AREN'T LONG ENOUGH 
    % TO DO AN ANGLE CALCULATION. (THEY ARE NOT RELIABLE)
    pokelen = rd_peak_intervals_ALL(:,2)-rd_peak_intervals_ALL(:,1);
    to_remove = pokelen<2;
    rd_peaks_ALL(to_remove) = [];
    rd_peak_intervals_ALL(to_remove,:) = [];
    rd_peak_height_ALL(to_remove) = [];    
    
    left_flank_ALL = rd_peak_intervals_ALL(:,1); %beginning of head poke is the start of radial excursion outside the lawn
    right_flank_ALL = rd_peak_intervals_ALL(:,2);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % LOOK FOR FORWARD MOVEMENT, REVERSALS AND PAUSES that follow head pokes 
    
    reversal_ints   = get_intervals( track.reverse, 1 ); %get intervals when worm was reversing
    if ~isempty(reversal_ints)
        reversal_starts = reversal_ints(:,1);
        no_reversals = false;
    else
        reversal_starts = [];
        no_reversals = true;
    end
    
    %define when the animal is pausing as when the absolute speed falls
    %below threshold.
    pausing = abs(track.speed)<speed_thresh;
    
    if isempty(rd_peaks_ALL) %if there are no headpokes, continue to the next track
        donewithtrack = true;
    end
    
    %CATEGORIZE HEAD POKES
    if ~donewithtrack %if there ARE reversals close enough to the boundary
        if no_reversals
            idx_rev_select = [];
        else
            [~, idx_rev_select] = findClosestToPeak(reversal_starts,rd_peaks_ALL,time_tol);
        end
        
        for idx = 1:length(rd_peaks_ALL)
            pokeforward = false; pokerev = false; pokepause = false; %initialize
            potentialhp = rd_peaks_ALL(idx);
            if isempty(left_flank_ALL(idx)) || isempty(right_flank_ALL(idx)) %don't include peaks at the beginning or end of a track, they can be unreliable
                continue;
            else
                %ADD A NEW POKE
                if ~logical(sum(potentialhp>banned_intervals(:,1) & potentialhp<banned_intervals(:,2))) %&& track.tailinlawn(potentialhp) %check that this head poke is not in one of the banned intervals (when the worm is OUT OF THE LAWN) AND that the tail is still in the lawn at the peak of the headpoke (actually getting rid of this stipulation)
                    disp(['NEW POKE idx = ' num2str(idx)]);
                    POKE_TRK_KEY = [POKE_TRK_KEY; trk];
                    POKE_INTS_BY_TRACK = [POKE_INTS_BY_TRACK; left_flank_ALL(idx) right_flank_ALL(idx)];
                    POKE_INTS_GLOBAL = [POKE_INTS_GLOBAL; left_flank_ALL(idx)+track.framesActive(1)-1 right_flank_ALL(idx)+track.framesActive(1)-1];
                    HEAD_POKES(potentialhp)=1; %add this peak of poke to the HEAD_POKES vector
                    POKE_PEAK_IDX_BY_INT = [POKE_PEAK_IDX_BY_INT; potentialhp-left_flank_ALL(idx)+1]; %this is the index of the head poke peak relative to this interval
                    POKE_PEAKS_GLOBAL = [POKE_PEAKS_GLOBAL; potentialhp+track.framesActive(1)-1];
                    POKE_PEAK_IDX_BY_TRACK = [POKE_PEAK_IDX_BY_TRACK; potentialhp];
                    POKE_DIST_MINSUBTRACT = [POKE_DIST_MINSUBTRACT; rd_peak_height_ALL(idx)];
                    
                    %get poke approach angle
                    eh = bg_struct(track.bgvidindex(potentialhp)).ev_ho_crp_rel; %get event horizon for this head poke
                    angle = getpokeapproachangle(track.head,track.centroid,eh,[left_flank_ALL(idx) potentialhp]);
                    POKE_APPROACH_ANGLE = [POKE_APPROACH_ANGLE ; angle];
                    HEAD_POKE_ANGLE(potentialhp) = angle;
                    %get poke speed
                    AVG_POKE_SPEED = [AVG_POKE_SPEED; nanmean(track.speed(left_flank_ALL(idx):potentialhp))]; %average speed of the radial excursion (until peak)
                    
                    POKE_SPEED(struct_counter).trackid = trk;
                    POKE_SPEED(struct_counter).speed = track.speed(left_flank_ALL(idx):right_flank_ALL(idx))';
                    POKE_RAD_DIST(struct_counter).trackid = trk;
                    POKE_RAD_DIST(struct_counter).rad_dist = RAD_DIST(left_flank_ALL(idx):right_flank_ALL(idx))';
                    POKE_EVHO_DIST(struct_counter).trackid = trk;
                    POKE_EVHO_DIST(struct_counter).evho_dist = EV_HO_DIST(left_flank_ALL(idx):right_flank_ALL(idx))';
                    POKE_GS(struct_counter).trackid = trk;
                    POKE_GS(struct_counter).grayscale = HEAD_GS(left_flank_ALL(idx):right_flank_ALL(idx))';
                    struct_counter = struct_counter+1;
                    
                    % CATEGORIZE HEAD POKE BY MOVEMENT AFTER PEAK
                    int_after_peak = potentialhp:right_flank_ALL(idx);
                    if nanmean(track.forward(int_after_peak))>0.5
                        pokeforward = true;
                    elseif nanmean(pausing(int_after_peak))>0.5
                        pokepause = true;
                    end
                    if ismember(idx,idx_rev_select) %this method supersedes the previous 2
                        pokerev = true;
                        pokeforward = false;
                        pokepause = false;
                    end
                    POKE_IS_FWD = [POKE_IS_FWD; pokeforward];
                    POKE_IS_REV = [POKE_IS_REV; pokerev];
                    POKE_IS_PAUSE = [POKE_IS_PAUSE; pokepause];
                    HEAD_POKE_FWD(potentialhp) = pokeforward;
                    HEAD_POKE_REV(potentialhp) = pokerev;
                    HEAD_POKE_PAUSE(potentialhp) = pokepause;
                end
            end
        end
    end
    TRACKS(trk).head_pokes = HEAD_POKES;
    TRACKS(trk).head_poke_forward = HEAD_POKE_FWD; %new field for just the forward head pokes
    TRACKS(trk).head_poke_reversal = HEAD_POKE_REV;
    TRACKS(trk).head_poke_pause = HEAD_POKE_PAUSE;
    TRACKS(trk).radial_dist = RAD_DIST; %save the non-smoothed version
    TRACKS(trk).ev_ho_dist = EV_HO_DIST;
    TRACKS(trk).head_poke_angle = HEAD_POKE_ANGLE;
end

POKE_IS_FWD = logical(POKE_IS_FWD);
POKE_IS_REV = logical(POKE_IS_REV);
POKE_IS_PAUSE = logical(POKE_IS_PAUSE);

POKE_STRUCT.POKE_INTS_GLOBAL = POKE_INTS_GLOBAL;
POKE_STRUCT.POKE_INTS_BY_TRACK = POKE_INTS_BY_TRACK;
POKE_STRUCT.POKE_TRK_KEY = POKE_TRK_KEY;
POKE_STRUCT.POKE_PEAK_IDX_BY_INT = POKE_PEAK_IDX_BY_INT;
POKE_STRUCT.POKE_PEAKS_GLOBAL = POKE_PEAKS_GLOBAL;
POKE_STRUCT.POKE_DIST_MINSUBTRACT = POKE_DIST_MINSUBTRACT;
POKE_STRUCT.POKE_APPROACH_ANGLE = POKE_APPROACH_ANGLE;
POKE_STRUCT.POKE_IS_FWD = POKE_IS_FWD;
POKE_STRUCT.POKE_IS_REV = POKE_IS_REV;
POKE_STRUCT.POKE_IS_PAUSE = POKE_IS_PAUSE;
POKE_STRUCT.AVG_POKE_SPEED = AVG_POKE_SPEED;
POKE_STRUCT.POKE_SPEED = POKE_SPEED;
POKE_STRUCT.POKE_PEAK_IDX_BY_TRACK = POKE_PEAK_IDX_BY_TRACK;
POKE_STRUCT.POKE_RAD_DIST = POKE_RAD_DIST;
POKE_STRUCT.POKE_EVHO_DIST = POKE_EVHO_DIST;
POKE_STRUCT.POKE_GS = POKE_GS;
end

%LOCAL FUNCTIONS
function [rd_peak_select, idx_select] = findClosestToPeak(idx,rd_peaks,timetolerance)
dist_mat = pdist2(rd_peaks,idx); %time separation of rev_idx and peaks in rd
[min_dist , idx] = min( dist_mat,[],1 ); %get the closest peaks
select = min_dist < timetolerance;  %select out those close enough together based on the tolerance we specified above.
% idxlist_used = find(select); %which of the original reversals have a peak in radial distance close enough
idx_select = unique(idx(select)); %the same peak may be chosen multiple times -- ensure that it is listed only once.
rd_peak_select = rd_peaks(idx_select); %get the actual matches (in rd)
end

function angle = getpokeapproachangle(head,cent,eh,poke_int)
y = eh(:,2);
x = eh(:,1);

poke_head = head(poke_int(1):poke_int(end),:);

if size(poke_head,1)==1
    head_crossings = [];
else
    try
        head_crossings = InterX([x y]',poke_head')';
    catch
        error('debug: problem with InterX get angle!');
    end
end

% 1a. look for intersections of the head path with the event horizon. if
% one exists, add it to the event horizon points
if ~isempty(head_crossings) && size(head_crossings,1)==1 && ~ismember(head_crossings,[x y],'rows')
    x = [x; head_crossings(1)]; y = [y; head_crossings(2)]; %add the intersection point to event horizon
    [x,y] = sortPointsCw(x,y); %make sure the points are in order (clockwise)
    closest_eh_point = head_crossings;
    [~,which_eh_point] = min(sqrt((x-closest_eh_point(1)).^2 + (y-closest_eh_point(2)).^2)); %find out the index of the head crossing in the new event horizon
else
    [eh_points,head_dist,~] = distance2curve([x y],poke_head);
    % 1b. look for the closest point in the head trajectory leading up to HP to
    % a point on the event horizon (if there was a head intersection, this
    % should be that point)
    [~,h_i] = min(head_dist);
    closest_eh_point = eh_points(h_i,:);
    [~,which_eh_point] = min(sqrt((x-closest_eh_point(1)).^2 + (y-closest_eh_point(2)).^2));
end
slope = gradient(y,x);

% 2. what is the tangent line to that point?
x_range = (x(which_eh_point)-5:1:x(which_eh_point)+5)';
tangent = (x_range-x(which_eh_point))*slope(which_eh_point)+y(which_eh_point);
%find a local segment of this tangent that does not loop over the range of x 
t_vec = [x_range(end) tangent(end)]-[x_range(1) tangent(1)]; t_vec = [t_vec 0];

% 3. find the centroid vector through the points in poke_int
smth_window = 3;
c_smth = [movmean(cent(:,1),smth_window,'omitnan') movmean(cent(:,2),smth_window,'omitnan')];
poke_cent = c_smth(poke_int(1):poke_int(end),:);
% c_vec = poke_cent(end,:)-poke_cent(1,:); c_vec = [c_vec 0]; %centroid vector formulated as the last centroid position minus the first
c_vec = mean(diff(poke_cent,1,1),1); c_vec = [c_vec 0]; %alternate formulation - average centroid during lead up to head poke

% 4. calculate the angle between the centroid vector and the closest point
% tangent line
angle = abs(atan2d(norm(cross(c_vec,t_vec)),dot(c_vec,t_vec)));
if angle>90 %may need to subtract angle from 180 to get the angle between 0 and 90
    angle = 180-angle;
end

% disp(['Angle of Approach is ... ' num2str(angle)]);
% h=figure(); hold on;
% plot(x,y); axis equal;
% plot(poke_head(:,1),poke_head(:,2));
% plot(poke_cent(:,1),poke_cent(:,2));
% scatter(poke_cent(1,1),poke_cent(1,2),1,'b+');
% scatter(poke_cent(end,1),poke_cent(end,2),1,'r+');
% scatter(closest_eh_point(1),closest_eh_point(2),2,'g+');
% plot(x_range,tangent);
% pause(); close(h);
end


function [peaks, peak_intervals, peak_height] = refine_peak_borders(timevec, peak_centers, borders, peakprom, peakheight)
%refine these boundaries by looking for intersections of the line
%of half-maximum height for each peak with the peak on either side. (by
%interpolation).
peaks = zeros(length(peak_centers),1);
peak_intervals = zeros(length(peak_centers),2);
peak_height = zeros(length(peak_centers),1);

% halfheight = peakheight./2;
halfheight = peakheight-peakprom;
for i = 1:length(halfheight)
    peak_center = peak_centers(i);
    leftborder = borders(i,1); rightborder = borders(i,2);
    halfheight_line = [leftborder rightborder ; halfheight(i) halfheight(i)]; %first row is x values of borders, second row is the half height (horizontal line)
    chunk = [leftborder:rightborder; timevec(leftborder:rightborder)']; %chunk of the smth vector in which to look for crossings
    crossings = InterX(halfheight_line,chunk); %find the intersections
    left_flank = leftborder; right_flank = rightborder; %initialize
    
    if ~isempty(crossings) %if there are intersections within the borders, use those instead, since they are more refined.
        x_vals = crossings(1,:);
        lessthan = x_vals(x_vals<peak_center);
        if ~isempty(lessthan)
            test = round(max(lessthan)); %greatest member less than peak center
            if test > left_flank
                left_flank = test;
            end
        end
        greaterthan = x_vals(x_vals>peak_center);
        if ~isempty(greaterthan)
            test = round(min(greaterthan)); %least member greater than peak center
            if test < right_flank
                right_flank = test;
            end
        end
    end
    peaks(i) = peak_center;
    peak_intervals(i,:) = [left_flank right_flank];
    peak_height(i) = peakheight(i);
end

end

