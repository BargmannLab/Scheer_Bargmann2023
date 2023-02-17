function POKE_STRUCT = get_head_pokes_from_summary( bgvidindex, pixpermm, IN_OR_OUT, Lawn_Boundary_Pts_x, Lawn_Boundary_Pts_y, Lawn_Boundary_Dist, Head_cent, Midbody_cent, forward, reverse, speed )
%GET_HEAD_POKES_FROM_SUMMARY.m This function identifies times that the worm pokes its
%head out of the lawn while the rest of the body remains in the lawn.
%This method looks specifically at peaks in radial distance to define all
%headpokes and also demarcates a subcategory, headpoke-reversals, in which
%the headpoke is coupled to a backwards movement.

%thresholds
min_rd = (-5/pixpermm); %minimum distance from the lawn boundary = 5/112*pixpermm
min_prom = (5/pixpermm); %minimum peak prominence in the radial distance (default = 5/112)
min_wid = 0;
time_tol = 8; %tolerance for peak overlap (2.67 seconds)
speed_thresh = 0.02;

min_ED_neg = min(-1*Lawn_Boundary_Dist);
ED_neg_shifted = (-1*Lawn_Boundary_Dist)-(min(-1*Lawn_Boundary_Dist));
min_rd_shifted = min_rd-min_ED_neg;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EVENT HORIZON DISTANCE METHOD - look for negative peaks in event horizon distance
[peak_height,peak_centers,~,peak_prom, borders] = findpeaks_Elias(ED_neg_shifted,'MinPeakHeight',min_rd_shifted,'MinPeakProminence',min_prom,'MinPeakWidth',min_wid,'WidthReference','halfheight');
[rd_peaks_ALL, rd_peak_intervals_ALL, rd_peak_height_ALL] = refine_peak_borders(ED_neg_shifted, peak_centers, borders, peak_prom, peak_height);
rd_peak_height_ALL = rd_peak_height_ALL+min_ED_neg; %shift peak heights back to negative inside and positive outside the lawn.

% REMOVE POTENTIAL POKE INTERVALS THAT AREN'T LONG ENOUGH TO DO AN ANGLE CALCULATION. (THEY ARE NOT RELIABLE)
pokelen = rd_peak_intervals_ALL(:,2)-rd_peak_intervals_ALL(:,1);
to_remove = pokelen<2;
rd_peaks_ALL(to_remove) = [];
rd_peak_intervals_ALL(to_remove,:) = [];
rd_peak_height_ALL(to_remove) = [];

left_flank_ALL = rd_peak_intervals_ALL(:,1); %beginning of head poke is the start of radial excursion outside the lawn
right_flank_ALL = rd_peak_intervals_ALL(:,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOOK FOR FORWARD MOVEMENT, REVERSALS AND PAUSES that follow head pokes

reversal_ints = get_intervals( reverse, 1 ); %get intervals when worm was reversing
if ~isempty(reversal_ints)
    reversal_starts = reversal_ints(:,1);
    no_reversals = false;
else
    reversal_starts = [];
    no_reversals = true;
end
%define when the animal is pausing as when the absolute speed falls below threshold.
pausing = abs(speed)<speed_thresh;

%CATEGORIZE HEAD POKES
if no_reversals
    idx_rev_select = [];
else
    [~, idx_rev_select] = findClosestToPeak(reversal_starts,rd_peaks_ALL,time_tol);
end

%remove intervals when the animal is outside the lawn - make a banned
%intervals variable.
banned_intervals = get_intervals(IN_OR_OUT,0);
if isempty(banned_intervals)
    banned_intervals = [-10 -10]; %an interval that will never be a problem
end

%initialize measures
HeadPokeIntervals = false(length(speed),1);
HeadPokesAll = false(length(speed),1);
HeadPokeFwd = false(length(speed),1);
HeadPokeRev = false(length(speed),1);
HeadPokePause = false(length(speed),1);
HeadPokeDist = NaN(length(speed),1);
HeadPokeAngle = NaN(length(speed),1);
HeadPokeSpeed = NaN(length(speed),1);

for idx = 1:length(rd_peaks_ALL)
    pokeforward = false; pokerev = false; pokepause = false; %initialize
    potentialhp = rd_peaks_ALL(idx);
    if isempty(left_flank_ALL(idx)) || isempty(right_flank_ALL(idx)) %don't include peaks at the beginning or end of a track, they can be unreliable
        continue;
    else
        %ADD A NEW POKE
        if ~logical(sum(potentialhp>banned_intervals(:,1) & potentialhp<banned_intervals(:,2))) %&& track.tailinlawn(potentialhp) %check that this head poke is not in one of the banned intervals (when the worm is OUT OF THE LAWN) AND that the tail is still in the lawn at the peak of the headpoke (actually getting rid of this stipulation)
%             disp(['NEW POKE idx = ' num2str(idx)]);
            
            HeadPokeIntervals(left_flank_ALL(idx):right_flank_ALL(idx))=1; %put 1s wherever there are headpoke intervals.
            HeadPokesAll(potentialhp)=1; %add this peak of poke to the HEAD_POKES vector
         
            HeadPokeDist(potentialhp) = rd_peak_height_ALL(idx); %add headpoke distance (mm)
            
            %get poke approach angle
            eh = [Lawn_Boundary_Pts_x(bgvidindex(potentialhp),:)' Lawn_Boundary_Pts_y(bgvidindex(potentialhp),:)'];
            angle = getpokeapproachangle(Head_cent,Midbody_cent,eh,[left_flank_ALL(idx) potentialhp]);
            HeadPokeAngle(potentialhp) = angle;
            
            %get poke speed
            HeadPokeSpeed(potentialhp) = nanmean(speed(left_flank_ALL(idx):potentialhp)); %average speed of the radial excursion (until peak)
            
            % CATEGORIZE HEAD POKE BY MOVEMENT AFTER PEAK
            int_after_peak = potentialhp:right_flank_ALL(idx);
            if nanmean(forward(int_after_peak))>0.5
                pokeforward = true;
            elseif nanmean(pausing(int_after_peak))>0.5
                pokepause = true;
            end
            if ismember(idx,idx_rev_select) %this method supersedes the previous 2
                pokerev = true;
                pokeforward = false;
                pokepause = false;
            end
            HeadPokeFwd(potentialhp) = pokeforward;
            HeadPokeRev(potentialhp) = pokerev;
            HeadPokePause(potentialhp) = pokepause;
        end
    end
end

POKE_STRUCT = [];
POKE_STRUCT.HeadPokeIntervals = HeadPokeIntervals;
POKE_STRUCT.HeadPokesAll = HeadPokesAll;
POKE_STRUCT.HeadPokeFwd = HeadPokeFwd;
POKE_STRUCT.HeadPokeRev = HeadPokeRev;
POKE_STRUCT.HeadPokePause = HeadPokePause;
POKE_STRUCT.HeadPokeDist = HeadPokeDist;
POKE_STRUCT.HeadPokeAngle = HeadPokeAngle;
POKE_STRUCT.HeadPokeSpeed = HeadPokeSpeed;
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
% smth_window = 3;
% c_smth = [movmean(cent(:,1),smth_window,'omitnan') movmean(cent(:,2),smth_window,'omitnan')];
poke_cent = cent(poke_int(1):poke_int(end),:);
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
