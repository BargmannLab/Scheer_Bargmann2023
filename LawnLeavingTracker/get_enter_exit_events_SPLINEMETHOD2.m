function [TRACKS, EXIT_STRUCT] = get_enter_exit_events_SPLINEMETHOD2( TRACKS, stat_int, before_sec, after_sec , bg_struct)
%GET_ENTER_EXIT_EVENTS_SPLINEMETHOD.M This function takes in a set of tracks and
%identifies the indices of lawn exits and entries .
%   Also returns a set of durations outside the lawn on a per track basis.
%   Also returns a set of indices surrounding each exit event, which can be
%   used to extract other information corresponding to behavior at these
%   times like speed, position, etc.

FRAMES_IN_LAWN = [];%get the frames in which the animal was in the lawn expressed in overall frame indices
EXIT_INTS = [];
INTS_IN_BY_TRACK = [];
INTS_OUT_BY_TRACK = [];
INTS_IN_GLOBAL = [];
INTS_OUT_GLOBAL = [];
IN_INT_TRK_KEY = [];
OUT_INT_TRK_KEY = [];

tracknum = NaN(stat_int(2),1);
trackidx = NaN(stat_int(2),1);
ALLil = false(stat_int(2),1);
ALLol = false(stat_int(2),1);
HIL = false(stat_int(2),1);
splineExists = false(stat_int(2),1);
trackExists = false(stat_int(2),1);

for trk = 1:length(TRACKS)
    track = TRACKS(trk);
    [OK_trk_idx, OK_int_idx] = ismember(track.framesActive,1:stat_int(2));
    OK_trk_idx = find(OK_trk_idx); %#ok<EFIND>
    OK_int_idx = OK_int_idx(OK_int_idx~=0);
    
    tracknum(OK_int_idx)= repmat(trk,length(OK_trk_idx),1); %keep track of track numbers and indices
    trackidx(OK_int_idx)= OK_trk_idx;
    trackExists(OK_int_idx) = true;
    
    %initialize lawn entries and exits per track, to be filled in later
    TRACKS(trk).lawn_entries = false(track.age,1);
    TRACKS(trk).lawn_exits = false(track.age,1);
    
    if ~isempty(OK_trk_idx)
        bgvidindex = track.bgvidindex;
        
        % preserve old functionality
        cil = track.centroidinlawn;
        FRAMES_IN_LAWN = [FRAMES_IN_LAWN; find(cil)+track.framesActive(1)-1];
        
        % new method using entire spline 01/15/19
        spline = track.spline;
        splineDoesNotExist = cellfun(@(x) isequal(size(x),[1 1]),spline,'UniformOutput',true);
        spline(splineDoesNotExist) = deal({[NaN NaN]});
        
        % catalog when the spline exists or not.
        sE = ~(splineDoesNotExist(OK_trk_idx));
        splineExists(OK_int_idx) = sE;
        
        splinePtsInLawn = cell(size(spline));
        for i = 1:track.age
            ev_ho = bg_struct(bgvidindex(i)).ev_ho_crp_rel; %get the event horizon that corresponds to this frame of the video
            currspline = spline{i};
            splinePtsInLawn(i) = {inpolygon(currspline(:,1),currspline(:,2),ev_ho(:,1),ev_ho(:,2))};
        end
        
        inlawn = cellfun(@(x) sum(x)==size(x,1),splinePtsInLawn,'UniformOutput',true);
        ALLil(OK_int_idx) = inlawn(OK_trk_idx);
        outlawn = cellfun(@(x) sum(x)==0,splinePtsInLawn,'UniformOutput',true);
        ALLol(OK_int_idx) = outlawn(OK_trk_idx);
        
        % catalog head in and out events
        hil = track.headinlawn;
        HIL(OK_int_idx) = hil(OK_trk_idx);
    end
end

%remove indices when spline does not exist
ALLil(~splineExists)=false;
ALLol(~splineExists)=false;

% look for lawn entry as paired decrement of ALLol and increment of ALLil,
%vice versa for lawn exit
inside_to_X = find([0 diff(ALLil)']==-1)'; %all inside --> NOT all inside
X_to_inside = find([0 diff(ALLil)']==1)';  %NOT all inside --> all inside
outside_to_X = find([0 diff(ALLol)']==-1)';%all outside --> NOT all outside
X_to_outside = find([0 diff(ALLol)']==1)'; %NOT all outside --> all outside

% % mark indices where the head position marks the transition of inside_to_X
% % and outside_to_X
% head_ItoX_idx = ismember(inside_to_X,find(~HIL));
% head_ItoX = inside_to_X(head_ItoX_idx); %these are potential HEAD-LAWN-LEAVING events
% head_OtoX_idx = ismember(outside_to_X,find(HIL));
% head_OtoX = outside_to_X(head_OtoX_idx); %these are potential HEAD-LAWN-ENTRY events

% by convention, we refer to leaving events as +1 and entering as -1
glom =  [[inside_to_X ones(size(inside_to_X))];...
    [X_to_outside ones(size(X_to_outside))];
    [outside_to_X -1*ones(size(outside_to_X))];
    [X_to_inside -1*ones(size(X_to_inside))]];

glom = sortrows(glom,1);
consec = [1;diff(glom(:,2))]==0;
consec = find(consec)-1;
cross_inds = [glom(consec,1) glom(consec,2)]; %look for consecutive upsteps and downsteps

% GET THE INDICES OF LAWN ENTRY AND EXIT (GLOBAL INDEX)
enter_ind_GLOBAL = cross_inds(cross_inds(:,2)==-1,1); % one before two consecutive upsteps = entering
exit_ind_GLOBAL = cross_inds(cross_inds(:,2)==1,1);% one before two consecutive downsteps = exit

% GET ENTER AND EXIT DATA IN TRACK INDICES AND ADD DATA BACK TO TRACKS THEMSELVES.
ENTER_FRAMES_TRK = zeros(size(enter_ind_GLOBAL,1),2);
for idx = 1:length(enter_ind_GLOBAL)
    tnum = tracknum(enter_ind_GLOBAL(idx));
    tidx = trackidx(enter_ind_GLOBAL(idx));
    if isnan(tnum) || isnan(tidx) %sometimes the exit frame is not in a track. catch this error.
        ENTER_FRAMES_TRK(idx,1) = NaN;
        ENTER_FRAMES_TRK(idx,2) = NaN;
    else
        ENTER_FRAMES_TRK(idx,1)=tnum;
        ENTER_FRAMES_TRK(idx,2)=tidx;
        TRACKS(tnum).lawn_entries(tidx) = true;
    end
end
EXIT_FRAMES_TRK = zeros(size(exit_ind_GLOBAL,1),2);
for idx = 1:length(exit_ind_GLOBAL)
    tnum = tracknum(exit_ind_GLOBAL(idx));
    tidx = trackidx(exit_ind_GLOBAL(idx));
    if isnan(tnum) || isnan(tidx) %sometimes the exit frame is not in a track. catch this error.
        EXIT_FRAMES_TRK(idx,1) = NaN;
        EXIT_FRAMES_TRK(idx,2) = NaN;
    else
        EXIT_FRAMES_TRK(idx,1)=tnum;
        EXIT_FRAMES_TRK(idx,2)=tidx;
        TRACKS(tnum).lawn_exits(tidx) = true;
    end
end

% GET INTERVALS SURROUNDING LAWN-LEAVING EVENTS
after = (after_sec*3)-1; 
before = (before_sec*3);
ALIGNED_EXIT_IND = before+1; %the exit_ind sits at index 136
chopped_exits = [exit_ind_GLOBAL-before exit_ind_GLOBAL+after];

%abbreviate these intervals at the ends of tracks and whenever another
%entry or exit event is encountered -- pad with NaNs in these indices.
if ~isempty(chopped_exits)
    for i = 1:size(chopped_exits,1)
        ext = exit_ind_GLOBAL(i);
        int = chopped_exits(i,1):chopped_exits(i,2);
        inrange = int(int>0 & int<=stat_int(2));
        abbrev = inrange;
        badboys = sort(intersect(union(setdiff(exit_ind_GLOBAL,ext),enter_ind_GLOBAL),inrange)); %check if these intervals run into subsequent lawn leaving or entering events or the end or beginning of the track
        if ~isempty(badboys)
            upperlim = min(badboys(badboys>ext))-1;
            lowerlim = max(badboys(badboys<ext))+1;
            if ~isempty(upperlim)
                abbrev = abbrev(1):upperlim;
            end
            if ~isempty(lowerlim)
                abbrev = lowerlim:abbrev(end);
            end
        end
        intIdx = ismember(int,abbrev); %replace out of bounds indices from int with NaNs, then add it
        trk_int = NaN(size(int));
        trk_int(intIdx) = trackidx(abbrev);
        EXIT_INTS = [EXIT_INTS; trk_int];
    end
end

EXIT_INT_TRK_KEY = EXIT_FRAMES_TRK(:,2);
ENTER_FRAMES = enter_ind_GLOBAL;
EXIT_FRAMES = exit_ind_GLOBAL; %expressed in terms of total frames in the video

% IN_OR_OUT_GLOBAL
%0 means IN 1 means OUT -- this logical vector goes over the entire video 1-stat_int(2)
if sum(ALLol) == sum(splineExists) %if the worm is out of the lawn for as many frames as the spline exists
    IN_OR_OUT_GLOBAL = ones(size(tracknum));
else %otherwise, initialize all to 0 and then fill in intervals when the worm is in or out below.
    IN_OR_OUT_GLOBAL = zeros(size(tracknum)); 
end

%FIGURE OUT THE INTERVALS IN AND OUT
if ~(isempty(enter_ind_GLOBAL) && isempty(exit_ind_GLOBAL)) %if there is at least a single enter or exit event in the track, proceed
    tmp_entry = enter_ind_GLOBAL; tmp_exit = exit_ind_GLOBAL;
    if min(enter_ind_GLOBAL)<min(exit_ind_GLOBAL) %worm began outside lawn, then entered, so 1st frame acts as an exit
        tmp_exit = union(1,tmp_exit);
        started_out = 1;
    elseif min(exit_ind_GLOBAL)<min(enter_ind_GLOBAL) %worm began inside lawn, then exited, so 1st frame acts as an entry
        tmp_entry = union(1,tmp_entry);
        started_out = 0;
    elseif isempty(enter_ind_GLOBAL)
        tmp_entry = union(1,tmp_entry);
        started_out = 0;
    elseif isempty(exit_ind_GLOBAL)
        tmp_exit = union(1,tmp_exit);
        started_out = 1;
    else
        error('no other option!');
    end
    if max(enter_ind_GLOBAL)<max(exit_ind_GLOBAL) % the last exit happened after last entry, so end of the track acts as an entry
        tmp_entry = union(tmp_entry, stat_int(2));
    elseif max(exit_ind_GLOBAL)<max(enter_ind_GLOBAL) % the last entry happened after the last exit, so end of the track acts as an exit
        tmp_exit = union(tmp_exit, stat_int(2));
    elseif isempty(enter_ind_GLOBAL)
        tmp_entry = union(tmp_entry, stat_int(2));
    elseif isempty(exit_ind_GLOBAL)
        tmp_exit = union(tmp_exit, stat_int(2));
    else
        error('no other option!');
    end
    
    total_events = length(tmp_entry)+length(tmp_exit);
    t = zeros(1,total_events);
    twoways = logical(toeplitz(mod(1:total_events,2),mod(1:2,2))); %column 1 is alternating starting with 1, column 2 with 0, 1 means you're OUT
    if started_out %started out so index 1 means youre out of the lawn
        t(1:2:total_events) = tmp_exit;
        t(2:2:total_events) = tmp_entry;
        in_or_out = twoways(:,1);
    else %started in so index 1 means youre in the lawn
        t(1:2:total_events) = tmp_entry;
        t(2:2:total_events) = tmp_exit;
        in_or_out = twoways(:,2);
    end
    
    alternatingInts = zeros(length(t)-1,2);
    for x = 1:length(t)-1
        alternatingInts(x,1) = t(x);
        alternatingInts(x,2) = t(x+1);
    end
    for y = 1:size(alternatingInts,1)
        IN_OR_OUT_GLOBAL(alternatingInts(y,1):alternatingInts(y,2))=in_or_out(y);
    end
    
    %now, assign bits of each track to IN or OUT intervals
    for trk = 1:length(TRACKS)
        firstframe = find(tracknum==trk,1,'first');
        chunk = IN_OR_OUT_GLOBAL(tracknum==trk);
        
        ints_out = get_intervals(chunk,0); %find intervals when the worm is OUT of the lawn (track idx)
        INTS_OUT_BY_TRACK = [INTS_OUT_BY_TRACK; ints_out];
        OUT_INT_TRK_KEY = [OUT_INT_TRK_KEY; repmat(trk,size(ints_out,1),1)];
        INTS_OUT_GLOBAL = [INTS_OUT_GLOBAL; ints_out+firstframe-1];
        
        ints_in = get_intervals(~chunk,0); %find intervals when the worm is IN the lawn (track idx)
        INTS_IN_BY_TRACK = [INTS_IN_BY_TRACK; ints_in];
        IN_INT_TRK_KEY = [IN_INT_TRK_KEY; repmat(trk,size(ints_out,1),1)];
        INTS_IN_GLOBAL = [INTS_IN_GLOBAL; ints_in+firstframe-1];
    end
end


EXIT_STRUCT.FRAMES_IN_LAWN = FRAMES_IN_LAWN;
EXIT_STRUCT.ENTER_FRAMES = ENTER_FRAMES;
EXIT_STRUCT.EXIT_FRAMES = EXIT_FRAMES;
EXIT_STRUCT.ENTER_FRAMES_TRK = ENTER_FRAMES_TRK; %new 01/31/19
EXIT_STRUCT.EXIT_FRAMES_TRK = EXIT_FRAMES_TRK; %new 01/31/19
EXIT_STRUCT.EXIT_INTS = EXIT_INTS;
EXIT_STRUCT.EXIT_TRK_KEY = EXIT_INT_TRK_KEY;
EXIT_STRUCT.ALIGNED_EXIT_IND = ALIGNED_EXIT_IND;
EXIT_STRUCT.IN_OR_OUT_GLOBAL = IN_OR_OUT_GLOBAL;
EXIT_STRUCT.INTS_IN_BY_TRACK = INTS_IN_BY_TRACK;
EXIT_STRUCT.INTS_OUT_BY_TRACK = INTS_OUT_BY_TRACK;
EXIT_STRUCT.INTS_IN_GLOBAL = INTS_IN_GLOBAL;
EXIT_STRUCT.INTS_OUT_GLOBAL = INTS_OUT_GLOBAL;
EXIT_STRUCT.IN_INT_TRK_KEY = IN_INT_TRK_KEY;
EXIT_STRUCT.OUT_INT_TRK_KEY = OUT_INT_TRK_KEY;
