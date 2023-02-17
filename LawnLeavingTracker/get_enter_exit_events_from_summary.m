function EXIT_STRUCT = get_enter_exit_events_from_summary( bgvidindex, spline_x, spline_y, Lawn_Boundary_Pts_x, Lawn_Boundary_Pts_y )
%GET_ENTER_EXIT_EVENTS_FROM_SUMMARY.M This function identifies the indices of lawn exits and entries .
%   Also returns a set of durations outside the lawn on a per track basis.
%   Also returns a set of indices surrounding each exit event, which can be
%   used to extract other information corresponding to behavior at these
%   times like speed, position, etc.

splineMissing = sum(isnan(spline_x),2)==size(spline_x,2);

splinePtsInLawn = cell(size(spline_x,1),1);
for i = 1:size(spline_x,1) %loop over time
    currSpline = [spline_x(i,:)' spline_y(i,:)'];
    currLawnBoundary = [Lawn_Boundary_Pts_x(bgvidindex(i),:)' Lawn_Boundary_Pts_y(bgvidindex(i),:)']; %get the event horizon that corresponds to this frame of the video
    splinePtsInLawn(i) = {inpolygon(currSpline(:,1),currSpline(:,2),currLawnBoundary(:,1),currLawnBoundary(:,2))};
end
ALLil = cellfun(@(x) sum(x)==size(x,1),splinePtsInLawn,'UniformOutput',true);
ALLol = cellfun(@(x) sum(x)==0,splinePtsInLawn,'UniformOutput',true);

%remove indices when spline does not exist
ALLil(splineMissing)=false;
ALLol(splineMissing)=false;

% look for lawn entry as paired decrement of ALLol and increment of ALLil,vice versa for lawn exit
inside_to_X = find([0 diff(ALLil)']==-1)'; %all inside --> NOT all inside
X_to_inside = find([0 diff(ALLil)']==1)';  %NOT all inside --> all inside
outside_to_X = find([0 diff(ALLol)']==-1)';%all outside --> NOT all outside
X_to_outside = find([0 diff(ALLol)']==1)'; %NOT all outside --> all outside

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
Lawn_Entry_Idx = cross_inds(cross_inds(:,2)==-1,1); % one before two consecutive upsteps = entering
Lawn_Exit_Idx = cross_inds(cross_inds(:,2)==1,1);% one before two consecutive downsteps = exit
Lawn_Entry = zeros(size(spline_x,1),1); Lawn_Entry(Lawn_Entry_Idx)=1;
Lawn_Exit = zeros(size(spline_x,1),1); Lawn_Exit(Lawn_Exit_Idx)=1;

% IN_OR_OUT_GLOBAL
%0 means IN 1 means OUT
if sum(ALLol) == sum(~splineMissing) %if the worm is out of the lawn for as many frames as the spline exists
    In_Or_Out = ones(size(spline_x,1),1);
else %otherwise, initialize all to 0 and then fill in intervals when the worm is in or out below.
    In_Or_Out = zeros(size(spline_x,1),1); 
end

%FIGURE OUT THE INTERVALS IN AND OUT
if ~(isempty(Lawn_Entry_Idx) && isempty(Lawn_Exit_Idx)) %if there is at least a single enter or exit event in the track, proceed
    tmp_entry = Lawn_Entry_Idx; tmp_exit = Lawn_Exit_Idx;
    if min(Lawn_Entry_Idx)<min(Lawn_Exit_Idx) %worm began outside lawn, then entered, so 1st frame acts as an exit
        tmp_exit = union(1,tmp_exit);
        started_out = 1;
    elseif min(Lawn_Exit_Idx)<min(Lawn_Entry_Idx) %worm began inside lawn, then exited, so 1st frame acts as an entry
        tmp_entry = union(1,tmp_entry);
        started_out = 0;
    elseif isempty(Lawn_Entry_Idx)
        tmp_entry = union(1,tmp_entry);
        started_out = 0;
    elseif isempty(Lawn_Exit_Idx)
        tmp_exit = union(1,tmp_exit);
        started_out = 1;
    else
        error('no other option!');
    end
    if max(Lawn_Entry_Idx)<max(Lawn_Exit_Idx) % the last exit happened after last entry, so end of the track acts as an entry
        tmp_entry = union(tmp_entry, size(spline_x,1));
    elseif max(Lawn_Exit_Idx)<max(Lawn_Entry_Idx) % the last entry happened after the last exit, so end of the track acts as an exit
        tmp_exit = union(tmp_exit, size(spline_x,1));
    elseif isempty(Lawn_Entry_Idx)
        tmp_entry = union(tmp_entry, size(spline_x,1));
    elseif isempty(Lawn_Exit_Idx)
        tmp_exit = union(tmp_exit, size(spline_x,1));
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
        In_Or_Out(alternatingInts(y,1):alternatingInts(y,2))=in_or_out(y);
    end
    
end

EXIT_STRUCT = [];
EXIT_STRUCT.Lawn_Entry = Lawn_Entry;
EXIT_STRUCT.Lawn_Exit = Lawn_Exit;
EXIT_STRUCT.In_Or_Out = In_Or_Out;
