function POKE_STRUCT = get_HP_during_interval2( POKE_STRUCT, NUMWORMS, FRAMES_IN_LAWN, stat_int )

%PLOT_HP_OVERTIME.m This function takes in a list of head poke events (track
%number, frame number pairs) and plots a head-poking rate per worm per
%minute for 40 minutes after transfer to the assay plate. Also generates a
%single statistic, aggregate HP rate for that entire run.

OK_idx_ALL = POKE_STRUCT.POKE_PEAKS_GLOBAL<stat_int(2) & POKE_STRUCT.POKE_PEAKS_GLOBAL>stat_int(1);
HEADPOKES_ALL = sort(POKE_STRUCT.POKE_PEAKS_GLOBAL(OK_idx_ALL),'ascend');

%ALL HEADPOKES
if ~isempty(HEADPOKES_ALL)
    OK_POKE_DIST = POKE_STRUCT.POKE_DIST_MINSUBTRACT(OK_idx_ALL);
    OK_frames_inlawn = sort(FRAMES_IN_LAWN(FRAMES_IN_LAWN<stat_int(2) & FRAMES_IN_LAWN>stat_int(1)),'ascend');
    DUR_IN_LAWN = length(OK_frames_inlawn)/180; %minutes
    
    HEADPOKES_ALL = HEADPOKES_ALL-stat_int(1); %adjust everyone to the beginning of the interval
    poke_timevec = zeros(1,stat_int(2)-stat_int(1));
    %put a 1 everywhere there was an exit, take care to add the count of
    %non-unique elements
    [c,ia,ic] = unique(HEADPOKES_ALL);
    [count, ~, idxcount] = histcounts(ic,numel(ia));
    idx_unique = count(idxcount)==1;
    poke_timevec(HEADPOKES_ALL(idx_unique))=1; %put a 1 everywhere there was a unique exit ind
    idx_nonunique = count(idxcount)>1;
    nonunique_pokes = unique(c(ic(idx_nonunique)));
    nonunique_poke_counts = count(unique(ic(idx_nonunique)));
    poke_timevec(nonunique_pokes)=nonunique_poke_counts; %put a #counts everywhere there was a non-unique exit ind
    
    POKE_COUNT_OVERTIME_ALL = movsum(poke_timevec,180); %sum in 1 minute windows
    POKE_RATE_STATIC_ALL = length(HEADPOKES_ALL)/(NUMWORMS*DUR_IN_LAWN); %total number of exits per worm per minute over 40 minute window
    AVG_POKE_DIST_ALL = nanmean(OK_POKE_DIST);
else
    POKE_COUNT_OVERTIME_ALL = zeros(1,stat_int(2)-stat_int(1));
    POKE_RATE_STATIC_ALL = 0;
    AVG_POKE_DIST_ALL = 0;

end

%FORWARD HEAD POKES
HEADPOKES_DURING_INTERVAL_FWD = POKE_STRUCT.POKE_PEAKS_GLOBAL(POKE_STRUCT.POKE_IS_FWD);
OK_idx_HEADPOKES_DURING_INTERVAL_FWD = HEADPOKES_DURING_INTERVAL_FWD<stat_int(2) & HEADPOKES_DURING_INTERVAL_FWD>stat_int(1);
HEADPOKES_DURING_INTERVAL_FWD = sort(HEADPOKES_DURING_INTERVAL_FWD(OK_idx_HEADPOKES_DURING_INTERVAL_FWD),'ascend');

if ~isempty(HEADPOKES_DURING_INTERVAL_FWD)
    OK_POKE_DIST = POKE_STRUCT.POKE_DIST_MINSUBTRACT(OK_idx_HEADPOKES_DURING_INTERVAL_FWD);
    OK_frames_inlawn = sort(FRAMES_IN_LAWN(FRAMES_IN_LAWN<stat_int(2) & FRAMES_IN_LAWN>stat_int(1)),'ascend');
    DUR_IN_LAWN = length(OK_frames_inlawn)/180; %minutes
    
    HEADPOKES_DURING_INTERVAL_FWD = HEADPOKES_DURING_INTERVAL_FWD-stat_int(1); %adjust everyone to the beginning of the interval
    poke_timevec = zeros(1,stat_int(2)-stat_int(1));
    %put a 1 everywhere there was an exit, take care to add the count of
    %non-unique elements
    [c,ia,ic] = unique(HEADPOKES_DURING_INTERVAL_FWD);
    [count, ~, idxcount] = histcounts(ic,numel(ia));
    idx_unique = count(idxcount)==1;
    poke_timevec(HEADPOKES_DURING_INTERVAL_FWD(idx_unique))=1; %put a 1 everywhere there was a unique exit ind
    idx_nonunique = count(idxcount)>1;
    nonunique_pokes = unique(c(ic(idx_nonunique)));
    nonunique_poke_counts = count(unique(ic(idx_nonunique)));
    poke_timevec(nonunique_pokes)=nonunique_poke_counts; %put a #counts everywhere there was a non-unique exit ind
    
    POKE_COUNT_OVERTIME_HEADPOKES_DURING_INTERVAL_FWD = movsum(poke_timevec,180); %sum in 1 minute windows
    POKE_RATE_STATIC_HEADPOKES_DURING_INTERVAL_FWD = length(HEADPOKES_DURING_INTERVAL_FWD)/(NUMWORMS*DUR_IN_LAWN); %total number of exits per worm per minute over 40 minute window
    AVG_POKE_DIST_HEADPOKES_DURING_INTERVAL_FWD = nanmean(OK_POKE_DIST);
else
    POKE_COUNT_OVERTIME_HEADPOKES_DURING_INTERVAL_FWD = zeros(1,stat_int(2)-stat_int(1));
    POKE_RATE_STATIC_HEADPOKES_DURING_INTERVAL_FWD = 0;
    AVG_POKE_DIST_HEADPOKES_DURING_INTERVAL_FWD = 0;
end

%HEADPOKE REVERSALS
HEADPOKES_DURING_INTERVAL_REV = POKE_STRUCT.POKE_PEAKS_GLOBAL(POKE_STRUCT.POKE_IS_REV);
OK_idx_HEADPOKES_DURING_INTERVAL_REV = HEADPOKES_DURING_INTERVAL_REV<stat_int(2) & HEADPOKES_DURING_INTERVAL_REV>stat_int(1);
HEADPOKES_DURING_INTERVAL_REV = sort(HEADPOKES_DURING_INTERVAL_REV(OK_idx_HEADPOKES_DURING_INTERVAL_REV),'ascend');

if ~isempty(HEADPOKES_DURING_INTERVAL_REV)
    OK_POKE_DIST = POKE_STRUCT.POKE_DIST_MINSUBTRACT(OK_idx_HEADPOKES_DURING_INTERVAL_REV);
    OK_frames_inlawn = sort(FRAMES_IN_LAWN(FRAMES_IN_LAWN<stat_int(2) & FRAMES_IN_LAWN>stat_int(1)),'ascend');
    DUR_IN_LAWN = length(OK_frames_inlawn)/180; %minutes
    
    HEADPOKES_DURING_INTERVAL_REV = HEADPOKES_DURING_INTERVAL_REV-stat_int(1); %adjust everyone to the beginning of the interval
    poke_timevec = zeros(1,stat_int(2)-stat_int(1));
    %put a 1 everywhere there was an exit, take care to add the count of
    %non-unique elements
    [c,ia,ic] = unique(HEADPOKES_DURING_INTERVAL_REV);
    [count, ~, idxcount] = histcounts(ic,numel(ia));
    idx_unique = count(idxcount)==1;
    poke_timevec(HEADPOKES_DURING_INTERVAL_REV(idx_unique))=1; %put a 1 everywhere there was a unique exit ind
    idx_nonunique = count(idxcount)>1;
    nonunique_pokes = unique(c(ic(idx_nonunique)));
    nonunique_poke_counts = count(unique(ic(idx_nonunique)));
    poke_timevec(nonunique_pokes)=nonunique_poke_counts; %put a #counts everywhere there was a non-unique exit ind
    
    POKE_COUNT_OVERTIME_HEADPOKES_DURING_INTERVAL_REV = movsum(poke_timevec,180); %sum in 1 minute windows
    POKE_RATE_STATIC_HEADPOKES_DURING_INTERVAL_REV = length(HEADPOKES_DURING_INTERVAL_REV)/(NUMWORMS*DUR_IN_LAWN); %total number of exits per worm per minute over 40 minute window
    AVG_POKE_DIST_HEADPOKES_DURING_INTERVAL_REV = nanmean(OK_POKE_DIST);
else
    POKE_COUNT_OVERTIME_HEADPOKES_DURING_INTERVAL_REV = zeros(1,stat_int(2)-stat_int(1));
    POKE_RATE_STATIC_HEADPOKES_DURING_INTERVAL_REV = 0;
    AVG_POKE_DIST_HEADPOKES_DURING_INTERVAL_REV = 0;
end

%HEADPOKE PAUSES
HEADPOKES_DURING_INTERVAL_PAUSE = POKE_STRUCT.POKE_PEAKS_GLOBAL(POKE_STRUCT.POKE_IS_PAUSE);
OK_idx_HEADPOKES_DURING_INTERVAL_PAUSE = HEADPOKES_DURING_INTERVAL_PAUSE<stat_int(2) & HEADPOKES_DURING_INTERVAL_PAUSE>stat_int(1);
HEADPOKES_DURING_INTERVAL_PAUSE = sort(HEADPOKES_DURING_INTERVAL_PAUSE(OK_idx_HEADPOKES_DURING_INTERVAL_PAUSE),'ascend');

if ~isempty(HEADPOKES_DURING_INTERVAL_PAUSE)
    OK_POKE_DIST = POKE_STRUCT.POKE_DIST_MINSUBTRACT(OK_idx_HEADPOKES_DURING_INTERVAL_PAUSE);
    OK_frames_inlawn = sort(FRAMES_IN_LAWN(FRAMES_IN_LAWN<stat_int(2) & FRAMES_IN_LAWN>stat_int(1)),'ascend');
    DUR_IN_LAWN = length(OK_frames_inlawn)/180; %minutes
    
    HEADPOKES_DURING_INTERVAL_PAUSE = HEADPOKES_DURING_INTERVAL_PAUSE-stat_int(1); %adjust everyone to the beginning of the interval
    poke_timevec = zeros(1,stat_int(2)-stat_int(1));
    %put a 1 everywhere there was an exit, take care to add the count of
    %non-unique elements
    [c,ia,ic] = unique(HEADPOKES_DURING_INTERVAL_PAUSE);
    [count, ~, idxcount] = histcounts(ic,numel(ia));
    idx_unique = count(idxcount)==1;
    poke_timevec(HEADPOKES_DURING_INTERVAL_PAUSE(idx_unique))=1; %put a 1 everywhere there was a unique exit ind
    idx_nonunique = count(idxcount)>1;
    nonunique_pokes = unique(c(ic(idx_nonunique)));
    nonunique_poke_counts = count(unique(ic(idx_nonunique)));
    poke_timevec(nonunique_pokes)=nonunique_poke_counts; %put a #counts everywhere there was a non-unique exit ind
    
    POKE_COUNT_OVERTIME_HEADPOKES_DURING_INTERVAL_PAUSE = movsum(poke_timevec,180); %sum in 1 minute windows
    POKE_RATE_STATIC_HEADPOKES_DURING_INTERVAL_PAUSE = length(HEADPOKES_DURING_INTERVAL_PAUSE)/(NUMWORMS*DUR_IN_LAWN); %total number of exits per worm per minute over 40 minute window
    AVG_POKE_DIST_HEADPOKES_DURING_INTERVAL_PAUSE = nanmean(OK_POKE_DIST);
else
    POKE_COUNT_OVERTIME_HEADPOKES_DURING_INTERVAL_PAUSE = zeros(1,stat_int(2)-stat_int(1));
    POKE_RATE_STATIC_HEADPOKES_DURING_INTERVAL_PAUSE = 0;
    AVG_POKE_DIST_HEADPOKES_DURING_INTERVAL_PAUSE = 0;
end

% GET HEAD POKE ANGLE DISTRIBUTIONS DURING INTERVAL FOR FORWARD + REVERSE HEADPOKES
ALL_POKE_APPROACH_ANGLE = POKE_STRUCT.POKE_APPROACH_ANGLE(OK_idx_ALL);
FWD_POKE_APPROACH_ANGLE = POKE_STRUCT.POKE_APPROACH_ANGLE(OK_idx_HEADPOKES_DURING_INTERVAL_FWD);
REV_POKE_APPROACH_ANGLE = POKE_STRUCT.POKE_APPROACH_ANGLE(OK_idx_HEADPOKES_DURING_INTERVAL_REV);
PAUSE_POKE_APPROACH_ANGLE = POKE_STRUCT.POKE_APPROACH_ANGLE(OK_idx_HEADPOKES_DURING_INTERVAL_PAUSE);

% % GET HEAD POKE SPEED DISTRIBUTIONS DURING INTERVAL FOR FORWARD + REVERSE HEADPOKES
ALL_AVG_POKE_SPEED = POKE_STRUCT.AVG_POKE_SPEED(OK_idx_ALL);
FWD_AVG_POKE_SPEED = POKE_STRUCT.AVG_POKE_SPEED(OK_idx_HEADPOKES_DURING_INTERVAL_FWD);
REV_AVG_POKE_SPEED = POKE_STRUCT.AVG_POKE_SPEED(OK_idx_HEADPOKES_DURING_INTERVAL_REV);
PAUSE_AVG_POKE_SPEED = POKE_STRUCT.AVG_POKE_SPEED(OK_idx_HEADPOKES_DURING_INTERVAL_PAUSE);

%SAVE NEW INFORMATION IN POKE_STRUCT
POKE_STRUCT.HEADPOKES_DURING_INTERVAL_ALL = HEADPOKES_ALL;
POKE_STRUCT.POKE_COUNT_OVERTIME_ALL = POKE_COUNT_OVERTIME_ALL;
POKE_STRUCT.POKE_RATE_STATIC_ALL = POKE_RATE_STATIC_ALL;
POKE_STRUCT.AVG_POKE_DIST_ALL = AVG_POKE_DIST_ALL;

POKE_STRUCT.HEADPOKES_DURING_INTERVAL_FWD = HEADPOKES_DURING_INTERVAL_FWD;
POKE_STRUCT.POKE_COUNT_OVERTIME_FWD = POKE_COUNT_OVERTIME_HEADPOKES_DURING_INTERVAL_FWD;
POKE_STRUCT.POKE_RATE_STATIC_FWD = POKE_RATE_STATIC_HEADPOKES_DURING_INTERVAL_FWD;
POKE_STRUCT.AVG_POKE_DIST_ALL_FWD = AVG_POKE_DIST_HEADPOKES_DURING_INTERVAL_FWD;

POKE_STRUCT.HEADPOKES_DURING_INTERVAL_REV = HEADPOKES_DURING_INTERVAL_REV;
POKE_STRUCT.POKE_COUNT_OVERTIME_REV = POKE_COUNT_OVERTIME_HEADPOKES_DURING_INTERVAL_REV;
POKE_STRUCT.POKE_RATE_STATIC_REV = POKE_RATE_STATIC_HEADPOKES_DURING_INTERVAL_REV;
POKE_STRUCT.AVG_POKE_DIST_ALL_REV = AVG_POKE_DIST_HEADPOKES_DURING_INTERVAL_REV;

POKE_STRUCT.HEADPOKES_DURING_INTERVAL_PAUSE = HEADPOKES_DURING_INTERVAL_PAUSE;
POKE_STRUCT.POKE_COUNT_OVERTIME_PAUSE = POKE_COUNT_OVERTIME_HEADPOKES_DURING_INTERVAL_PAUSE;
POKE_STRUCT.POKE_RATE_STATIC_PAUSE = POKE_RATE_STATIC_HEADPOKES_DURING_INTERVAL_PAUSE;
POKE_STRUCT.AVG_POKE_DIST_ALL_PAUSE = AVG_POKE_DIST_HEADPOKES_DURING_INTERVAL_PAUSE;

POKE_STRUCT.ALL_POKE_APPROACH_ANGLE = ALL_POKE_APPROACH_ANGLE;
POKE_STRUCT.FWD_POKE_APPROACH_ANGLE = FWD_POKE_APPROACH_ANGLE;
POKE_STRUCT.REV_POKE_APPROACH_ANGLE = REV_POKE_APPROACH_ANGLE;
POKE_STRUCT.PAUSE_POKE_APPROACH_ANGLE = PAUSE_POKE_APPROACH_ANGLE;

POKE_STRUCT.ALL_AVG_POKE_SPEED = ALL_AVG_POKE_SPEED;
POKE_STRUCT.FWD_AVG_POKE_SPEED = FWD_AVG_POKE_SPEED;
POKE_STRUCT.REV_AVG_POKE_SPEED = REV_AVG_POKE_SPEED;
POKE_STRUCT.PAUSE_AVG_POKE_SPEED = PAUSE_AVG_POKE_SPEED;
end

