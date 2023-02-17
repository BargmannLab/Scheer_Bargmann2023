function [OK_frames_inlawn, EXIT_STRUCT] = get_LL_during_interval2( EXIT_STRUCT, NUMWORMS, stat_int)
%PLOT_LL_OVERTIME.m This function takes in a list of EXIT events (track
%number, frame number pairs) and plots a lawn-leaving rate per worm per
%minute for 40 minutes after transfer to the assay plate. Also generates a
%single statistic, aggregate LL rate for that entire run.

ENTER_FRAMES = EXIT_STRUCT.ENTER_FRAMES;
EXIT_FRAMES = EXIT_STRUCT.EXIT_FRAMES;
FRAMES_IN_LAWN = EXIT_STRUCT.FRAMES_IN_LAWN;

EXITS_DURING_INTERVAL = sort(EXIT_FRAMES(EXIT_FRAMES<stat_int(2) & EXIT_FRAMES>stat_int(1)),'ascend');
OK_frames_inlawn = sort(FRAMES_IN_LAWN(FRAMES_IN_LAWN<stat_int(2) & FRAMES_IN_LAWN>stat_int(1)),'ascend');
if ~isempty(EXITS_DURING_INTERVAL)
    DUR_IN_LAWN = length(OK_frames_inlawn)/180; %minutes (@ 3fps)
    EXITS_DURING_INTERVAL = EXITS_DURING_INTERVAL-stat_int(1); %adjust everyone to the beginning of the interval
    EXIT_timevec = zeros(1,stat_int(2)-stat_int(1));
    %put a 1 everywhere there was an EXIT, take care to add the count of
    %non-unique elements
    [c,ia,ic] = unique(EXITS_DURING_INTERVAL);
    [count, ~, idxcount] = histcounts(ic,numel(ia));
    idx_unique = count(idxcount)==1;
    EXIT_timevec(EXITS_DURING_INTERVAL(idx_unique))=1; %put a 1 everywhere there was a unique EXIT ind
    idx_nonunique = count(idxcount)>1;
    nonunique_EXITS = unique(c(ic(idx_nonunique)));
    nonunique_EXIT_counts = count(unique(ic(idx_nonunique)));
    EXIT_timevec(nonunique_EXITS)=nonunique_EXIT_counts; %put a #counts everywhere there was a non-unique EXIT ind
    
    EXIT_COUNT_OVERTIME = movsum(EXIT_timevec,180); %sum in 1 minute windows
    EXIT_RATE_STATIC = length(EXITS_DURING_INTERVAL)/(NUMWORMS*DUR_IN_LAWN); %total number of EXITS per worm per minute over 40 minute window
else
    EXIT_COUNT_OVERTIME = zeros(1,stat_int(2)-stat_int(1));
    EXIT_RATE_STATIC = 0;
end


ENTERS_DURING_INTERVAL = sort(ENTER_FRAMES(ENTER_FRAMES<stat_int(2) & ENTER_FRAMES>stat_int(1)),'ascend');
if ~isempty(ENTERS_DURING_INTERVAL)
    DUR_IN_LAWN = length(OK_frames_inlawn)/180; %minutes
    ENTERS_DURING_INTERVAL = ENTERS_DURING_INTERVAL-stat_int(1); %adjust everyone to the beginning of the interval
    ENTER_timevec = zeros(1,stat_int(2)-stat_int(1));
    %put a 1 everywhere there was an ENTER, take care to add the count of
    %non-unique elements
    [c,ia,ic] = unique(ENTERS_DURING_INTERVAL);
    [count, ~, idxcount] = histcounts(ic,numel(ia));
    idx_unique = count(idxcount)==1;
    ENTER_timevec(ENTERS_DURING_INTERVAL(idx_unique))=1; %put a 1 everywhere there was a unique ENTER ind
    idx_nonunique = count(idxcount)>1;
    nonunique_ENTERS = unique(c(ic(idx_nonunique)));
    nonunique_ENTER_counts = count(unique(ic(idx_nonunique)));
    ENTER_timevec(nonunique_ENTERS)=nonunique_ENTER_counts; %put a #counts everywhere there was a non-unique ENTER ind
    
    ENTER_COUNT_OVERTIME = movsum(ENTER_timevec,180); %sum in 1 minute windows
    ENTER_RATE_STATIC = length(ENTERS_DURING_INTERVAL)/(NUMWORMS*DUR_IN_LAWN); %total number of ENTERS per worm per minute over 40 minute window
else
    ENTER_COUNT_OVERTIME = zeros(1,stat_int(2)-stat_int(1));
    ENTER_RATE_STATIC = 0;

end

EXIT_STRUCT.EXITS_DURING_INTERVAL = EXITS_DURING_INTERVAL;
EXIT_STRUCT.EXIT_COUNT_OVERTIME = EXIT_COUNT_OVERTIME;
EXIT_STRUCT.EXIT_RATE_STATIC = EXIT_RATE_STATIC;
EXIT_STRUCT.ENTERS_DURING_INTERVAL = ENTERS_DURING_INTERVAL;
EXIT_STRUCT.ENTER_COUNT_OVERTIME = ENTER_COUNT_OVERTIME;
EXIT_STRUCT.ENTER_RATE_STATIC = ENTER_RATE_STATIC;
end

