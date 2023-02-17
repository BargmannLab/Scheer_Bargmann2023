function [TRACKS, EXIT_STRUCT, POKE_STRUCT, SUMMARY_STRUCT] = tracks_postprocessing6( allTracks, ev_ho_dict, bg_struct, pixpermm, titlestr)
%TRACKS_POSTPROCESSING.m This function takes in the current tracks and the
%tracks that finished throughout the video and extracts a few more
%behavioral parameters from them.

%ROAMING AND DWELLING
%     trans = [0.995, 0.005; 0.07, 0.93]; %for HMM (Steve's numbers)
%     emis = [0.96, 0.04; 0.07, 0.93];
%     cutoff = 450;

%my numbers from N2 OD2
trans = [0.9645    0.0355; 0.0802    0.9198];
emis =  [0.9790    0.0210; 0.5448    0.4552];
cutoff = 35;
x_offset = 2.5;
binSize = 10*3;%10 seconds = 30 frames
train = false;

TRACKS = struct(...
    'id', {}, ...
    'bgvidindex',{},...
    'x_shift',{},...
    'y_shift',{},...
    'centroid', {}, ...
    'bbox', {}, ...
    'cropworm',{}, ...
    'cropworm_orig',{},...
    'spline',{},...
    'curvature',{},...
    'posture_angle',{},...
    'head',{},...
    'head_smooth',{},...
    'tail',{},...
    'tail_smooth',{},...
    'end1',{},...
    'end2',{},...
    'centroid_smooth',{},...
    'speed',{},...
    'speed_smooth',{},...
    'msd',{},...
    'msd_smooth',{},...
    'angspeed',{},...
    'actualratio',{},...
    'roamdwell_hmm',{},...
    'roamdwell_2d',{},...
    'head_gs',{},...
    'tail_gs',{},...
    'age', {}, ...
    'totalVisibleCount', {}, ...
    'consecutiveInvisibleCount', {}, ...
    'framesActive', {},...
    'videoname',{},...
    'videoframe',{},...
    'omega',{},...
    'forward',{},...
    'reverse',{},...
    'fullyinlawn',{},...
    'headinlawn',{},...
    'centroidinlawn',{},...
    'tailinlawn',{});

outTrkIdx = 1;
for i = 1:length(allTracks)
    disp(['curr track is ' num2str(i)]);
    % occasionally you end up with different number of centroid positions
    % than the age of the track and this should be fixed in the main
    % tracking code, but also check for it here and fix it.
    age = size(allTracks(i).centroid,1);
    if allTracks(i).age < age
        age = allTracks(i).age;
    end
    %also check for tracks that end by skipping to a new location far from
    %where they just were -- these final observations should be cut off.
    A = allTracks(i).centroid(1:age,:);
    dist = sqrt( sum( abs( diff( A ) ).^2, 2 ) );
    if dist(end)>mean(dist(1:end-1))+2*std(dist(1:end-1)) %if the track moves too far at the last frame, cut it back
        age = age-1;
    end
    %%% copy most fields directly to TRACKS
    TRACKS(outTrkIdx).id = allTracks(i).id;
    TRACKS(outTrkIdx).x_shift = allTracks(i).x_shift(1:age);
    TRACKS(outTrkIdx).y_shift = allTracks(i).y_shift(1:age);
    TRACKS(outTrkIdx).centroid = allTracks(i).centroid(1:age,:);
    TRACKS(outTrkIdx).cropworm = allTracks(i).cropworm(1:age);
    TRACKS(outTrkIdx).cropworm_orig = allTracks(i).cropworm_orig(1:age);
    TRACKS(outTrkIdx).bbox = allTracks(i).bbox(1:age,:);
    TRACKS(outTrkIdx).end1 = allTracks(i).end1(1:age,:); %nice to keep track of these in case you need them later
    TRACKS(outTrkIdx).end2 = allTracks(i).end2(1:age,:);
    TRACKS(outTrkIdx).end1_g = allTracks(i).end1_g(1:age);
    TRACKS(outTrkIdx).end2_g = allTracks(i).end2_g(1:age);
    TRACKS(outTrkIdx).omega = allTracks(i).omega(1:age);
    TRACKS(outTrkIdx).spline = allTracks(i).spline(1:age);
    TRACKS(outTrkIdx).curvature = allTracks(i).curvature(1:age);
    TRACKS(outTrkIdx).posture_angle = allTracks(i).posture_angle(1:age);
    TRACKS(outTrkIdx).age = age;
    TRACKS(outTrkIdx).consecutiveInvisibleCount = allTracks(i).consecutiveInvisibleCount;
    TRACKS(outTrkIdx).totalVisibleCount = age;
    TRACKS(outTrkIdx).framesActive = allTracks(i).framesActive(1:age);
    TRACKS(outTrkIdx).videoname = allTracks(i).videoname(1:age);
    TRACKS(outTrkIdx).videoframe = allTracks(i).videoframe(1:age);
    
    frames = TRACKS(outTrkIdx).framesActive;
    
    %to make this code extensible to previously tracked videos where
    %ev_ho_dict was not retained, check to see if the tracks already have
    %this field
    if ~isfield(allTracks(i),'bgvidindex')
        bgvidindex = ev_ho_dict(frames);
        TRACKS(outTrkIdx).bgvidindex = bgvidindex(1:age); %this can be used to index indto the bg_struct to access any of that information on a per frame basis
    else
        TRACKS(outTrkIdx).bgvidindex = allTracks(i).bgvidindex;
    end
    %%%
    %SMOOTH CENTROID, GET MEAN SQUARED DISPLACEMENT
    centroid = TRACKS(outTrkIdx).centroid;
    centroid_smooth = [movmean(centroid(:,1),3,'omitnan') movmean(centroid(:,2),3,'omitnan')];
    TRACKS(outTrkIdx).centroid_smooth = centroid_smooth;
    
    [msd, msd_smth] = get_msd( centroid_smooth, pixpermm, 3 );
    if nanmean(msd)<=1e-5 %if the msd is less than threshold, delete this track and move on. it is most likely dust, etc.
        TRACKS(outTrkIdx) = []; %dont increment outTrkIdx, so you re-use this index.
        continue;
    end
    TRACKS(outTrkIdx).msd = msd;
    TRACKS(outTrkIdx).msd_smooth = msd_smth;
    
    %GET SPEED AND ANGULAR SPEED %updated 01/28/2020
%     set1 = centroid_smooth(1:end-3,:);
%     set2 = centroid_smooth(4:end,:); %1 second ahead for calculation
%     speed = [NaN(3,1); diag(pdist2(set1,set2))./pixpermm]; % D is the displacement between successive centroid positions in millimeters
%     speed(1:3)=speed(4); %fill backward with the first speed value calculated.
%     if i==8
%         disp('debug');
%     end
    % Navin method %updated 01/28/2020
    fps = 3;
    StepSize = 3;
    Xdif = CalcDif(centroid_smooth(:,1), StepSize) * fps;
    Ydif = -CalcDif(centroid_smooth(:,2), StepSize) * fps;
    speed = (sqrt(Xdif.^2 + Ydif.^2)./pixpermm)';
    
	angspeed = getAngularSpeed_NavinMethod(centroid_smooth); %as of 2/27/19
    TRACKS(outTrkIdx).angspeed = angspeed;
    
    %IDENTIFY HEAD AND TAIL, GET GRAYSCALE ASSOCIATED WITH THEM, THEN SMOOTH THEM, TOO
    TRACKS(outTrkIdx) = head_and_tail_kalman5(TRACKS(outTrkIdx), pixpermm);
    
    head_smooth = [movmean(TRACKS(outTrkIdx).head(:,1),3,'omitnan') movmean(TRACKS(outTrkIdx).head(:,2),3,'omitnan')];
    TRACKS(outTrkIdx).head_smooth = head_smooth;
    
    tail_smooth = [movmean(TRACKS(outTrkIdx).tail(:,1),3,'omitnan') movmean(TRACKS(outTrkIdx).tail(:,2),3,'omitnan')];
    TRACKS(outTrkIdx).tail_smooth = tail_smooth;
    
    %EXTRACT BOUTS OF FORWARD AND REVERSE MOVEMENT
    coherence_thresh = 90; %in degrees -- this requires that centroid vector and head or tail vector must be within this angle range of each other to be considered coherent motion
    speed_thresh = 0.02; %mm/sec -- this is the required speed to be considered moving
    [~, ~, ~, speed, speed_smooth, forward, reverse] = getforwardreverse2(TRACKS(outTrkIdx).centroid,TRACKS(outTrkIdx).head,TRACKS(outTrkIdx).tail,speed,coherence_thresh, speed_thresh);
    TRACKS(outTrkIdx).speed = speed; % (mm/sec) absolute value of the velocity (scale this per 1/3 second = 3fps)
    TRACKS(outTrkIdx).speed_smooth = speed_smooth;
    TRACKS(outTrkIdx).forward = forward;
    TRACKS(outTrkIdx).reverse = reverse;
    
    %DETERMINE WHEN ANIMAL WAS IN LAWN
    [headinlawn, centroidinlawn, tailinlawn, fullyinlawn] = countBlobsInOut(TRACKS(outTrkIdx), bg_struct);
    TRACKS(outTrkIdx).headinlawn = headinlawn;
    TRACKS(outTrkIdx).centroidinlawn = centroidinlawn;
    TRACKS(outTrkIdx).tailinlawn = tailinlawn;
    TRACKS(outTrkIdx).fullyinlawn = fullyinlawn;
    
    if TRACKS(outTrkIdx).age>=binSize %if the track length is longer than binSize, categories roaming and dwelling states
        [expSeq, expStates, ~, ~, actualratio] = getHMMStates3(TRACKS(outTrkIdx),binSize,TRACKS(outTrkIdx).centroidinlawn,cutoff,trans,emis,x_offset,train);
        TRACKS(outTrkIdx).roamdwell_2d = expSeq;
        TRACKS(outTrkIdx).roamdwell_hmm = expStates;
        TRACKS(outTrkIdx).actualratio = actualratio;
    else
        TRACKS(outTrkIdx).roamdwell_2d = NaN(TRACKS(outTrkIdx).age,1);
        TRACKS(outTrkIdx).roamdwell_hmm = NaN(TRACKS(outTrkIdx).age,1);
        TRACKS(outTrkIdx).actualratio = NaN(TRACKS(outTrkIdx).age,1);
    end
    outTrkIdx = outTrkIdx+1;
end

% % if there are multiple tracks existing during the same frames, figure out
% % which one to keep.
% 
% % *) assume multiple coexisting tracks comes from dust or some other object
% % -- not another worm.
% 
% % find tracks with intersecting frame lists.
% fa = {TRACKS.framesActive}';
% intersectingFrames = false(length(TRACKS),length(TRACKS));
% a = 1:length(TRACKS);
% b = a;
% [A,B] = meshgrid(a,b);
% c=cat(2,A',B');
% d=reshape(c,[],2);
% d = unique(sort(d,2), 'rows'); %remove permutations
% d(d(:,1)==d(:,2),:) = []; %remove self-combinations
% for comboIdx = 1:length(d)
%     idx = d(comboIdx,:);
%     i = idx(1); j = idx(2);
%     intersectingFrames(i,j)=~isempty(intersect(fa{i},fa{j}));
% end
% [track1idx,track2idx] = find(intersectingFrames);
% for i = length(track1idx) %for each conflict (where two tracks are active at the same time), adjudicate which one is correct
%     [~,i1,i2]=intersect(fa{track1idx(i)},fa{track2idx(i)});
%     %if tracks overlap on one frame, check if it is the last frame of the
%     %first and the first frame of the second -- if so, trim each back one.
%     if length(i1)==1
%         if i1==length(fa{track1idx(i)}) && i2==1
%             %code to trim back track1 and trim forward track2
%         else %otherwise ...?
%             
%         end
%     end
% end

% % REMOVE dust or other objects that don't move
% % USE MEAN SQUARED DISPLACEMENT
% msds = {TRACKS.msd}';
% avg_msd = cellfun(@nanmean,msds);
% TRACKS = TRACKS(avg_msd>1e-5); %remove any tracks with inordinately low MSD

%CURRENT ANALYSIS PARAMETERS AS OF FEB 2019 -- consider relegating to a
%.mat file to make this code more flexible

before_sec = 300; %5 minutes before
after_sec = 60; %1 minute after

startmin = 20; endmin = 60;
NUMWORMS = 1;
stat_int = [startmin*60*3 endmin*60*3]; %frames between 20 minutes and 60 minutes since the start of the assay
% tracklen = stat_int(2)-stat_int(1); %tracklen
% curr_dir = pwd;
% cd('..');
% str = pwd ;
% idx = strfind(str,'\') ;
% foldername = str(idx(end)+1:end) ;
% cd(curr_dir);

%ENTER AND EXIT EVENTS, HEAD POKES
[TRACKS, EXIT_STRUCT] = get_enter_exit_events_SPLINEMETHOD2( TRACKS, stat_int, before_sec, after_sec, bg_struct);

[TRACKS, POKE_STRUCT] = get_head_pokes_PERMISSIVE( TRACKS, bg_struct, EXIT_STRUCT.INTS_OUT_BY_TRACK, EXIT_STRUCT.OUT_INT_TRK_KEY, pixpermm );

%get LAWN LEAVING events during the acceptable interval, compute statistics
[OK_frames_inlawn, EXIT_STRUCT] = get_LL_during_interval2( EXIT_STRUCT, NUMWORMS, stat_int);

%get HEAD POKE events during the acceptable interval, compute statistics
%(now with a new category : HP+FWD, HP+REV, HP+PAUSE, ANGLES of APPROACH)
POKE_STRUCT = get_HP_during_interval2( POKE_STRUCT, NUMWORMS, EXIT_STRUCT.FRAMES_IN_LAWN, stat_int );

% SUMMARIZE ALL TRACKING DATA during the acceptable interval, compute statistics
SUMMARY_STRUCT = summarize_data_during_interval( TRACKS, EXIT_STRUCT, stat_int, OK_frames_inlawn, pixpermm, binSize, cutoff, trans, emis, x_offset);

if ~isempty(SUMMARY_STRUCT)
    % MAKE SUMMARY PLOT
    makeSummaryFig(SUMMARY_STRUCT,EXIT_STRUCT,POKE_STRUCT,stat_int,titlestr);
end

end

