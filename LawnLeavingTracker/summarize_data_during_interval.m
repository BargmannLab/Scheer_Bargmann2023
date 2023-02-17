function SUMMARY_STRUCT = summarize_data_during_interval( TRACKS, EXIT_STRUCT, stat_int, OK_frames_inlawn, pixpermm, binSize, cutoff, trans, emis, x_offset)
%summarize_data_during_interval.m This function collates speed,
%angular speed, and roaming and dwelling information from all tracks and
%selects out the data that falls into the acceptable interval.
tracklen = stat_int(2)-stat_int(1); %usually 7200
%stuff for video playback
TRACKNUM        = NaN(tracklen,1);
TRACKIDX        = NaN(tracklen,1);
BGVIDINDEX      = NaN(tracklen,1);
VIDEONAME       = cell(tracklen,1); %will fill in missing data
VIDEOFRAME      = NaN(tracklen,1);

% animal position
BBOX            = NaN(tracklen,4);
CENTROID        = NaN(tracklen,2);
CENTROID_SMTH   = NaN(tracklen,2);%smoothed
HEAD            = NaN(tracklen,2);
HEAD_SMTH       = NaN(tracklen,2);%smoothed
TAIL            = NaN(tracklen,2);
TAIL_SMTH       = NaN(tracklen,2);%smoothed

%grayscale values for head and tail
HEAD_GS         = NaN(tracklen,1);
TAIL_GS         = NaN(tracklen,1);

%animal segmentation, and derived features
SPLINE_x        = NaN(tracklen,49);
SPLINE_y        = NaN(tracklen,49);
CURVATURE       = NaN(tracklen,49);
POSTURE_ANGLE   = NaN(tracklen,48);% "angle array", which will be used later to compute eigenworms

%animal movement
FORWARD         = false(tracklen,1);
REVERSE         = false(tracklen,1);
OMEGA           = false(tracklen,1);
MSD             = NaN(tracklen,1);%not smoothed
SPEED           = NaN(tracklen,1);%not smoothed
SPEED_SMTH      = NaN(tracklen,1);
ANGSPEED        = NaN(tracklen,1);%not smoothed

%stuff having to do with the lawn
CENTROIDINLAWN  = false(tracklen,1);
HEADINLAWN      = false(tracklen,1);
TAILINLAWN      = false(tracklen,1);
FULLYINLAWN     = false(tracklen,1);
OK_FRAMES_IN_LAWN = false(tracklen,1);
IN_OR_OUT = EXIT_STRUCT.IN_OR_OUT_GLOBAL(stat_int(1)+1:stat_int(2));

%head pokes
HEADPOKE_FWD    = false(tracklen,1);
HEADPOKE_REV    = false(tracklen,1);
HEADPOKE_PAUSE  = false(tracklen,1);

RADIAL_DIST     = NaN(tracklen,1);
EV_HO_DIST      = NaN(tracklen,1);
HEADPOKE_ANGLE  = NaN(tracklen,1);

%lawn leaving / entry
LAWN_EXIT       = false(tracklen,1);
LAWN_ENTRY      = false(tracklen,1);

SPEED_ON_FOOD = NaN(tracklen,1);
SPEED_OFF_FOOD = NaN(tracklen,1);

badTrackCounter = 0;
for trk = 1:length(TRACKS)
    track = TRACKS(trk);
    [OK_trk_idx, OK_int_idx] = ismember(track.framesActive,stat_int(1)+1:stat_int(2)); %have to add +1 to make it tracklen long
    OK_trk_idx = find(OK_trk_idx);
    OK_int_idx = OK_int_idx(OK_int_idx~=0);
    
    if ~isempty(OK_trk_idx)
        TRACKNUM(OK_int_idx)        = repmat(trk,length(OK_trk_idx),1);
        TRACKIDX(OK_int_idx)        = OK_trk_idx;
        
        BGVIDINDEX(OK_int_idx)      = track.bgvidindex(OK_trk_idx,:);
        VIDEONAME(OK_int_idx)       = track.videoname(OK_trk_idx);
        VIDEOFRAME(OK_int_idx)      = track.videoframe(OK_trk_idx);
        
        BBOX(OK_int_idx,:)          = track.bbox(OK_trk_idx,:);
        
        CENTROID(OK_int_idx,:)      = track.centroid(OK_trk_idx,:);
        HEAD(OK_int_idx,:)          = track.head(OK_trk_idx,:);
        TAIL(OK_int_idx,:)          = track.tail(OK_trk_idx,:);
        
        CENTROID_SMTH(OK_int_idx,:)      = track.centroid_smooth(OK_trk_idx,:);
        HEAD_SMTH(OK_int_idx,:)          = track.head_smooth(OK_trk_idx,:);
        TAIL_SMTH(OK_int_idx,:)          = track.tail_smooth(OK_trk_idx,:);
        
        HEAD_GS(OK_int_idx)         = track.head_gs(OK_trk_idx);
        TAIL_GS(OK_int_idx)         = track.tail_gs(OK_trk_idx);
        
        splineDoesNotExist = find(cellfun(@(x) isequal(size(x),[1 1]),track.spline,'UniformOutput',true)); %because of the silly way I have stored single NaNs before
        [spline_trk_idx,ia] = setdiff(OK_trk_idx,splineDoesNotExist);
        spline_int_idx = OK_int_idx(ia); %MUST ALSO REMOVE NaN idx from the int idx!
        tmpspline = cell2mat(track.spline(spline_trk_idx)'); %this makes a 49x(2*length(OK_trk_idx)) size matrix
        SPLINE_x(spline_int_idx,:)      = tmpspline(:,1:2:size(tmpspline,2))';
        SPLINE_y(spline_int_idx,:)      = tmpspline(:,2:2:size(tmpspline,2))';
        CURVATURE(spline_int_idx,:)     = cell2mat(track.curvature(spline_trk_idx)')';%silly transpositions that a necessary
        POSTURE_ANGLE(spline_int_idx,:) = cell2mat(track.posture_angle(spline_trk_idx));
        
        FORWARD(OK_int_idx)         = track.forward(OK_trk_idx);
        REVERSE(OK_int_idx)         = track.reverse(OK_trk_idx);
        OMEGA(OK_int_idx)           = track.omega(OK_trk_idx);
        
        MSD(OK_int_idx)             = track.msd(OK_trk_idx);
        SPEED(OK_int_idx)           = track.speed(OK_trk_idx);
        SPEED_SMTH(OK_int_idx)      = track.speed_smooth(OK_trk_idx);
        ANGSPEED(OK_int_idx)        = track.angspeed(OK_trk_idx);
  
%         %calculate these freshly after concatenating all tracks in the
%         %interval (see below)
%         ROAMDWELL_2D(OK_int_idx)    = track.roamdwell_2d(OK_trk_idx);
%         ROAMDWELL_HMM(OK_int_idx)   = track.roamdwell_hmm(OK_trk_idx);
        
        CENTROIDINLAWN(OK_int_idx)  = track.centroidinlawn(OK_trk_idx);
        HEADINLAWN(OK_int_idx)      = track.headinlawn(OK_trk_idx);
        TAILINLAWN(OK_int_idx)      = track.tailinlawn(OK_trk_idx);
        FULLYINLAWN(OK_int_idx)     = track.fullyinlawn(OK_trk_idx);
        
        HEADPOKE_FWD(OK_int_idx)    = track.head_poke_forward(OK_trk_idx);
        HEADPOKE_REV(OK_int_idx)    = track.head_poke_reversal(OK_trk_idx);
        HEADPOKE_PAUSE(OK_int_idx)  = track.head_poke_pause(OK_trk_idx);
        
        RADIAL_DIST(OK_int_idx)     = track.radial_dist(OK_trk_idx);
        EV_HO_DIST(OK_int_idx)      = track.ev_ho_dist(OK_trk_idx);
        HEADPOKE_ANGLE(OK_int_idx)  = track.head_poke_angle(OK_trk_idx);
        
        LAWN_EXIT(OK_int_idx)       = track.lawn_exits(OK_trk_idx);
        LAWN_ENTRY(OK_int_idx)      = track.lawn_entries(OK_trk_idx);
        
        [OK_trk_inlawn_idx_logical, ~] = ismember(track.framesActive,OK_frames_inlawn); %get the speed when the animal is only ON FOOD
        OK_trk_inlawn_idx = find(OK_trk_inlawn_idx_logical);
        [OK_int_inlawn_idx_logical, ~] = ismember(OK_int_idx,OK_frames_inlawn-stat_int(1));
        OK_int_inlawn_idx = OK_int_idx(OK_int_inlawn_idx_logical);
        if ~isempty(OK_trk_inlawn_idx)
            SPEED_ON_FOOD(OK_int_inlawn_idx) = track.speed(OK_trk_inlawn_idx);
        end
        OK_trk_outlawn_idx = setdiff(OK_trk_idx,OK_trk_inlawn_idx);
        OK_int_outlawn_idx = setdiff(OK_int_idx,OK_int_inlawn_idx);
        if ~isempty(OK_trk_outlawn_idx)
            SPEED_OFF_FOOD(OK_int_outlawn_idx) = track.speed(OK_trk_outlawn_idx);
        end
        
    else
        badTrackCounter = badTrackCounter+1;
        continue;
    end
end
if badTrackCounter == length(TRACKS) %none of the tracks cover the stat_int; generate an empty SUMMARY_STRUCT 
    SUMMARY_STRUCT = [];
    return;
end
%fill in missing data for video playback:
%fill in singleton frames that are not segmented
if sum(isnan(BGVIDINDEX))>0
    x = single(isnan(BGVIDINDEX));
    y = conv(x,[1,1,1],'same');
    singleEmptyIdx = find(x==y & x); %this looks for solo unsegmented frames surrounded by segmented ones
    for k = 1:length(singleEmptyIdx)
        curr_idx = singleEmptyIdx(k);
        prevIdx = curr_idx-1; nextIdx = curr_idx+1;
        if prevIdx<1 %if the unsegmented frame is the first frame, just take the next one
            BGVIDINDEX(curr_idx) = BGVIDINDEX(nextIdx);
            VIDEONAME(curr_idx) = VIDEONAME(nextIdx);
            VIDEOFRAME(curr_idx) = VIDEOFRAME(nextIdx)-1;
        elseif nextIdx>tracklen %if the unsegmented frame is the last frame, just take the previous one
            BGVIDINDEX(curr_idx) = BGVIDINDEX(prevIdx);
            VIDEONAME(curr_idx) = VIDEONAME(prevIdx);
            VIDEOFRAME(curr_idx) = VIDEOFRAME(prevIdx)+1;
        elseif BGVIDINDEX(prevIdx)==BGVIDINDEX(nextIdx) %if the same information is on both sides of the lost information, clone it through
            BGVIDINDEX(curr_idx) = BGVIDINDEX(prevIdx);
            VIDEONAME(curr_idx) = VIDEONAME(prevIdx);
            VIDEOFRAME(curr_idx) = VIDEOFRAME(prevIdx)+1;
        end
        %IMPORTANT: IF the worm is not segmented on the last frame before the
        %video switches (rare, but possible), the NaN will remain. --> deal
        %with this in the playframes_summary.m function.
    end
end
%fill in intervals
if sum(isnan(BGVIDINDEX))>0 && sum(isnan(BGVIDINDEX))<7200
    emptyVidInts = get_intervals(isnan(BGVIDINDEX),1);   
    for k = 1:size(emptyVidInts,1)
        prevIdx = emptyVidInts(k,1)-2; nextIdx = emptyVidInts(k,2)+2;
        if prevIdx<1 %if the interval starts at 1, count backwards from the next segmented frame
            prevIdx = 1; %set it back to 1.
            interval = 1:nextIdx-1;
            BGVIDINDEX(interval) = BGVIDINDEX(nextIdx);
            VIDEONAME(interval) = VIDEONAME(nextIdx);
            nextVidFrame = VIDEOFRAME(nextIdx);
            VIDEOFRAME(interval) = nextVidFrame-length(interval):nextVidFrame-1;
        end
        if nextIdx>tracklen %if the interval ends at tracklen, count forwards from the last segmented frame
            nextIdx = tracklen; %set it back to tracklen.
            interval = prevIdx+1:tracklen;
            BGVIDINDEX(interval) = BGVIDINDEX(prevIdx);
            VIDEONAME(interval) = VIDEONAME(prevIdx);
            prevVidFrame = VIDEOFRAME(prevIdx);
            VIDEOFRAME(interval) = prevVidFrame+1:prevVidFrame+length(interval);
        end
        if BGVIDINDEX(prevIdx)==BGVIDINDEX(nextIdx) %if there is the same information on either side of non-segmented frames, fill it in.
            interval = prevIdx+1:nextIdx-1;
            BGVIDINDEX(interval) = BGVIDINDEX(prevIdx);
            VIDEONAME(interval) = VIDEONAME(prevIdx);
            prevVidFrame = VIDEOFRAME(prevIdx);
            VIDEOFRAME(interval) = prevVidFrame+1:prevVidFrame+length(interval);
        end
        %IMPORTANT: IF the worm is not segmented on the last frame before the
        %video switches (rare, but possible), the NaN will remain. --> deal
        %with this in the playframes_summary.m function.
    end
end

%re-compute Roaming and Dwelling after centroid has been concatenated
%across tracks
[expSeq, expStates, ~, ~, ~] = getHMMStatesConcatenatedTracks(SPEED,ANGSPEED,binSize,CENTROIDINLAWN,cutoff,trans,emis,x_offset,false);
ROAMDWELL_2D = expSeq;
ROAMDWELL_HMM = expStates;

%put information into summary struct
SUMMARY_STRUCT.PIXPERMM = pixpermm;
SUMMARY_STRUCT.TRACKNUM = TRACKNUM;
SUMMARY_STRUCT.TRACKIDX = TRACKIDX;

SUMMARY_STRUCT.BGVIDINDEX = BGVIDINDEX;
SUMMARY_STRUCT.VIDEONAME = VIDEONAME;
SUMMARY_STRUCT.VIDEOFRAME = VIDEOFRAME;

OK_FRAMES_IN_LAWN(OK_int_inlawn_idx)=true;
SUMMARY_STRUCT.OK_FRAMES_IN_LAWN = OK_FRAMES_IN_LAWN; %zero or one if the track is in the lawn on that frame
SUMMARY_STRUCT.IN_OR_OUT = IN_OR_OUT;

SUMMARY_STRUCT.BBOX = BBOX;
SUMMARY_STRUCT.CENTROID = CENTROID;
SUMMARY_STRUCT.HEAD = HEAD;
SUMMARY_STRUCT.TAIL = TAIL;
SUMMARY_STRUCT.CENTROID_SMTH = CENTROID_SMTH;
SUMMARY_STRUCT.HEAD_SMTH = HEAD_SMTH;
SUMMARY_STRUCT.TAIL_SMTH = TAIL_SMTH;
SUMMARY_STRUCT.HEAD_GS = HEAD_GS;
SUMMARY_STRUCT.TAIL_GS = TAIL_GS;
SUMMARY_STRUCT.SPLINE_x = SPLINE_x;
SUMMARY_STRUCT.SPLINE_y = SPLINE_y;
SUMMARY_STRUCT.CURVATURE = CURVATURE;
SUMMARY_STRUCT.POSTURE_ANGLE = POSTURE_ANGLE;
SUMMARY_STRUCT.FORWARD = FORWARD;
SUMMARY_STRUCT.REVERSE = REVERSE;
SUMMARY_STRUCT.OMEGA = OMEGA;
SUMMARY_STRUCT.MSD = MSD;
SUMMARY_STRUCT.SPEED = SPEED;
SUMMARY_STRUCT.SPEED_SMTH = SPEED_SMTH;
SUMMARY_STRUCT.ANGSPEED = ANGSPEED;
SUMMARY_STRUCT.ROAMDWELL_2D = ROAMDWELL_2D;
SUMMARY_STRUCT.ROAMDWELL_HMM = ROAMDWELL_HMM;
SUMMARY_STRUCT.CENTROIDINLAWN = CENTROIDINLAWN;
SUMMARY_STRUCT.HEADINGLAWN = HEADINLAWN;
SUMMARY_STRUCT.TAILINLAWN = TAILINLAWN;
SUMMARY_STRUCT.FULLYINLAWN = FULLYINLAWN;
SUMMARY_STRUCT.HEADPOKE_FWD = HEADPOKE_FWD;
SUMMARY_STRUCT.HEADPOKE_REV = HEADPOKE_REV;
SUMMARY_STRUCT.HEADPOKE_PAUSE = HEADPOKE_PAUSE;
SUMMARY_STRUCT.RADIAL_DIST = RADIAL_DIST;
SUMMARY_STRUCT.EV_HO_DIST = EV_HO_DIST;
SUMMARY_STRUCT.HEADPOKE_ANGLE = HEADPOKE_ANGLE;
SUMMARY_STRUCT.LAWN_EXIT = LAWN_EXIT;
SUMMARY_STRUCT.LAWN_ENTRY = LAWN_ENTRY;

%DERIVED QUANTITIES
FRAC_ROAMING_HMM = nansum(ROAMDWELL_HMM==2)/sum(~isnan(ROAMDWELL_HMM));
SUMMARY_STRUCT.FRAC_ROAMING_HMM = FRAC_ROAMING_HMM;

MEAN_SPEED_ON_FOOD = nanmean(abs(SPEED_ON_FOOD));
MEAN_SPEED_OFF_FOOD = nanmean(abs(SPEED_OFF_FOOD));

SUMMARY_STRUCT.SPEED_ON_FOOD = SPEED_ON_FOOD;
SUMMARY_STRUCT.MEAN_SPEED_ON_FOOD = MEAN_SPEED_ON_FOOD;
SUMMARY_STRUCT.SPEED_OFF_FOOD = SPEED_OFF_FOOD;
SUMMARY_STRUCT.MEAN_SPEED_OFF_FOOD = MEAN_SPEED_OFF_FOOD;

end

