% % 01-02-2019
% % Elias Scheer
% %
% % with tracking some code copied from
% % \MATLAB2016b\toolbox\vision\visiondemos\multiObjectTracking.m
% % and from Ev Yemini's segWorm_stuff\ComputerVision functions for
% % segmentation

% objective: track worms as they hang out in lawns of bacteria and
% occasionally leave. extract lots of other behavioral metrics associated
% with worm locomotion for subsequent in-depth analysis.

%This version incorporates changes to segmentation processes, event-horizon
%detection, and introduces a slew of new behavioral metrics in the
%tracks_postprocessing5.m method. Introduces the SUMMARY_STRUCT object,
%which contains data only for the statistical interval (stat_int) in
%question -- for most of my movies taken at 3fps for 1 hour (total 10800
%frames), this interval is from frame 3601 to 10800 (7200 frames = 40
%minutes in the latter part of the assay).

function [TRACKS, EXIT_STRUCT, POKE_STRUCT, TRACKS_slim, SUMMARY_STRUCT, bg_struct] = tracking_enter_exit_010219(loadconfig, showtracking_flag, camera_params_path)
warnid = 'images:initSize:adjustingMag';
warning('off',warnid);

timenow = datestr(now,'mm_dd_yy');
lawn_string = ['010219_SOLO_' timenow];
maxwormnumber = 3;
p=10; %use the 10th percentile to generate the background
num_gaussian = 10; %default was 3

% camera_params_path = which('tracking_params\camera_params_02_02_2019.mat');%locate the tracking parameters file
if nargin < 3 %if no camera params file was specified.
    camera_params_path = which('tracking_params\camera_params_NORMAL_LED_setup_09_12_2019.mat');
end
% camera_params_path = which(['tracking_params\' camera_params_path]);
load(camera_params_path, 'camera_params'); %load in data

%if pixpermm is not supplied, get this value from camera_params
% if nargin<3
    files = dir('*.avi');
    uncutmovies = files(not([files.isdir]'));
    moviename = uncutmovies(1).name;
    cameracode = moviename(1:2);
    camera_names = {'s1','s2','s3','s4','s5','s6','c1','c2','c3','c4','c5','c6'};
    recognized = strcmp(camera_names,cameracode);
    while ~sum(recognized)
        disp('Camera number not recognized.');
        cameracode = input('PLEASE SUPPLY CAMERA CODE (e.g. ''c1'')');
        recognized = strcmp(camera_names,cameracode);
    end
    switch cameracode
        case 's1'
            level = camera_params.s1.level;
            pixpermm = camera_params.s1.pixpermm;
            th = camera_params.s1.th;
        case 's2'
            level = camera_params.s2.level;
            pixpermm = camera_params.s2.pixpermm;
            th = camera_params.s2.th;
        case 's3'
            level = camera_params.s3.level;
            pixpermm = camera_params.s3.pixpermm;
            th = camera_params.s3.th;
        case 's4'
            level = camera_params.s4.level;
            pixpermm = camera_params.s4.pixpermm;
            th = camera_params.s4.th;
        case 's5'
            level = camera_params.s5.level;
            pixpermm = camera_params.s5.pixpermm;
            th = camera_params.s5.th;
        case 's6'
            level = camera_params.s6.level;
            pixpermm = camera_params.s6.pixpermm;
            th = camera_params.s6.th;
        case 'c1'
            level = camera_params.c1.level;
            pixpermm = camera_params.c1.pixpermm;
            th = camera_params.c1.th;
        case 'c2'
            level = camera_params.c2.level;
            pixpermm = camera_params.c2.pixpermm;
            th = camera_params.c2.th;
        case 'c3'
            level = camera_params.c3.level;
            pixpermm = camera_params.c3.pixpermm;
            th = camera_params.c3.th;
        case 'c4'
            level = camera_params.c4.level;
            pixpermm = camera_params.c4.pixpermm;
            th = camera_params.c4.th;
        case 'c5'
            level = camera_params.c5.level;
            pixpermm = camera_params.c5.pixpermm;
            th = camera_params.c5.th;
        case 'c6'
            level = camera_params.c6.level;
            pixpermm = camera_params.c6.pixpermm;
            th = camera_params.c6.th;
        otherwise
            warning('Camera data cannot be found!');
            level = 0.90; %default
            th = 0.003; %default
            pixpermm = input('Enter pixel size (pixels/mm).');
    end
% end

[filename, pathname, ~] = uigetfile({'*'}); %this is nice because you can see which ones are in progress
cd(pathname);
if loadconfig
    load(filename, '-regexp', '^(?!showtracking_flag)\w') %load everything except the tracking flag (you may wish to watch the segmentation live even if that was not done the last time tracked.)
%     videoname = bg_struct(1).videoname;
else
    %% Create the video object
    close all;
    %javier's setups make multiple movie files per experiment
    movielist = dir('*.avi');
    movienames = {movielist.name}';
    
    % Get next file and delete it from the list of movies to be processed
    [ movienames, videoname ] = getnextvideo( movienames );
    v = VideoReader([pathname videoname]);
    number_of_frames = v.NumberOfFrames;
    
    close all;
    
    %% Pick event horizon and outer boundary
    last_frame = imadjust(rgb2gray(im2double(read(v,number_of_frames))));
    
    %select region for correcting plate bumps
    disp('SELECT REGION OUTSIDE OF THE PLATE FOR FRAME REGISTRATION.');
    x = figure(1);
    ax = axes(); hparent = imshow(last_frame);
    h = imrect(ax,[10 10 100 100]); wait(h);
    crop_pos = getPosition(h);
    close(x);
    
%     % get pixelsize if requested
%     if nargin==3
%         pixpermm = round(getpixpermm(last_frame, 10));
%         disp(['pixpermm = ' num2str(pixpermm)]);
%     end
    
    %get outer boundary mask
    rad_scale_factor = 45/112;
    rad_to_subtract = pixpermm*rad_scale_factor; %NEW 09/27/18
    [ outer_boundary_mask, outer_boundary_line, region_rounded ] = get_outer_boundary_rough( last_frame, rad_to_subtract, 10 );
    outer_boundary_line_crp_rel = [outer_boundary_line(:,1)-region_rounded(1) outer_boundary_line(:,2)-region_rounded(2)];%shift to the correct coordinates
    
%     %now mask and crop before getting the event_horizon coordinates
%     masked = last_frame.*outer_boundary_mask;
%     cropped = imcrop(masked,region_rounded);
    
    %% Generate background and clean it
    
    [orig_background, clean_background] = getbackground7_prctile_cleanup(v, level, p, num_gaussian, pixpermm, outer_boundary_mask);
    bg_struct(1).videoname = videoname;
    bg_struct(1).videoframe = 1;
    bg_struct(1).curr_frame = 1;
    bg_struct(1).region_rounded = region_rounded;
    bg_struct(1).outer_boundary_mask = outer_boundary_mask;
    bg_struct(1).outer_boundary_line = outer_boundary_line;
    bg_struct(1).orig_background = orig_background;
    bg_struct(1).clean_background = clean_background;
    bg_struct(1).level = level;
    bg_struct(1).th = th;
    
    %% Get the event horizon from the background image
%     [ ev_ho_x, ev_ho_y, lawn_limit_line] = geteventhorizon022119(orig_background, outer_boundary_mask, [], pixpermm);
    [ ev_ho_x, ev_ho_y, lawn_limit_line] = geteventhorizon022119(orig_background, outer_boundary_mask, [], pixpermm, th);
    ev_ho_crp_rel = [ev_ho_x-region_rounded(1) ev_ho_y-region_rounded(2)]; %event horizon in cropped coordinates
    lawn_limit_mask_wf = poly2mask(lawn_limit_line(:,1),lawn_limit_line(:,2),size(last_frame,1),size(last_frame,2));
    fullyblurred_background = imgaussfilt(regionfill(orig_background,lawn_limit_mask_wf),4);
    bg_struct(1).fullyblurred_background = fullyblurred_background;
    bg_struct(1).lawn_limit_mask_wf = lawn_limit_mask_wf;  % in whole frame coordinates
    bg_struct(1).lawn_limit_line = lawn_limit_line;     % in whole frame coordinates
    bg_struct(1).ev_ho = [ev_ho_x ev_ho_y];             % in whole frame coordinates
    bg_struct(1).ev_ho_crp_rel = ev_ho_crp_rel;         % in cropped frame coordinates
    
    %% Check threshold level and modify it if necessary
    checkThreshold();
    disp(['level = ' num2str(level)]);
    
    %% Save information used for tracking
    save([videoname(1:end-11) '_' lawn_string '_CONFIG' '.mat']);
end

%% TRACKING!
hblob = vision.BlobAnalysis;
hblob.Connectivity = 8;
hblob.ExcludeBorderBlobs = true;
hblob = vision.BlobAnalysis;
hblob.Connectivity = 8;
hblob.ExcludeBorderBlobs = true;
hblob.MaximumCount = maxwormnumber;
minblob_scale_factor = (300/112);
minblobarea = round(pixpermm*minblob_scale_factor); % new addition 09/27/18
hblob.MinimumBlobArea = minblobarea;
hblob.MaximumBlobArea = 2500;
hblob.OutputDataType = 'double';

% fastest way to read in the movie
v2 = VideoReader([pathname videoname]); %have to re-instantiate this object so we can use the faster readFrame method.
v2.CurrentTime = 0; % start at the beginning movie THIS DOESNT EXIST IN 2014a

%for playback
trackPlayer = vision.VideoPlayer('Position', [1043, 32, 816, 593]);

% initialize tracks
[tracks,tracksThatLeft] = initializeTracks_wholevideo2();

nextId = 1;
curr_frame = 1;
videoframe = 1;
skip_loop = 0;
bgvidindex = 1;
lookback = 0; %boolean -- whether to look back to get a frame with the head's grayscale
pastframes = 360; %look back 2 minutes to find the current grayscale at the head
gs_lookback = cell(pastframes,1);
prev_bump_roi = [];
bump_roi = [];
y_shift = 0;
x_shift = 0;
usfac = 100; %tolerance for dft registration

ev_ho_dict = zeros(21700,1); %at most 21600 frame long video, just add some extra frames, will be chopped (?)
ev_ho_dict(curr_frame) = bgvidindex;

while hasFrame(v2) || ~isempty(movienames) %MAIN LOOP
    if ~hasFrame(v2) %change the video here
        disp('CHANGE VIDEO!');
        videoframe = 1;
        bgvidindex = bgvidindex+1;
        [ movienames, videoname ] = getnextvideo( movienames );
        v = VideoReader(videoname);%just for info
        [orig_background, clean_background] = getbackground7_prctile_cleanup(v, level, p, num_gaussian, pixpermm, outer_boundary_mask);
        bg_struct(bgvidindex).videoname = videoname; %#ok<*AGROW>
        bg_struct(bgvidindex).videoframe = videoframe;
        bg_struct(bgvidindex).curr_frame = curr_frame;
        bg_struct(bgvidindex).region_rounded = region_rounded;
        bg_struct(bgvidindex).outer_boundary_mask = outer_boundary_mask;
        bg_struct(bgvidindex).outer_boundary_line = outer_boundary_line;
        bg_struct(bgvidindex).orig_background = orig_background;
        bg_struct(bgvidindex).clean_background = clean_background;
        bg_struct(bgvidindex).level = level;
        %         [ ev_ho_x, ev_ho_y, lawn_limit_line] = geteventhorizon022119(orig_background, outer_boundary_mask, lawn_limit_line, pixpermm);
        [ ev_ho_x, ev_ho_y, lawn_limit_line] = geteventhorizon022119(orig_background, outer_boundary_mask, lawn_limit_line, pixpermm, th);
        ev_ho_crp_rel = [ev_ho_x-region_rounded(1) ev_ho_y-region_rounded(2)]; %event horizon in cropped coordinates
        lawn_limit_mask_wf = poly2mask(lawn_limit_line(:,1),lawn_limit_line(:,2),size(orig_background,1),size(orig_background,2));
        fullyblurred_background = imgaussfilt(regionfill(orig_background,lawn_limit_mask_wf),4);
        bg_struct(bgvidindex).fullyblurred_background = fullyblurred_background;
        bg_struct(bgvidindex).lawn_limit_mask_wf = lawn_limit_mask_wf;
        bg_struct(bgvidindex).lawn_limit_line = lawn_limit_line;
        bg_struct(bgvidindex).ev_ho = [ev_ho_x ev_ho_y];
        bg_struct(bgvidindex).ev_ho_crp_rel = ev_ho_crp_rel;  % in cropped frame coordinates
        clear v2;
        v2 = VideoReader(videoname);%for tracking
        v2.CurrentTime = 0;
    end
    
    disp(['curr frame: ' num2str(curr_frame)]);
    %     DEBUG STEP
    %     if curr_frame == 1752
    %         disp('debug');
    %     end
    frame_ui8 = rgb2gray(readFrame(v2)); %read in the next frame, uint8
    
    checkForBumps(); % see if the plate has bumped and shift everything over if need be.
    
    frame = imcomplement(im2double(frame_ui8));
    %%%%%%%%%
    masked = frame.*outer_boundary_mask;
    cropped = imcrop(masked,region_rounded);
    %changed 11/29/18 - now subtracting the orig background instead of
    %cleaned background - this should help deal with agar bubbles. %changed
    %back!
    bgsub = (imcomplement(cropped - imcrop(clean_background.*outer_boundary_mask,region_rounded)));
    thresh = imcomplement(im2bw(bgsub,level));
%     pixremoval_scale_factor = (500/96);
%     thresh_cleaned = bwareaopen(thresh,round(pixremoval_scale_factor*pixpermm));
    thresh_cleaned = bwareaopen(thresh,minblobarea);
    BWfinal = thresh_cleanup_110818(thresh_cleaned); %this just erodes and dilates, no contour smoothing.
    %%% Detect worms: %%%
    [~, centroids, bboxes] = step(hblob,BWfinal); %image coordinates
    bboxes = double(bboxes);
    %%%%%%%%%%%%%%%%%%%%%
    %get a cleaned, worm-removed version of this image for extracting the
    %grayscale value (proxy for bacterial thickness)
    cropped_gs = get_clean_grayscale( frame_ui8, region_rounded, outer_boundary_mask, BWfinal, bboxes, fullyblurred_background );
    cropped_gs_detect =  get_head_grayscale(cropped_gs);
    %%%%%%%%%
    cropworms = cell(size(bboxes,1),1);
    cropworms_orig = cell(size(bboxes,1),1);
    splines = cell(size(bboxes,1),1);
    end1s = NaN(size(bboxes,1),2);
    end2s = NaN(size(bboxes,1),2);
    end1s_g = NaN(size(bboxes,1),1);
    end2s_g = NaN(size(bboxes,1),1);
    curvatures = cell(size(bboxes,1),1);
    posture_angles = cell(size(bboxes,1),1);
    omegas = zeros(size(bboxes,1),1);
    
    for j = 1:size(bboxes,1)
        bbox_crop = [bboxes(j,1)-5 bboxes(j,2)-5 bboxes(j,3)+10 bboxes(j,4)+10];%provide some pixel padding
        bbox_crop(bbox_crop<1) = 1; %correct for out of bounds
        if bbox_crop(1)+bbox_crop(3)>size(BWfinal,2)
            bbox_crop(3) = size(BWfinal,2)-bbox_crop(1)-1;
        end
        if bbox_crop(2)+bbox_crop(4)>size(BWfinal,1)
            bbox_crop(4) = size(BWfinal,1)-bbox_crop(2)-1;
        end
        
        wormcrop = imcrop(BWfinal, bbox_crop);
        wormcrop = padarray(wormcrop,[1 1]); %sometimes at the edge, the whole worm is up against the edge
        x_offset = bbox_crop(1)-1; y_offset = bbox_crop(2)-1; %account for this padarray operation.
        wormcrop = imclearborder(wormcrop, 4);
        
        cropworms{j} = wormcrop;
        wormcrop_orig = imcomplement(imcrop(cropped,bbox_crop)); %would it be better as just cropped instead of bgsub?
        cropworms_orig{j} = wormcrop_orig;
        %only calculate the spline if the centroid is a minimum
        %distance away from the edge of the video
        [~,dist,~] = distance2curve(outer_boundary_line_crp_rel,centroids(j,:));
        if lookback
            curr_lookback_frame = cropped_gs_detect{j};
        end
        dist_scale_factor = (50/112); %increase to 50 01/16/19
        if dist > pixpermm*dist_scale_factor %allowed to get spline at this distance!, try to segment worm
            [wormskel, pt1, pt2, pt1_g, pt2_g, omega] = segment_worm();
            skip_loop = 0;
        else
            disp('TOO CLOSE TO BOUNDARY');
            skip_loop = 1;
            break;
        end
        omegas(j) = omega;
        if ~isnan(wormskel) %if the worm has been successfully segmented, get it's spline and associated values
            [spline, curvature, posture_angle] = getWormSpline010219(wormskel);
        else
            spline = NaN(49,2);
            curvature = NaN(49,1);
            posture_angle = NaN(1,48); %the only oddball that got transposed. can fix later if desired.
        end
        splines{j} = spline;
        curvatures{j} = curvature;
        posture_angles{j} = posture_angle;
        end1s(j,:) = pt1;
        end2s(j,:) = pt2;
        end1s_g(j) = pt1_g;
        end2s_g(j) = pt2_g;
    end
    
    if skip_loop
        if ~isempty(tracks) %update the consecutive invisible counts of tracks in this case
            disp('TRACKS EXIST AND WILL END');
            unassignedTracks = 1:length(tracks); %indices
            [tracks, tracksThatLeft] = updateUnassignedTracks_wholevideo(tracks,tracksThatLeft,unassignedTracks);
        end
        curr_frame = curr_frame + 1;
        videoframe = videoframe +1;
        ev_ho_dict(curr_frame) = bgvidindex;
        release(trackPlayer); %sometimes you need to do this, not really sure why
        continue;
    end
    %%% MAIN TRACKING FUNCTIONS
    tracks = predictNewLocationsOfTracks_wholevideo(tracks);
    
    [assignments, unassignedTracks, unassignedDetections] = detectionToTrackAssignment_wholevideo(tracks,centroids);
    
    if length(unassignedTracks) >= 1
        warning('WE LOST THE TRACK (NOT OUT OF BOUNDS)');
        disp(curr_frame);
        release(trackPlayer); %sometimes you need to do this, not really sure why
    end
    
    tracks = updateAssignedTracks_wholevideo2(tracks, assignments, x_shift, y_shift, centroids, bboxes, cropworms, cropworms_orig, splines, end1s, end2s, end1s_g, end2s_g, omegas, curvatures, posture_angles, curr_frame, videoframe, videoname);
    
    [tracks, tracksThatLeft] = updateUnassignedTracks_wholevideo(tracks,tracksThatLeft,unassignedTracks);
    
    [tracks, nextId] = createNewTracks_wholevideo2(tracks,nextId,x_shift,y_shift,unassignedDetections,centroids,bboxes,cropworms,cropworms_orig,splines,end1s, end2s, end1s_g, end2s_g, omegas,curvatures, posture_angles, curr_frame,videoframe,videoname);
    
    if showtracking_flag
        displayTrackingResults_wholevideo_lawn(trackPlayer, tracks, imcomplement(cropped), end1s, end2s, ev_ho_crp_rel(:,1), ev_ho_crp_rel(:,2));
    end
    
    curr_frame = curr_frame + 1;
    videoframe = videoframe +1;
    ev_ho_dict(curr_frame) = bgvidindex;
end %END MAIN LOOP
if curr_frame<length(ev_ho_dict) %chop it down to the correct size
    ev_ho_dict = ev_ho_dict(1:curr_frame);
end

%% DO POST-PROCESSING ON TRACKS TO DERIVE SPEED, ANGULAR VELOCITY --> ROAMING / DWELLING, LAWN LEAVING EVENTS
allTracks = [tracksThatLeft,tracks];
WorthSavingTracksThatLeft_idx = [allTracks(:).age]'>9; %double check that all tracks are still long enough -- sometimes the last track is too short to be processed
allTracks = allTracks(WorthSavingTracksThatLeft_idx);
save([videoname '_' lawn_string '_allTracks.mat'],'allTracks','ev_ho_dict','bg_struct','pixpermm','videoname','lawn_string');
[TRACKS, EXIT_STRUCT, POKE_STRUCT, SUMMARY_STRUCT] = tracks_postprocessing6( allTracks, ev_ho_dict, bg_struct, pixpermm, [videoname(1:end-4) '_' lawn_string] );

%% SAVING!
fields2remove = {'cropworm','cropworm_orig'}; %<--- USE THIS ONE FOR ALL FAST DATA MANIPULATIONS
TRACKS_slim = rmfield(TRACKS,fields2remove);

save([videoname '_' lawn_string '_BACKGROUND.mat'],'bg_struct');
save([videoname '_' lawn_string '_FINAL' '.mat'],'TRACKS_slim','EXIT_STRUCT','POKE_STRUCT','SUMMARY_STRUCT','bg_struct');
save([videoname '_' lawn_string '_SUMMARY.mat'],'SUMMARY_STRUCT');
save([videoname '_' lawn_string '_TRACKS_CROPWORMS.mat'],'TRACKS','-v7.3');

% delete([videoname '_' lawn_string '_allTracks.mat']); %if you successfully make it here, delete allTracks to save room.
%% %%%%%%%%%%%%%%%%%%%%NESTED FUNCTIONS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function checkThreshold()
        frames_to_check = randi([fix(number_of_frames/2) number_of_frames],100,1);
        goodcounter = 0;
        badcounter = 0;
        for k = 1:length(frames_to_check)
            if goodcounter > 3 || badcounter > 0
                break;
            end
            frame = imcomplement(im2double(rgb2gray((read(v,frames_to_check(k))))));
            masked = frame.*outer_boundary_mask;
            cropped = imcrop(masked,region_rounded);
%             bgsub = (imcomplement(cropped - imcrop(orig_background.*outer_boundary_mask,region_rounded)));
            bgsub = (imcomplement(cropped - imcrop(clean_background.*outer_boundary_mask,region_rounded)));
            thresh = im2bw(bgsub,level);
            imshow(thresh);
            
            ButtonName = questdlg('Is threshold OK?', ...
                'Check Threshold', ...
                'GOOD', 'BAD', 'NO WORMS', 'NO WORMS');
            switch ButtonName,
                case 'BAD',
                    badcounter = badcounter+1;
                case 'GOOD',
                    goodcounter = goodcounter+1;
                case 'NO WORMS',
                    continue;
            end
        end
        close all;
        %if there is something wrong, choose the new threshold
        if badcounter > 0
            levels = NaN;
            goodcounter = 0;
            for k = 1:length(frames_to_check)
                if goodcounter > 3
                    break;
                end
                frame = imcomplement(im2double(rgb2gray((read(v,frames_to_check(k))))));
                masked = frame.*outer_boundary_mask;
                cropped = imcrop(masked,region_rounded);
%                 bgsub = (imcomplement(cropped - imcrop(orig_background.*outer_boundary_mask,region_rounded)));
                bgsub = (imcomplement(cropped - imcrop(clean_background.*outer_boundary_mask,region_rounded)));
                [~, level, ~] = choose_thresh(bgsub,level);
                close all;
                thresh = im2bw(bgsub,level);
                imshow(thresh);
                button = questdlg('is threshold good (with worms in it)?');
                if strcmp(button,'No')
                    continue;
                else
                    levels = [levels level];
                    goodcounter = goodcounter+1;
                end
            end
            close all;
            level = nanmean(levels);
        end
    end

    function checkForBumps()
        %%%% CHECK FOR VIDEO TRANSLATIONAL SHIFTS I.E. FROM PLATE BUMPS
        prev_bump_roi = bump_roi;
        bump_roi = imcrop(frame_ui8,crop_pos); %use this for checking if plate moved
        change_x = 0; change_y = 0;
        if curr_frame>1
            [output, ~] = dftregistration(fft2(prev_bump_roi),fft2(bump_roi),usfac); %register successive images
            add_y = output(3); add_x = output(4);
            if abs(add_y)>=1 %only take it seriously if the bump is at least 1 pixel
                y_shift = round(y_shift+add_y);
                change_y = 1;
            else
                change_y = 0;
            end
            if abs(add_x)>=1
                x_shift = round(x_shift+add_x);
                change_x = 1;
            else
                change_x = 0;
            end
        end
        %SHIFT EVERYTHING: OUTER BOUNDARY LINES, MASKS, BACKGROUND, EVENT
        %HORIZON (all masks and lines are defined with respect to the frame
        %size)
        if change_x || change_y
            region_rounded = [region_rounded(1)-x_shift region_rounded(2)-y_shift region_rounded(3)-abs(x_shift) region_rounded(4)-abs(y_shift)]; %the outer boundary line will always shrink if the plate is bumped --therefore minus the abs val
            tmpax = axes(); h = imellipse(tmpax,region_rounded);
            outer_boundary_line = getVertices(h); %this can move or change shape
            outer_boundary_mask = poly2mask(outer_boundary_line(:,1),outer_boundary_line(:,2),size(frame,1),size(frame,2));
            outer_boundary_line_crp_rel = [outer_boundary_line(:,1)-region_rounded(1) outer_boundary_line(:,2)-region_rounded(2)];%shift to the correct coordinates
            lawn_limit_line = [lawn_limit_line(:,1)-x_shift lawn_limit_line(:,2)-y_shift];
            lawn_limit_mask_wf = poly2mask(lawn_limit_line(:,1),lawn_limit_line(:,2),size(frame,1),size(frame,2));
            ev_ho_x = ev_ho_x-x_shift; ev_ho_y = ev_ho_y-y_shift;
            ev_ho_crp_rel = [ev_ho_x-region_rounded(1) ev_ho_y-region_rounded(2)]; %event horizon in cropped coordinates, this should basically do nothing since you are adding and subtracting. just added for sake of clarity and to be sure everything matches up
            close all;
            %change these, save changes
            bgvidindex = bgvidindex+1;
            bg_struct(bgvidindex).videoname = videoname;
            bg_struct(bgvidindex).videoframe = videoframe;
            bg_struct(bgvidindex).curr_frame = curr_frame;
            bg_struct(bgvidindex).region_rounded = region_rounded;
            bg_struct(bgvidindex).outer_boundary_mask = outer_boundary_mask;
            bg_struct(bgvidindex).outer_boundary_line = outer_boundary_line;
            bg_struct(bgvidindex).orig_background = orig_background;
            bg_struct(bgvidindex).clean_background = clean_background;
            bg_struct(bgvidindex).level = level;
            bg_struct(bgvidindex).th = th;
            bg_struct(bgvidindex).fullyblurred_background = fullyblurred_background;
            %         bg_struct(bgvidindex).abovethresh = abovethresh;
            bg_struct(bgvidindex).lawn_limit_mask_wf = lawn_limit_mask_wf;
            bg_struct(bgvidindex).lawn_limit_line = lawn_limit_line;
            bg_struct(bgvidindex).ev_ho = [ev_ho_x ev_ho_y];
            bg_struct(bgvidindex).ev_ho_crp_rel = ev_ho_crp_rel;
            release(trackPlayer); %you have to do this so that a new sized frame can be displayed
        end
    end

    function cropped_gs_detect =  get_head_grayscale(cropped_gs)
        %%% ASSIGN GRAYSCALE VALUE ASSOCIATED WITH THE HEAD AND TAIL
        cropped_gs_detect = cell(size(centroids,1),1);
        %this is a first in first out queue
        if curr_frame<=pastframes
            gs_lookback{curr_frame} = cropped_gs;
            lookback = 0;
        else %start looking back
            %figure out which frame to use to get the grayscale head value
            for k = 1:size(centroids,1) %get one past frame for each centroid detected
                cent_lookback = getprevframe4gs2(tracks,tracksThatLeft,curr_frame,centroids(k,:),pastframes,cropped,pixpermm);
                numlookback = pastframes-cent_lookback+1;
                cropped_gs_detect{k} = gs_lookback{numlookback}; %1 is pastframes ago
            end
            gs_lookback(1) = [];%remove the first element
            gs_lookback{pastframes} = cropped_gs;%add the new element
            lookback = 1;
        end
    end

    function [wormskel, pt1, pt2, pt1_g, pt2_g, omega] = segment_worm()
        pt1_g = NaN; %default values, only to be overridden if lookback is active
        pt2_g = NaN;
%         x_offset = bbox_crop(1); y_offset = bbox_crop(2);
        [wormskel, pt1, pt2, omega] = segWormOnly_Elias2(wormcrop, x_offset, y_offset, pixpermm);
        if lookback && sum(isnan(pt1))==0 && sum(isnan(pt2))==0
            pt1_g = curr_lookback_frame(pt1(2),pt1(1)); %get the grayscale value associated with the head and tail - remember this is image coordinates, have to reverse again!
            pt2_g = curr_lookback_frame(pt2(2),pt2(1));
        end
    end

end
