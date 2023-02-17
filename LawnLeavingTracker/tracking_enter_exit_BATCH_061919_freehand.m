function tracking_enter_exit_BATCH_061919_freehand(loadconfig, tracklocally, recomputeEvHo, camera_params_file)
clc; close all;
warnid = 'images:initSize:adjustingMag';
warning('off',warnid);

timenow = datestr(now,'mm_dd_yy');
lawn_string = ['010219_BATCH_' timenow];
maxwormnumber = 3;
p=10; %use the 10th percentile to generate the background
num_gaussian = 10; %default was 3

% camera_params_path = which('tracking_params\camera_params_02_02_2019.mat');%locate the tracking parameters file
if nargin < 4 %if no camera params file was specified.
    camera_params_file = 'camera_params_NORMAL_LED_setup_09_12_2019.mat';
end

camera_params_path = which(['tracking_params\' camera_params_file]);
load(camera_params_path, 'camera_params'); %load in data

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


if nargin < 3
    recomputeEvHo = false;
end

%% Open the directory containing subdirectories of videos
[filename, pathname, ~] = uigetfile({'*'}); %this is nice because you can see which ones are in progress
cd(pathname);
files = dir('*');
folders = files([files.isdir]'); %get the subdirectories
folders = folders(~contains({folders.name}','.')); %get rid of the extra folders starting with '.' and '..'

%make a struct for all of the necessary information to process each video
outer_directory = pwd;
if ~loadconfig
    batch_struct = struct();
    %BATCH_STRUCT HAS THE FOLLOWING FIELDS:
    % PATH
    % MOVIENAMES
    % PIXPERMM
    % CROP_POS
    % OUTER_BOUNDARY_MASK
    % OUTER_BOUNDARY_LINE
    % OUTER_BOUNDARY_LINE_CRP_REL
    % REGION_ROUNDED
    % ORIG_BACKGROUND
    % CLEAN_BACKGROUND
    % LEVEL
    % FULLYBLURRED_BACKGROUND
    % LAWN_LIMIT_MASK_WF
    % LANW_LIMIT_LINE
    % EV_HO
    % EV_HO_CRP_REL
    
    for i = 1:length(folders) %go into each subdirectory and get all the necessary information for tracking
        curr_dir = folders(i).name;
        cd(curr_dir);
        
        batch_struct(i).trackdate = timenow; %date files were tracked; helpful for knowing whether you are done tracking or not.
        batch_struct(i).path = pwd;
        
        movielist = dir('*.avi');
        if isempty(movielist) %skip this folder if there are no videos in it.
            continue;
        end
        movienames = {movielist.name}';
        batch_struct(i).movienames = movienames; %save it
        [ movienames, videoname ] = getnextvideo( movienames );
        v = VideoReader(videoname);
        number_of_frames = v.NumberOfFrames;
        
        %Pick event horizon and outer boundary
        last_frame = imadjust(rgb2gray(im2double(read(v,number_of_frames))));
        
        batch_struct(i).pixpermm = pixpermm;
        
        %select region for correcting plate bumps
        disp('SELECT REGION OUTSIDE OF THE PLATE FOR FRAME REGISTRATION.');
        x = figure(1);
        ax = axes(); hparent = imshow(last_frame);
        h = imrect(ax,[10 10 100 100]); wait(h);
        crop_pos = getPosition(h);
        close(x);
        batch_struct(i).crop_pos = crop_pos;
        
        %get outer boundary mask
        rad_scale_factor = 45/112;
        rad_to_subtract = pixpermm*rad_scale_factor; %NEW 09/27/18
        [ outer_boundary_mask, outer_boundary_line, region_rounded ] = get_outer_boundary_rough( last_frame, rad_to_subtract, 10 );
        outer_boundary_line_crp_rel = [outer_boundary_line(:,1)-region_rounded(1) outer_boundary_line(:,2)-region_rounded(2)];%shift to the correct coordinates
        
        batch_struct(i).outer_boundary_mask = outer_boundary_mask;%save it
        batch_struct(i).outer_boundary_line = outer_boundary_line;
        batch_struct(i).outer_boundary_line_crp_rel = outer_boundary_line_crp_rel;
        batch_struct(i).region_rounded = region_rounded;
        
        %Generate background and clean it
        [orig_background, clean_background] = getbackground7_prctile_cleanup(v, level, p, num_gaussian, pixpermm, outer_boundary_mask);
        batch_struct(i).orig_background = orig_background;
        batch_struct(i).clean_background = clean_background;
        batch_struct(i).level = level;
        batch_struct(i).th = th;
        
        %Get the event horizon from the background image
%         [ ev_ho_x, ev_ho_y, lawn_limit_line] = geteventhorizon_freehand(orig_background, outer_boundary_mask,[], pixpermm);
        [ ev_ho_x, ev_ho_y, lawn_limit_line] = geteventhorizon_freehand(orig_background, outer_boundary_mask, [], pixpermm, th);
        ev_ho_crp_rel = [ev_ho_x-region_rounded(1) ev_ho_y-region_rounded(2)]; %event horizon in cropped coordinates
        lawn_limit_mask_wf = poly2mask(lawn_limit_line(:,1),lawn_limit_line(:,2),size(last_frame,1),size(last_frame,2));
        fullyblurred_background = imgaussfilt(regionfill(orig_background,lawn_limit_mask_wf),4);
        batch_struct(i).fullyblurred_background = fullyblurred_background;
        batch_struct(i).lawn_limit_mask_wf = lawn_limit_mask_wf;  % in whole frame coordinates
        batch_struct(i).lawn_limit_line = lawn_limit_line;     % in whole frame coordinates
        batch_struct(i).ev_ho = [ev_ho_x ev_ho_y];             % in whole frame coordinates
        batch_struct(i).ev_ho_crp_rel = ev_ho_crp_rel;         % in cropped frame coordinates
        
        close all;
        cd(outer_directory);
        save([videoname(1:end-11) '_' lawn_string '_CONFIG' '.mat'],'batch_struct');
    end
else
    load(filename); %load in the config file
end

%% TRACKING!
%BG_STRUCT HAS THE FOLLOWING FIELDS:
% VIDEONAME
% VIDEOFRAME
% CURR_FRAME
% REGION_ROUNDED
% OUTER_BOUNDARY_MASK
% OUTER_BOUNDARY_LINE
% ORIG_BACKGROUND
% CLEAN_BACKGROUND
% LEVEL
% FULLYBLURRED_BACKGROUND
% LAWN_LIMIT_MASK_WF
% LAWN_LIMIT_LINE
% EV_HO
% EV_HO_CRP_REL

for vid = 1:length(batch_struct)
    if tracklocally
        tmp = batch_struct(vid).path;
        split = strsplit(tmp,'\');
        path = [outer_directory '\' split{end}];
    else
        path = batch_struct(vid).path;
    end
    %go to the correct directory
    cd(path);
    
    % CHECK WHICH STEPS REMAIN TO DO AND DO THEM.
    localfiles = dir('*.mat');
    localfiles = {localfiles.name}';
    
    skipToPostProc = false;
    if isfield(batch_struct(vid),'trackdate')
        trackdate = batch_struct(vid).trackdate;
        
        allTracksFinished = sum(contains(localfiles,[trackdate '_allTracks.mat'])>0);
        TracksCropwormsFinished = sum(contains(localfiles,[trackdate '_TRACKS_CROPWORMS.mat'])>0);
        
        if TracksCropwormsFinished %if you've already tracked through TRACKS_CROPWORMS.mat, skip this folder.
            continue;
        elseif allTracksFinished && ~TracksCropwormsFinished %if you've tracked up to allTracks.mat, start there.
            allTracksFile = localfiles{contains(localfiles,[trackdate '_allTracks.mat'])};
            load(allTracksFile,'pixpermm','lawn_string','videoname','bg_struct','ev_ho_dict','allTracks');
            skipToPostProc = true;
        end
    end
 
    if ~skipToPostProc
        %%% GET OUT ALL DATA FROM BATCH STRUCT
%         movienames = batch_struct(vid).movienames;
        movielist = dir('*.avi'); %just track everything that's in the folder -- not what was previously logged in the batch_struct.
        movienames = {movielist.name}';
        pixpermm = batch_struct(vid).pixpermm;
        crop_pos = batch_struct(vid).crop_pos;
        outer_boundary_mask = batch_struct(vid).outer_boundary_mask;
        outer_boundary_line = batch_struct(vid).outer_boundary_line;
        outer_boundary_line_crp_rel = batch_struct(vid).outer_boundary_line_crp_rel;
        region_rounded = batch_struct(vid).region_rounded;
        orig_background = batch_struct(vid).orig_background;
        clean_background = batch_struct(vid).clean_background;
        level = batch_struct(vid).level;
%         th = 0.0030; %that's what all of them are anyway... (10/19/21)
%         WOW VERY BAD!!!(deleted 02/15/22)
        th = batch_struct(vid).th;
        fullyblurred_background = batch_struct(vid).fullyblurred_background;
        lawn_limit_mask_wf = batch_struct(vid).lawn_limit_mask_wf;
        lawn_limit_line = batch_struct(vid).lawn_limit_line;
        if recomputeEvHo
%             [ ev_ho_x, ev_ho_y, lawn_limit_line] = geteventhorizon_freehand(orig_background, outer_boundary_mask, lawn_limit_line, pixpermm);
            [ ev_ho_x, ev_ho_y, lawn_limit_line] = geteventhorizon_freehand(orig_background, outer_boundary_mask, lawn_limit_line, pixpermm,th);
            ev_ho = [ev_ho_x ev_ho_y];
            ev_ho_crp_rel = [ev_ho_x-region_rounded(1) ev_ho_y-region_rounded(2)];
        else
            ev_ho = batch_struct(vid).ev_ho;
            ev_ho_x = ev_ho(:,1); ev_ho_y = ev_ho(:,2); %ADDED 12/25 THIS IS NECESSARY AS THE LOCAL VARIABLE
            ev_ho_crp_rel = batch_struct(vid).ev_ho_crp_rel;
        end
        
        %%%
        [ movienames, videoname ] = getnextvideo( movienames );
        v = VideoReader(videoname);
        videoframe = 1;
        curr_frame = 1;
        
        %%% ADD ALL RELEVANT INFORMATION TO THE BG_STRUCT
        bg_struct = struct(); %make it fresh for every new set of videos
        bg_struct(1).videoname = videoname;
        bg_struct(1).videoframe = videoframe;
        bg_struct(1).curr_frame = curr_frame;
        bg_struct(1).region_rounded = region_rounded;
        bg_struct(1).outer_boundary_mask = outer_boundary_mask;
        bg_struct(1).outer_boundary_line = outer_boundary_line;
        bg_struct(1).orig_background = orig_background;
        bg_struct(1).clean_background = clean_background;
        bg_struct(1).level = level;
        bg_struct(1).th = th;
        bg_struct(1).fullyblurred_background = fullyblurred_background;
        bg_struct(1).lawn_limit_mask_wf = lawn_limit_mask_wf;
        bg_struct(1).lawn_limit_line = lawn_limit_line;
        bg_struct(1).ev_ho = ev_ho;
        bg_struct(1).ev_ho_crp_rel = ev_ho_crp_rel;
        %%%
        
        %%% SET UP BLOB ANALYSIS
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
        v2 = VideoReader(videoname); %have to re-instantiate this object so we can use the faster readFrame method.
        v2.CurrentTime = 0; % start at the beginning movie THIS DOESNT EXIST IN 2014a
        
        %%% INITIALIZE TRACKS AND VALUES NEEDED FOR TRACKING
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
                %                 [ ev_ho_x, ev_ho_y, lawn_limit_line] = geteventhorizon010819(orig_background,outer_boundary_mask,lawn_limit_line); %event horizon is in whole frame coordinates
                [ ev_ho_x, ev_ho_y, lawn_limit_line] = geteventhorizon_freehand(orig_background, outer_boundary_mask, lawn_limit_line, pixpermm,th);
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
%             if curr_frame == 2006
%                 disp('debug');
%             end
            frame_ui8 = rgb2gray(readFrame(v2)); %read in the next frame, uint8
            
            checkForBumps(); % see if the plate has bumped and shift everything over if need be.
            
            frame = imcomplement(im2double(frame_ui8));
            %%%%%%%%%
            masked = frame.*outer_boundary_mask;
            cropped = imcrop(masked,region_rounded);
            bgsub = (imcomplement(cropped - imcrop(clean_background.*outer_boundary_mask,region_rounded)));
            thresh = imcomplement(im2bw(bgsub,level));
%             pixremoval_scale_factor = (500/96);
%             thresh_cleaned = bwareaopen(thresh,round(pixremoval_scale_factor*pixpermm));
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
                continue;
            end
            
            %%% MAIN TRACKING FUNCTIONS
            tracks = predictNewLocationsOfTracks_wholevideo(tracks);
            
            [assignments, unassignedTracks, unassignedDetections] = detectionToTrackAssignment_wholevideo(tracks,centroids);
            
            if length(unassignedTracks) >= 1
                warning('WE LOST THE TRACK (NOT OUT OF BOUNDS)');
                disp(curr_frame);
            end
            
            tracks = updateAssignedTracks_wholevideo2(tracks, assignments, x_shift, y_shift, centroids, bboxes, cropworms, cropworms_orig, splines, end1s, end2s, end1s_g, end2s_g, omegas, curvatures, posture_angles, curr_frame, videoframe, videoname);
            
            [tracks, tracksThatLeft] = updateUnassignedTracks_wholevideo(tracks,tracksThatLeft,unassignedTracks);
            
            [tracks, nextId] = createNewTracks_wholevideo2(tracks,nextId,x_shift,y_shift,unassignedDetections,centroids,bboxes,cropworms,cropworms_orig,splines,end1s, end2s, end1s_g, end2s_g, omegas,curvatures, posture_angles, curr_frame,videoframe,videoname);
            
            curr_frame = curr_frame + 1;
            videoframe = videoframe +1;
            ev_ho_dict(curr_frame) = bgvidindex;
            
        end %END MAIN LOOP
        if curr_frame<length(ev_ho_dict) %chop down ev_ho_dict to the correct size
            ev_ho_dict = ev_ho_dict(1:curr_frame);
        end
        
        
        allTracks = [tracksThatLeft,tracks];
        WorthSavingTracksThatLeft_idx = [allTracks(:).age]'>9; %double check that all tracks are still long enough -- sometimes the last track is too short to be processed
        allTracks = allTracks(WorthSavingTracksThatLeft_idx);
        save([videoname '_' lawn_string '_allTracks.mat'],'allTracks','ev_ho_dict','bg_struct','pixpermm','videoname','lawn_string');
        
    end
    %%% DO POST-PROCESSING ON TRACKS TO DERIVE SPEED, ANGULAR VELOCITY --> ROAMING / DWELLING, LAWN LEAVING EVENTS
    [TRACKS, EXIT_STRUCT, POKE_STRUCT, SUMMARY_STRUCT] = tracks_postprocessing6( allTracks, ev_ho_dict, bg_struct, pixpermm, [videoname(1:end-4) '_' lawn_string] );
    
    %%% SAVING!
    fields2remove = {'cropworm','cropworm_orig'}; %<--- USE THIS ONE FOR ALL FAST DATA MANIPULATIONS
    TRACKS_slim = rmfield(TRACKS,fields2remove);
    
    save([videoname '_' lawn_string '_BACKGROUND.mat'],'bg_struct');
    save([videoname '_' lawn_string '_FINAL' '.mat'],'TRACKS_slim','EXIT_STRUCT','POKE_STRUCT','SUMMARY_STRUCT','bg_struct');
    save([videoname '_' lawn_string '_SUMMARY.mat'],'SUMMARY_STRUCT');
    save([videoname '_' lawn_string '_TRACKS_CROPWORMS.mat'],'TRACKS','-v7.3');
    
end

%%%%% NESTED FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [wormskel, pt1, pt2, pt1_g, pt2_g, omega] = segment_worm()
        pt1_g = NaN; %default values, only to be overridden if lookback is active
        pt2_g = NaN;
%         x_offset = bbox_crop(1); y_offset = bbox_crop(2);
        [wormskel, pt1, pt2, omega] = segWormOnly_Elias2(wormcrop, x_offset, y_offset, pixpermm);
        %rarely, the pt1 and pt2 can have a 0 coordinate due to rounding
        %errors: if this is the case, just make it 1.
%         pt1(pt1==0)=1; pt2(pt2==0)=1;
        if lookback && sum(isnan(pt1))==0 && sum(isnan(pt2))==0
            pt1_g = curr_lookback_frame(pt1(2),pt1(1)); %get the grayscale value associated with the head and tail - remember this is image coordinates, have to reverse again!
            pt2_g = curr_lookback_frame(pt2(2),pt2(1));
        end
    end

end
