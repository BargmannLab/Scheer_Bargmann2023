function mergedata_newFeatures_addFilestoMat_082421()
%mergedata_newFeatures_addFilestoMat_040121.m
%   This function loops over _FINAL.mat files in a directory and processes
%   them to derive new fields using NewFeaturesPostProcessing_021321.m
%   Adds to a pre-existing newFeatures.mat file

[featuresMatFile_filename, featuresMatFile_path, ~] = uigetfile({'*newFeatures.mat'},'Select newFeatures.mat file that you would like to add to.');
featuresMatFile = fullfile(featuresMatFile_path,featuresMatFile_filename);

[~, pathname, ~] = uigetfile({'*'},'Select folder where new files to add are.'); %this is nice because you can see which ones are in progress
cd(pathname);
files = dir('*FINAL*.mat');
files = {files.name}';

outpath = featuresMatFile_path; %save file in the same folder as the original new Features file

%load pre-existing features from mat file.
load(featuresMatFile,'MATFILENAME','PIXPERMM','VIDEONAME','VIDEOFRAME','BGVIDINDEX','SPLINE_x','SPLINE_y',...
    'POSTURE_ANGLE','OMEGA','CENTROID_bbox_x','CENTROID_bbox_y','HEAD_tip_x','HEAD_tip_y','TAIL_tip_x','TAIL_tip_y', ...
    'SPEED_bbox','ANGSPEED_bbox','FORWARD_bbox','REVERSE_bbox','PAUSE_bbox','MSD_bbox','FULLYINLAWN_bbox',...
    'HEADPOKE_FWD_tip','HEADPOKE_REV_tip','HEADPOKE_PAUSE_tip','RADIAL_DIST_tip','EV_HO_DIST_tip','HEADPOKE_ANGLE_tip',...
    'LAWN_EXIT_v0','LAWN_ENTRY_v0','IN_OR_OUT_v0',...
    'Lawn_Entry','Lawn_Exit','In_Or_Out','HeadPokeIntervals','HeadPokesAll','HeadPokeFwd','HeadPokeRev','HeadPokePause',...
    'HeadPokeDist','HeadPokeAngle','HeadPokeSpeed','centMissing','splineMissing',...
    'Head_cent_x','Head_cent_y','Head_cent_smth_x','Head_cent_smth_y',...
    'Midbody_cent_x','Midbody_cent_y','Midbody_cent_smth_x','Midbody_cent_smth_y',...
    'Tail_cent_x','Tail_cent_y','Tail_cent_smth_x','Tail_cent_smth_y',...
    'MovingForward','MovingReverse','Pause','Head_speed','Head_angspeed','Centroid_speed','Midbody_speed','Midbody_angspeed','Tail_speed','Tail_angspeed',...
    'headAngVel_relMid','headRadVel_relMid','Center_Point_x','Center_Point_y','Lawn_Boundary_Pts_x','Lawn_Boundary_Pts_y',...
    'Radial_Dist','Lawn_Boundary_Dist','HeadInLawn','MidbodyInLawn','TailInLawn','radTrajAngle','Head_grayscale','Head_grayscale_v0','Centroid_grayscale','Quirkiness',...
    'Centroid_Radial_Dist','Centroid_Lawn_Boundary_Dist','CentroidInLawn','Head_norm_grayscale','Centroid_norm_grayscale','Grayscale_bounds','mean_grayscale_profile','mean_norm_grayscale_profile','LBD_alignedTo_gs_profile');
%make sure that VIDEONAME is just column of the *first* video names
if size(VIDEONAME,2)>1
    VIDEONAME = VIDEONAME(:,1);
end

%declare all variables that will go in to hdf5 file
runLen = 7200;
PIXPERMM_new = zeros(length(files),1);
DATE_new = cell(length(files),1);
MATFILENAME_new = cell(length(files),1);
VIDEONAME_new = cell(length(files),1);
VIDEOFRAME_new = zeros(length(files),runLen); %cell array of strings (# rows = # animals, # cols = # datapoints)
BGVIDINDEX_new = zeros(length(files),runLen);
SPLINE_x_new = zeros(length(files)*49,runLen);
SPLINE_y_new = zeros(length(files)*49,runLen);
POSTURE_ANGLE_new = zeros(length(files)*48,runLen);
OMEGA_new = zeros(length(files),runLen);

%old features (can make a group for them)
CENTROID_bbox_x_new         = zeros(length(files),runLen);
CENTROID_bbox_y_new         = zeros(length(files),runLen);
HEAD_tip_x_new              = zeros(length(files),runLen);
HEAD_tip_y_new              = zeros(length(files),runLen);
TAIL_tip_x_new              = zeros(length(files),runLen);
TAIL_tip_y_new              = zeros(length(files),runLen);
SPEED_bbox_new              = zeros(length(files),runLen);
ANGSPEED_bbox_new           = zeros(length(files),runLen);
FORWARD_bbox_new            = false(length(files),runLen);
REVERSE_bbox_new            = false(length(files),runLen);
MSD_bbox_new                = zeros(length(files),runLen);
FULLYINLAWN_bbox_new        = false(length(files),runLen);
HEADPOKE_FWD_tip_new        = false(length(files),runLen);
HEADPOKE_REV_tip_new        = false(length(files),runLen);
HEADPOKE_PAUSE_tip_new      = false(length(files),runLen);
RADIAL_DIST_tip_new         = zeros(length(files),runLen);
EV_HO_DIST_tip_new          = zeros(length(files),runLen);
HEADPOKE_ANGLE_tip_new      = zeros(length(files),runLen);
LAWN_EXIT_v0_new            = false(length(files),runLen);
LAWN_ENTRY_v0_new           = false(length(files),runLen);
IN_OR_OUT_v0_new            = false(length(files),runLen);

%new features
Lawn_Entry_new              = false(length(files),runLen);
Lawn_Exit_new               = false(length(files),runLen);
In_Or_Out_new               = false(length(files),runLen);
HeadPokeIntervals_new       = false(length(files),runLen);
HeadPokesAll_new            = false(length(files),runLen);
HeadPokeFwd_new             = false(length(files),runLen);
HeadPokeRev_new             = false(length(files),runLen);
HeadPokePause_new           = false(length(files),runLen);
HeadPokeDist_new            = zeros(length(files),runLen);
HeadPokeAngle_new           = zeros(length(files),runLen);
HeadPokeSpeed_new           = zeros(length(files),runLen);
centMissing_new             = false(length(files),runLen);
splineMissing_new           = false(length(files),runLen);
Head_cent_x_new             = zeros(length(files),runLen);
Head_cent_y_new             = zeros(length(files),runLen);
Head_cent_smth_x_new        = zeros(length(files),runLen);
Head_cent_smth_y_new        = zeros(length(files),runLen);
Midbody_cent_x_new          = zeros(length(files),runLen);
Midbody_cent_y_new          = zeros(length(files),runLen);
Midbody_cent_smth_x_new     = zeros(length(files),runLen);
Midbody_cent_smth_y_new     = zeros(length(files),runLen);
Tail_cent_x_new             = zeros(length(files),runLen);
Tail_cent_y_new             = zeros(length(files),runLen);
Tail_cent_smth_x_new        = zeros(length(files),runLen);
Tail_cent_smth_y_new        = zeros(length(files),runLen);
MovingForward_new           = false(length(files),runLen);
MovingReverse_new           = false(length(files),runLen);
Head_speed_new              = zeros(length(files),runLen);
Head_angspeed_new           = zeros(length(files),runLen);
Midbody_speed_new           = zeros(length(files),runLen);
Midbody_angspeed_new        = zeros(length(files),runLen);
Tail_speed_new              = zeros(length(files),runLen);
Tail_angspeed_new           = zeros(length(files),runLen);
Centroid_speed_new          = zeros(length(files),runLen);
headAngVel_relMid_new       = zeros(length(files),runLen);
headRadVel_relMid_new       = zeros(length(files),runLen);
Center_Point_x_new          = zeros(length(files),20); %overkill but will fit any number of bg_struct videos
Center_Point_y_new          = zeros(length(files),20);
Lawn_Boundary_Pts_x_new     = zeros(length(files)*360,20); %there are 360 points in each lawn boundary
Lawn_Boundary_Pts_y_new     = zeros(length(files)*360,20);
Radial_Dist_new             = zeros(length(files),runLen);
Lawn_Boundary_Dist_new      = zeros(length(files),runLen);
HeadInLawn_new              = false(length(files),runLen);
MidbodyInLawn_new           = false(length(files),runLen);
TailInLawn_new              = false(length(files),runLen);
radTrajAngle_new            = zeros(length(files),runLen); %the angle of animal trajectory relative to radial vector
Head_grayscale_new          = zeros(length(files),runLen);
Head_grayscale_v0_new       = zeros(length(files),runLen);
Centroid_grayscale_new      = zeros(length(files),runLen);
Quirkiness_new              = zeros(length(files),runLen);
%new fields 08/24/21
Centroid_Radial_Dist_new        = zeros(length(files),runLen);
Centroid_Lawn_Boundary_Dist_new = zeros(length(files),runLen);
CentroidInLawn_new              = zeros(length(files),runLen);
Head_norm_grayscale_new         = zeros(length(files),runLen);
Centroid_norm_grayscale_new     = zeros(length(files),runLen);

Grayscale_bounds_new            = zeros(length(files),3);
mean_grayscale_profile_new      = zeros(length(files),1000);
mean_norm_grayscale_profile_new = zeros(length(files),1000);
LBD_alignedTo_gs_profile_new    = zeros(length(files),1000);

for i = 1:length(files)
    disp(files{i});
    
    
    tmpSummaryStruct = load(files{i},'SUMMARY_STRUCT');
    SUMMARY_STRUCT = tmpSummaryStruct.SUMMARY_STRUCT;
    tmpbgstruct = load(files{i},'bg_struct');
    bg_struct = tmpbgstruct.bg_struct;
    OUT_STRUCT = NewFeaturesPostProcessing_082421( bg_struct, SUMMARY_STRUCT );
    
    PIXPERMM_new(i)                = single(OUT_STRUCT.PIXPERMM)';
    MATFILENAME_new(i)             = files(i); %the name of the FINAL.mat file where we got the data
    VIDEONAME_new(i)             = OUT_STRUCT.VIDEONAME(1);
    %find datestring within videoname
    vn1 = VIDEONAME_new{i};
    [startIndex,endIndex] = regexp(vn1,'_[0-9]{6}_');
    datestring = vn1(startIndex+1:endIndex-1);
    DATE_new(i)                    = {datestring};
    VIDEOFRAME_new(i,:)            = OUT_STRUCT.VIDEOFRAME';
    BGVIDINDEX_new(i,:)            = OUT_STRUCT.BGVIDINDEX';
    
    bins = 1:49:((length(files)+1)*49); %indexing system for Spline points (49 rows per animal)
    startTostop = [bins(1:end-1)' bins(2:end)'-1];
    SPLINE_x_new(startTostop(i,1):startTostop(i,2),:) = single(OUT_STRUCT.SPLINE_x)';
    SPLINE_y_new(startTostop(i,1):startTostop(i,2),:) = single(OUT_STRUCT.SPLINE_y)';
    
    bins = 1:48:((length(files)+1)*48); %indexing system for Posture_Angle (48 rows per animal)
    startTostop = [bins(1:end-1)' bins(2:end)'-1];
    POSTURE_ANGLE_new(startTostop(i,1):startTostop(i,2),:) = single(OUT_STRUCT.POSTURE_ANGLE)';
    
    CENTROID_bbox_x_new(i,:)       = single(OUT_STRUCT.CENTROID(:,1))';
    CENTROID_bbox_y_new(i,:)       = single(OUT_STRUCT.CENTROID(:,2))';
    HEAD_tip_x_new(i,:)            = single(OUT_STRUCT.HEAD(:,1))';
    HEAD_tip_y_new(i,:)            = single(OUT_STRUCT.HEAD(:,2))';
    TAIL_tip_x_new(i,:)            = single(OUT_STRUCT.TAIL(:,1))';
    TAIL_tip_y_new(i,:)            = single(OUT_STRUCT.TAIL(:,2))';
    SPEED_bbox_new(i,:)            = single(OUT_STRUCT.SPEED)';
    ANGSPEED_bbox_new(i,:)         = single(OUT_STRUCT.ANGSPEED)';
    FORWARD_bbox_new(i,:)          = logical(OUT_STRUCT.FORWARD)';
    REVERSE_bbox_new(i,:)          = logical(OUT_STRUCT.REVERSE)';
    OMEGA_new(i,:)                 = logical(OUT_STRUCT.OMEGA)';
    MSD_bbox_new(i,:)              = single(OUT_STRUCT.MSD)';
    FULLYINLAWN_bbox_new(i,:)      = logical(OUT_STRUCT.FULLYINLAWN)';
    HEADPOKE_FWD_tip_new(i,:)      = logical(OUT_STRUCT.HEADPOKE_FWD)';
    HEADPOKE_REV_tip_new(i,:)      = logical(OUT_STRUCT.HEADPOKE_REV)';
    HEADPOKE_PAUSE_tip_new(i,:)    = logical(OUT_STRUCT.HEADPOKE_PAUSE)';
    RADIAL_DIST_tip_new(i,:)       = single(OUT_STRUCT.RADIAL_DIST)';
    EV_HO_DIST_tip_new(i,:)        = single(OUT_STRUCT.EV_HO_DIST)';
    HEADPOKE_ANGLE_tip_new(i,:)    = single(OUT_STRUCT.HEADPOKE_ANGLE)';
    LAWN_ENTRY_v0_new(i,:)         = logical(OUT_STRUCT.LAWN_ENTRY)';
    LAWN_EXIT_v0_new(i,:)          = logical(OUT_STRUCT.LAWN_EXIT)';
    IN_OR_OUT_v0_new(i,:)          = logical(fillmissing(OUT_STRUCT.IN_OR_OUT,'previous'))';
    
    %new fields
    Lawn_Entry_new(i,:)            = logical(OUT_STRUCT.Lawn_Entry)';
    Lawn_Exit_new(i,:)             = logical(OUT_STRUCT.Lawn_Exit)';
    In_Or_Out_new(i,:)             = logical(OUT_STRUCT.In_Or_Out)';
    HeadPokeIntervals_new(i,:)     = logical(OUT_STRUCT.HeadPokeIntervals)';
    HeadPokesAll_new(i,:)          = logical(OUT_STRUCT.HeadPokesAll)';
    HeadPokeFwd_new(i,:)           = logical(OUT_STRUCT.HeadPokeFwd)';
    HeadPokeRev_new(i,:)           = logical(OUT_STRUCT.HeadPokeRev)';
    HeadPokePause_new(i,:)         = logical(OUT_STRUCT.HeadPokePause)';
    HeadPokeDist_new(i,:)          = single(OUT_STRUCT.HeadPokeDist)';
    HeadPokeAngle_new(i,:)         = single(OUT_STRUCT.HeadPokeAngle)';
    HeadPokeSpeed_new(i,:)         = single(OUT_STRUCT.HeadPokeSpeed)';
    centMissing_new(i,:)           = logical(OUT_STRUCT.centMissing)';
    splineMissing_new(i,:)         = logical(OUT_STRUCT.splineMissing)';
    Head_cent_x_new(i,:)           = single(OUT_STRUCT.Head_cent(:,1))';
    Head_cent_y_new(i,:)           = single(OUT_STRUCT.Head_cent(:,2))';
    Head_cent_smth_x_new(i,:)      = single(OUT_STRUCT.Head_cent_smth(:,1))';
    Head_cent_smth_y_new(i,:)      = single(OUT_STRUCT.Head_cent_smth(:,2))';
    Midbody_cent_x_new(i,:)        = single(OUT_STRUCT.Midbody_cent(:,1))';
    Midbody_cent_y_new(i,:)        = single(OUT_STRUCT.Midbody_cent(:,2))';
    Midbody_cent_smth_x_new(i,:)   = single(OUT_STRUCT.Midbody_cent_smth(:,1))';
    Midbody_cent_smth_y_new(i,:)   = single(OUT_STRUCT.Midbody_cent_smth(:,2))';
    Tail_cent_x_new(i,:)           = single(OUT_STRUCT.Tail_cent(:,1))';
    Tail_cent_y_new(i,:)           = single(OUT_STRUCT.Tail_cent(:,2))';
    Tail_cent_smth_x_new(i,:)      = single(OUT_STRUCT.Tail_cent_smth(:,1))';
    Tail_cent_smth_y_new(i,:)      = single(OUT_STRUCT.Tail_cent_smth(:,2))';
    MovingForward_new(i,:)         = logical(OUT_STRUCT.MovingForward)';
    MovingReverse_new(i,:)         = logical(OUT_STRUCT.MovingForward)';
    Head_speed_new(i,:)            = single(OUT_STRUCT.Head_speed)';
    Head_angspeed_new(i,:)         = single(OUT_STRUCT.Head_angspeed)';
    Midbody_speed_new(i,:)         = single(OUT_STRUCT.Midbody_speed)';
    Midbody_angspeed_new(i,:)      = single(OUT_STRUCT.Midbody_angspeed)';
    
    
    Tail_speed_new(i,:)            = single(OUT_STRUCT.Tail_speed)';
    Tail_angspeed_new(i,:)         = single(OUT_STRUCT.Tail_angspeed)';
    Centroid_speed_new(i,:)        = single(OUT_STRUCT.Centroid_speed)';
    headAngVel_relMid_new(i,:)     = single(OUT_STRUCT.headAngVel_relMid)';
    headRadVel_relMid_new(i,:)     = single(OUT_STRUCT.headRadVel_relMid)';
    
    centerPt = OUT_STRUCT.Center_Point;
    Center_Point_x_new(i,1:length(centerPt)) = single(centerPt(:,1))';
    Center_Point_y_new(i,1:length(centerPt)) = single(centerPt(:,2))';
    
    bins = 1:360:((length(files)+1)*360); %indexing system for Lawn Boundary Points (360 rows per animal x # of bg_struct entries per SUMMARY_STRUCT)
    startTostop = [bins(1:end-1)' bins(2:end)'-1];
    lawnBoundaryPtsX = OUT_STRUCT.Lawn_Boundary_Pts_x';
    lawnBoundaryPtsY = OUT_STRUCT.Lawn_Boundary_Pts_y';
    Lawn_Boundary_Pts_x_new(startTostop(i,1):startTostop(i,2),1:size(lawnBoundaryPtsX,2)) = single(lawnBoundaryPtsX);
    Lawn_Boundary_Pts_y_new(startTostop(i,1):startTostop(i,2),1:size(lawnBoundaryPtsY,2)) = single(lawnBoundaryPtsY);
    
    Radial_Dist_new(i,:)           = single(OUT_STRUCT.Radial_Dist)';
    Lawn_Boundary_Dist_new(i,:)    = single(OUT_STRUCT.Lawn_Boundary_Dist)';
    HeadInLawn_new(i,:)            = logical(OUT_STRUCT.HeadInLawn)';
    MidbodyInLawn_new(i,:)         = logical(OUT_STRUCT.MidbodyInLawn)';
    TailInLawn_new(i,:)            = logical(OUT_STRUCT.TailInLawn)';
    radTrajAngle_new(i,:)          = single(OUT_STRUCT.radTrajAngle)';
    Head_grayscale_new(i,:)        = single(OUT_STRUCT.Head_grayscale)';
    Head_grayscale_v0_new(i,:)     = single(OUT_STRUCT.Head_grayscale_v0)';
    Centroid_grayscale_new(i,:)    = single(OUT_STRUCT.Centroid_grayscale)';
    Quirkiness_new(i,:)            = single(OUT_STRUCT.Quirkiness)';
    
    %new fields 08/24/21
    Centroid_Radial_Dist_new(i,:)  = single(OUT_STRUCT.Centroid_Radial_Dist)';
    Centroid_Lawn_Boundary_Dist_new(i,:) = single(OUT_STRUCT.Centroid_Lawn_Boundary_Dist)';
    CentroidInLawn_new(i,:)          = logical(OUT_STRUCT.CentroidInLawn)';
    Head_norm_grayscale_new(i,:)     = single(OUT_STRUCT.Head_norm_grayscale)';
    Centroid_norm_grayscale_new(i,:) = single(OUT_STRUCT.Centroid_norm_grayscale)';
    
    Grayscale_bounds_new(i,:)        = single(OUT_STRUCT.Grayscale_bounds)';
    mean_grayscale_profile_new(i,:)  = single(OUT_STRUCT.mean_grayscale_profile)';
    mean_norm_grayscale_profile_new(i,:) = single(OUT_STRUCT.mean_norm_grayscale_profile)';
    LBD_alignedTo_gs_profile_new(i,:) = single(OUT_STRUCT.LBD_alignedTo_gs_profile)';
    
    %     toc;
end
%derive pausing
PAUSE_bbox_new = abs(SPEED_bbox_new)<=0.02;
Pause_new = abs(Midbody_speed_new)<=0.02;

%% add the new fields to old ones:
PIXPERMM = [PIXPERMM; PIXPERMM_new];
MATFILENAME = [MATFILENAME; MATFILENAME_new];
VIDEONAME = [VIDEONAME; VIDEONAME_new];
VIDEOFRAME = [VIDEOFRAME; VIDEOFRAME_new];
BGVIDINDEX = [BGVIDINDEX; BGVIDINDEX_new];
SPLINE_x = [SPLINE_x; SPLINE_x_new];
SPLINE_y = [SPLINE_y; SPLINE_y_new];
POSTURE_ANGLE = [POSTURE_ANGLE; POSTURE_ANGLE_new];
CENTROID_bbox_x = [CENTROID_bbox_x; CENTROID_bbox_x_new];
CENTROID_bbox_y = [CENTROID_bbox_y; CENTROID_bbox_y_new];
HEAD_tip_x = [HEAD_tip_x; HEAD_tip_x_new];
HEAD_tip_y = [HEAD_tip_y; HEAD_tip_y_new];
TAIL_tip_x = [TAIL_tip_x; TAIL_tip_x_new];
TAIL_tip_y = [TAIL_tip_y; TAIL_tip_y_new];
SPEED_bbox = [SPEED_bbox; SPEED_bbox_new];
ANGSPEED_bbox = [ANGSPEED_bbox; ANGSPEED_bbox_new];
FORWARD_bbox = [FORWARD_bbox; FORWARD_bbox_new];
REVERSE_bbox = [REVERSE_bbox; REVERSE_bbox_new];
PAUSE_bbox = [PAUSE_bbox; PAUSE_bbox_new];
OMEGA = [OMEGA; OMEGA_new];
MSD_bbox = [MSD_bbox; MSD_bbox_new];
FULLYINLAWN_bbox = [FULLYINLAWN_bbox; FULLYINLAWN_bbox_new];
HEADPOKE_FWD_tip = [HEADPOKE_FWD_tip; HEADPOKE_FWD_tip_new];
HEADPOKE_REV_tip = [HEADPOKE_REV_tip; HEADPOKE_REV_tip_new];
HEADPOKE_PAUSE_tip = [HEADPOKE_PAUSE_tip; HEADPOKE_PAUSE_tip_new];
RADIAL_DIST_tip = [RADIAL_DIST_tip; RADIAL_DIST_tip_new];
EV_HO_DIST_tip = [EV_HO_DIST_tip; EV_HO_DIST_tip_new];
HEADPOKE_ANGLE_tip = [HEADPOKE_ANGLE_tip; HEADPOKE_ANGLE_tip_new];
LAWN_ENTRY_v0 = [LAWN_ENTRY_v0; LAWN_ENTRY_v0_new];
LAWN_EXIT_v0 = [LAWN_EXIT_v0; LAWN_EXIT_v0_new];
IN_OR_OUT_v0 = [IN_OR_OUT_v0; IN_OR_OUT_v0_new];
Lawn_Entry = [Lawn_Entry; Lawn_Entry_new];
Lawn_Exit = [Lawn_Exit; Lawn_Exit_new];
In_Or_Out = [In_Or_Out; In_Or_Out_new];
HeadPokeIntervals = [HeadPokeIntervals; HeadPokeIntervals_new];
HeadPokesAll = [HeadPokesAll; HeadPokesAll_new];
HeadPokeFwd = [HeadPokeFwd; HeadPokeFwd_new];
HeadPokeRev = [HeadPokeRev; HeadPokeRev_new];
HeadPokePause = [HeadPokePause; HeadPokePause_new];
HeadPokeDist = [HeadPokeDist; HeadPokeDist_new];
HeadPokeAngle = [HeadPokeAngle; HeadPokeAngle_new];
HeadPokeSpeed = [HeadPokeSpeed; HeadPokeSpeed_new];
centMissing = [centMissing; centMissing_new];
splineMissing = [splineMissing; splineMissing_new];
Head_cent_x = [Head_cent_x; Head_cent_x_new];
Head_cent_y = [Head_cent_y; Head_cent_y_new];
Head_cent_smth_x = [Head_cent_smth_x; Head_cent_smth_x_new];
Head_cent_smth_y = [Head_cent_smth_y; Head_cent_smth_y_new];
Midbody_cent_x = [Midbody_cent_x; Midbody_cent_x_new];
Midbody_cent_y = [Midbody_cent_y; Midbody_cent_y_new];
Midbody_cent_smth_x = [Midbody_cent_smth_x; Midbody_cent_smth_x_new];
Midbody_cent_smth_y = [Midbody_cent_smth_y; Midbody_cent_smth_y_new];
Tail_cent_x = [Tail_cent_x; Tail_cent_x_new];
Tail_cent_y = [Tail_cent_y; Tail_cent_y_new];
Tail_cent_smth_x = [Tail_cent_smth_x; Tail_cent_smth_x_new];
Tail_cent_smth_y = [Tail_cent_smth_y; Tail_cent_smth_y_new];
MovingForward = [MovingForward; MovingForward_new];
MovingReverse = [MovingReverse; MovingReverse_new];
Pause = [Pause; Pause_new];
Head_speed = [Head_speed; Head_speed_new];
Head_angspeed = [Head_angspeed; Head_angspeed_new];
Midbody_speed = [Midbody_speed; Midbody_speed_new];
Midbody_angspeed = [Midbody_angspeed; Midbody_angspeed_new];
Tail_speed = [Tail_speed; Tail_speed_new];
Tail_angspeed = [Tail_angspeed; Tail_angspeed_new];
Centroid_speed = [Centroid_speed; Centroid_speed_new];
headAngVel_relMid = [headAngVel_relMid; headAngVel_relMid_new];
headRadVel_relMid = [headRadVel_relMid; headRadVel_relMid_new];
Center_Point_x = [Center_Point_x; Center_Point_x_new];
Center_Point_y = [Center_Point_y; Center_Point_y_new];
Lawn_Boundary_Pts_x = [Lawn_Boundary_Pts_x; Lawn_Boundary_Pts_x_new];
Lawn_Boundary_Pts_y = [Lawn_Boundary_Pts_y; Lawn_Boundary_Pts_y_new];
Radial_Dist = [Radial_Dist; Radial_Dist_new];
Lawn_Boundary_Dist = [Lawn_Boundary_Dist; Lawn_Boundary_Dist_new];
HeadInLawn = [HeadInLawn; HeadInLawn_new];
MidbodyInLawn = [MidbodyInLawn; MidbodyInLawn_new];
TailInLawn = [TailInLawn; TailInLawn_new];
radTrajAngle = [radTrajAngle; radTrajAngle_new];
Head_grayscale = [Head_grayscale; Head_grayscale_new];
Head_grayscale_v0 = [Head_grayscale_v0; Head_grayscale_v0_new];
Centroid_grayscale = [Centroid_grayscale; Centroid_grayscale_new];
Quirkiness = [Quirkiness; Quirkiness_new];
%new 08/24/21
Centroid_Radial_Dist = [Centroid_Radial_Dist; Centroid_Radial_Dist_new];
Centroid_Lawn_Boundary_Dist = [Centroid_Lawn_Boundary_Dist; Centroid_Lawn_Boundary_Dist_new];
CentroidInLawn = [CentroidInLawn; CentroidInLawn_new];
Head_norm_grayscale = [Head_norm_grayscale; Head_norm_grayscale_new];
Centroid_norm_grayscale = [Centroid_norm_grayscale; Centroid_norm_grayscale_new];
Grayscale_bounds = [Grayscale_bounds; Grayscale_bounds_new];
mean_grayscale_profile = [mean_grayscale_profile; mean_grayscale_profile_new];
mean_norm_grayscale_profile = [mean_norm_grayscale_profile; mean_norm_grayscale_profile_new];
LBD_alignedTo_gs_profile = [LBD_alignedTo_gs_profile; LBD_alignedTo_gs_profile_new];

%% ensure that we haven't duplicated any files by comparing the first videonames

firstVids = VIDEONAME;
[~,ia,~] = unique(firstVids);
if length(ia)<length(firstVids)
    warning('There are matfiles corresponding to the same videos -- moving duplicates to subfolder!');
    mkdir('duplicates');
    MATFILENAME = MATFILENAME(ia,:);
    MATFILENAME_DUPLICATES = MATFILENAME(setdiff(1:length(MATFILENAME),ia),:);
    %move duplicates to the duplicates folder
    for k = 1:length(MATFILENAME_DUPLICATES)
        movefile(MATFILENAME_DUPLICATES{k}, 'duplicates');
    end
    
    PIXPERMM = PIXPERMM(ia,:);
    VIDEONAME = VIDEONAME(ia,:);
    VIDEOFRAME = VIDEOFRAME(ia,:);
    BGVIDINDEX = BGVIDINDEX(ia,:);
    SPLINE_x = SPLINE_x(ia,:);
    SPLINE_y = SPLINE_y(ia,:);
    POSTURE_ANGLE = POSTURE_ANGLE(ia,:);
    OMEGA = OMEGA(ia,:);
    CENTROID_bbox_x = CENTROID_bbox_x(ia,:);
    CENTROID_bbox_y = CENTROID_bbox_y(ia,:);
    HEAD_tip_x = HEAD_tip_x(ia,:);
    HEAD_tip_y = HEAD_tip_y(ia,:);
    TAIL_tip_x = TAIL_tip_x(ia,:);
    TAIL_tip_y = TAIL_tip_y(ia,:);
    
    SPEED_bbox = SPEED_bbox(ia,:);
    ANGSPEED_bbox = ANGSPEED_bbox(ia,:);
    FORWARD_bbox = FORWARD_bbox(ia,:);
    REVERSE_bbox = REVERSE_bbox(ia,:);
    PAUSE_bbox = PAUSE_bbox(ia,:);
    MSD_bbox = MSD_bbox(ia,:);
    FULLYINLAWN_bbox = FULLYINLAWN_bbox(ia,:);
    HEADPOKE_FWD_tip = HEADPOKE_FWD_tip(ia,:);
    HEADPOKE_REV_tip = HEADPOKE_REV_tip(ia,:);
    HEADPOKE_PAUSE_tip = HEADPOKE_PAUSE_tip(ia,:);
    RADIAL_DIST_tip = RADIAL_DIST_tip(ia,:);
    EV_HO_DIST_tip = EV_HO_DIST_tip(ia,:);
    HEADPOKE_ANGLE_tip = HEADPOKE_ANGLE_tip(ia,:);
    LAWN_EXIT_v0 = LAWN_EXIT_v0(ia,:);
    LAWN_ENTRY_v0 = LAWN_ENTRY_v0(ia,:);
    IN_OR_OUT_v0 = IN_OR_OUT_v0(ia,:);
    Lawn_Entry = Lawn_Entry(ia,:);
    Lawn_Exit = Lawn_Exit(ia,:);
    In_Or_Out = In_Or_Out(ia,:);
    HeadPokeIntervals = HeadPokeIntervals(ia,:);
    HeadPokesAll = HeadPokesAll(ia,:);
    HeadPokeFwd = HeadPokeFwd(ia,:);
    HeadPokeRev = HeadPokeRev(ia,:);
    HeadPokePause = HeadPokePause(ia,:);
    
    HeadPokeDist = HeadPokeDist(ia,:);
    HeadPokeAngle = HeadPokeAngle(ia,:);
    HeadPokeSpeed = HeadPokeSpeed(ia,:);
    centMissing = centMissing(ia,:);
    splineMissing = splineMissing(ia,:);
    
    Head_cent_x = Head_cent_x(ia,:);
    Head_cent_y = Head_cent_y(ia,:);
    Head_cent_smth_x = Head_cent_smth_x(ia,:);
    Head_cent_smth_y = Head_cent_smth_y(ia,:);
    
    Midbody_cent_x = Midbody_cent_x(ia,:);
    Midbody_cent_y = Midbody_cent_y(ia,:);
    Midbody_cent_smth_x = Midbody_cent_smth_x(ia,:);
    Midbody_cent_smth_y = Midbody_cent_smth_y(ia,:);
    
    Tail_cent_x = Tail_cent_x(ia,:);
    Tail_cent_y = Tail_cent_y(ia,:);
    Tail_cent_smth_x = Tail_cent_smth_x(ia,:);
    Tail_cent_smth_y = Tail_cent_smth_y(ia,:);
    
    MovingForward = MovingForward(ia,:);
    MovingReverse = MovingReverse(ia,:);
    Pause = Pause(ia,:);
    Head_speed = Head_speed(ia,:);
    Head_angspeed = Head_angspeed(ia,:);
    Midbody_speed = Midbody_speed(ia,:);
    Midbody_angspeed = Midbody_angspeed(ia,:);
    Tail_speed = Tail_speed(ia,:);
    Tail_angspeed = Tail_angspeed(ia,:);
    Centroid_speed = Centroid_speed(ia,:);
    
    headAngVel_relMid = headAngVel_relMid(ia,:);
    headRadVel_relMid = headRadVel_relMid(ia,:);
    Center_Point_x = Center_Point_x(ia,:);
    Center_Point_y = Center_Point_y(ia,:);
    Lawn_Boundary_Pts_x = Lawn_Boundary_Pts_x(ia,:);
    Lawn_Boundary_Pts_y = Lawn_Boundary_Pts_y(ia,:);
    
    Radial_Dist = Radial_Dist(ia,:);
    Lawn_Boundary_Dist = Lawn_Boundary_Dist(ia,:);
    HeadInLawn = HeadInLawn(ia,:);
    MidbodyInLawn = MidbodyInLawn(ia,:);
    TailInLawn = TailInLawn(ia,:);
    radTrajAngle = radTrajAngle(ia,:);
    Head_grayscale = Head_grayscale(ia,:);
    Head_grayscale_v0 = Head_grayscale_v0(ia,:);
    Centroid_grayscale = Centroid_grayscale(ia,:);
    Quirkiness = Quirkiness(ia,:);
    
    %new 08/24/21
    Centroid_Radial_Dist = Centroid_Radial_Dist(ia,:);
    Centroid_Lawn_Boundary_Dist = Centroid_Lawn_Boundary_Dist(ia,:);
    CentroidInLawn = CentroidInLawn(ia,:);
    Head_norm_grayscale = Head_norm_grayscale(ia,:);
    Centroid_norm_grayscale = Centroid_norm_grayscale(ia,:);
    Grayscale_bounds = Grayscale_bounds(ia,:);
    mean_grayscale_profile = mean_grayscale_profile(ia,:);
    mean_norm_grayscale_profile = mean_norm_grayscale_profile(ia,:);
    LBD_alignedTo_gs_profile = LBD_alignedTo_gs_profile(ia,:);
    
end

%save all variables to a -v7.3 .mat file, which can be read in by h5py
%package in python.

%change to outpath #this will actually overwrite the previous files!
cd(outpath);
save(featuresMatFile_filename,...,
    'MATFILENAME','PIXPERMM','VIDEONAME','VIDEOFRAME','BGVIDINDEX','SPLINE_x','SPLINE_y',...
    'POSTURE_ANGLE','OMEGA','CENTROID_bbox_x','CENTROID_bbox_y','HEAD_tip_x','HEAD_tip_y','TAIL_tip_x','TAIL_tip_y', ...
    'SPEED_bbox','ANGSPEED_bbox','FORWARD_bbox','REVERSE_bbox','PAUSE_bbox','MSD_bbox','FULLYINLAWN_bbox',...
    'HEADPOKE_FWD_tip','HEADPOKE_REV_tip','HEADPOKE_PAUSE_tip','RADIAL_DIST_tip','EV_HO_DIST_tip','HEADPOKE_ANGLE_tip',...
    'LAWN_EXIT_v0','LAWN_ENTRY_v0','IN_OR_OUT_v0',...
    'Lawn_Entry','Lawn_Exit','In_Or_Out','HeadPokeIntervals','HeadPokesAll','HeadPokeFwd','HeadPokeRev','HeadPokePause',...
    'HeadPokeDist','HeadPokeAngle','HeadPokeSpeed','centMissing','splineMissing',...
    'Head_cent_x','Head_cent_y','Head_cent_smth_x','Head_cent_smth_y',...
    'Midbody_cent_x','Midbody_cent_y','Midbody_cent_smth_x','Midbody_cent_smth_y',...
    'Tail_cent_x','Tail_cent_y','Tail_cent_smth_x','Tail_cent_smth_y',...
    'MovingForward','MovingReverse','Pause','Head_speed','Head_angspeed','Centroid_speed','Midbody_speed','Midbody_angspeed','Tail_speed','Tail_angspeed',...
    'headAngVel_relMid','headRadVel_relMid','Center_Point_x','Center_Point_y','Lawn_Boundary_Pts_x','Lawn_Boundary_Pts_y',...
    'Radial_Dist','Lawn_Boundary_Dist','HeadInLawn','MidbodyInLawn','TailInLawn','radTrajAngle','Head_grayscale','Head_grayscale_v0','Centroid_grayscale','Quirkiness',...
    'Centroid_Radial_Dist','Centroid_Lawn_Boundary_Dist','CentroidInLawn','Head_norm_grayscale','Centroid_norm_grayscale','Grayscale_bounds','mean_grayscale_profile','mean_norm_grayscale_profile','LBD_alignedTo_gs_profile',...
    '-v7.3')

%save string data -- like MATFILENAME and VIDEONAME in a different way:
VN = VIDEONAME; %just first videoname
% Generate a file
fid = H5F.create([featuresMatFile_filename(1:strfind(featuresMatFile_filename,'_newFeatures.mat')) 'Filenames.h5'],'H5F_ACC_TRUNC',...
    'H5P_DEFAULT','H5P_DEFAULT');

% Set variable length string type
VLstr_type = H5T.copy('H5T_C_S1');
H5T.set_size(VLstr_type,'H5T_VARIABLE');

% Create a dataspace for cellstr
H5S_UNLIMITED = H5ML.get_constant_value('H5S_UNLIMITED');
dspace_vn = H5S.create_simple(1,numel(VN),H5S_UNLIMITED);
dspace_mfn = H5S.create_simple(1,numel(MATFILENAME),H5S_UNLIMITED);

% Create a dataset plist for chunking
plist = H5P.create('H5P_DATASET_CREATE');
H5P.set_chunk(plist,2); % 2 strings per chunk

% Create dataset
dset_vn = H5D.create(fid,'videoname',VLstr_type,dspace_vn,plist); %videoname dataset
dset_mfn = H5D.create(fid,'matfilename',VLstr_type,dspace_mfn,plist); %matfilename dataset

% Write data
H5D.write(dset_vn,VLstr_type,'H5S_ALL','H5S_ALL','H5P_DEFAULT',VN);
H5D.write(dset_mfn,VLstr_type,'H5S_ALL','H5S_ALL','H5P_DEFAULT',MATFILENAME);

% Close file & resources
H5P.close(plist);
H5T.close(VLstr_type);
H5S.close(dspace_vn);
H5S.close(dspace_mfn);
H5D.close(dset_vn);
H5D.close(dset_mfn);
H5F.close(fid);

cd(pathname); %return at the end
