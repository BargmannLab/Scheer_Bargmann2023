function mergedata_newFeatures_082421( strain_name )
%mergedata_newFeatures_tocsvs_021621.m
%   This function loops over _FINAL.mat files in a directory and processes
%   them to derive new fields using NewFeaturesPostProcessing_021321.m
%   Generates .mat files

timenow = datestr(now,'mm_dd_yy');
outpath = uigetdir(pwd,'Select Directory to Save Output.');

[~, pathname, ~] = uigetfile({'*'}); %this is nice because you can see which ones are in progress
cd(pathname);
files = dir('*FINAL*.mat');
files = {files.name}';

%first check if there are any duplicated files and move them to a
%duplicates folder before subsequent processing.
%comparing the first video names
VIDEONAME = cell(length(files),1); %cell array of strings (# rows = # animals, # cols = # datapoints)
MATFILENAME = cell(length(files),1);
for i = 1:length(files)
%     try
        tmpSummaryStruct = load(files{i},'SUMMARY_STRUCT');
        SUMMARY_STRUCT = tmpSummaryStruct.SUMMARY_STRUCT;
        MATFILENAME(i)             = files(i); %the name of the FINAL.mat file where we got the data
        VIDEONAME(i)             = SUMMARY_STRUCT.VIDEONAME(1);
%     catch
%         continue
%     end

end
firstVids = VIDEONAME;
[~,ia,~] = unique(firstVids);
if length(ia)<length(firstVids)
    warning('There are matfiles corresponding to the same videos -- moving duplicates to subfolder!');
    mkdir('duplicates');
%     MATFILENAME = MATFILENAME(ia,:);
    MATFILENAME_DUPLICATES = MATFILENAME(setdiff(1:length(MATFILENAME),ia),:);
    %move duplicates to the duplicates folder
    for k = 1:length(MATFILENAME_DUPLICATES)
        movefile(MATFILENAME_DUPLICATES{k}, 'duplicates');
    end
end

%then get the files again (unique list only) and proceed
files = dir('*FINAL*.mat');
files = {files.name}';

%declare all variables that will go in to hdf5 file
runLen = 7200;
PIXPERMM = zeros(length(files),1);
DATE = cell(length(files),1);
MATFILENAME = cell(length(files),1);
VIDEONAME = cell(length(files),1); %cell array of strings (# rows = # animals, # cols = # datapoints)
VIDEOFRAME = zeros(length(files),runLen);
BGVIDINDEX = zeros(length(files),runLen);
SPLINE_x = zeros(length(files)*49,runLen);
SPLINE_y = zeros(length(files)*49,runLen);
POSTURE_ANGLE = zeros(length(files)*48,runLen);
OMEGA = zeros(length(files),runLen);

%old features (can make a group for them)
CENTROID_bbox_x         = zeros(length(files),runLen);
CENTROID_bbox_y         = zeros(length(files),runLen);
HEAD_tip_x              = zeros(length(files),runLen);
HEAD_tip_y              = zeros(length(files),runLen);
TAIL_tip_x              = zeros(length(files),runLen);
TAIL_tip_y              = zeros(length(files),runLen);
SPEED_bbox              = zeros(length(files),runLen);
ANGSPEED_bbox           = zeros(length(files),runLen);
FORWARD_bbox            = false(length(files),runLen);
REVERSE_bbox            = false(length(files),runLen);
MSD_bbox                = zeros(length(files),runLen);
FULLYINLAWN_bbox        = false(length(files),runLen);
HEADPOKE_FWD_tip        = false(length(files),runLen);
HEADPOKE_REV_tip        = false(length(files),runLen);
HEADPOKE_PAUSE_tip      = false(length(files),runLen);
RADIAL_DIST_tip         = zeros(length(files),runLen);
EV_HO_DIST_tip          = zeros(length(files),runLen);
HEADPOKE_ANGLE_tip      = zeros(length(files),runLen);
LAWN_EXIT_v0            = false(length(files),runLen);
LAWN_ENTRY_v0           = false(length(files),runLen);
IN_OR_OUT_v0            = false(length(files),runLen);

%new features
Lawn_Entry              = false(length(files),runLen);
Lawn_Exit               = false(length(files),runLen);
In_Or_Out               = false(length(files),runLen);
HeadPokeIntervals       = false(length(files),runLen);
HeadPokesAll            = false(length(files),runLen);
HeadPokeFwd             = false(length(files),runLen);
HeadPokeRev             = false(length(files),runLen);
HeadPokePause           = false(length(files),runLen);
HeadPokeDist            = zeros(length(files),runLen);
HeadPokeAngle           = zeros(length(files),runLen);
HeadPokeSpeed           = zeros(length(files),runLen);
centMissing             = false(length(files),runLen);
splineMissing           = false(length(files),runLen);
Head_cent_x             = zeros(length(files),runLen);
Head_cent_y             = zeros(length(files),runLen);
Head_cent_smth_x        = zeros(length(files),runLen);
Head_cent_smth_y        = zeros(length(files),runLen);
Midbody_cent_x          = zeros(length(files),runLen);
Midbody_cent_y          = zeros(length(files),runLen);
Midbody_cent_smth_x     = zeros(length(files),runLen);
Midbody_cent_smth_y     = zeros(length(files),runLen);
Tail_cent_x             = zeros(length(files),runLen);
Tail_cent_y             = zeros(length(files),runLen);
Tail_cent_smth_x        = zeros(length(files),runLen);
Tail_cent_smth_y        = zeros(length(files),runLen);
MovingForward           = false(length(files),runLen);
MovingReverse           = false(length(files),runLen);
Centroid_speed          = zeros(length(files),runLen);
Head_speed              = zeros(length(files),runLen);
Head_angspeed           = zeros(length(files),runLen);
Midbody_speed           = zeros(length(files),runLen);
Midbody_angspeed        = zeros(length(files),runLen);
Tail_speed              = zeros(length(files),runLen);
Tail_angspeed           = zeros(length(files),runLen);
headAngVel_relMid       = zeros(length(files),runLen);
headRadVel_relMid       = zeros(length(files),runLen);
Center_Point_x          = zeros(length(files),20); %overkill but will fit any number of bg_struct videos
Center_Point_y          = zeros(length(files),20);
Lawn_Boundary_Pts_x     = zeros(length(files)*360,20); %there are 360 points in each lawn boundary
Lawn_Boundary_Pts_y     = zeros(length(files)*360,20);
Radial_Dist             = zeros(length(files),runLen);
Lawn_Boundary_Dist      = zeros(length(files),runLen);
HeadInLawn              = false(length(files),runLen);
MidbodyInLawn           = false(length(files),runLen);
TailInLawn              = false(length(files),runLen);
radTrajAngle            = zeros(length(files),runLen); %the angle of animal trajectory relative to radial vector
Head_grayscale          = zeros(length(files),runLen);
Head_grayscale_v0       = zeros(length(files),runLen);
Centroid_grayscale      = zeros(length(files),runLen);
Quirkiness              = zeros(length(files),runLen);
%new fields 08/24/21
Centroid_Radial_Dist    = zeros(length(files),runLen);
Centroid_Lawn_Boundary_Dist = zeros(length(files),runLen);
CentroidInLawn          = zeros(length(files),runLen);
Head_norm_grayscale     = zeros(length(files),runLen);
Centroid_norm_grayscale = zeros(length(files),runLen);

Grayscale_bounds        = zeros(length(files),3);
mean_grayscale_profile  = zeros(length(files),1000);
mean_norm_grayscale_profile = zeros(length(files),1000);
LBD_alignedTo_gs_profile = zeros(length(files),1000);

for i = 1:length(files)
    %     tic;
    disp(files{i});
    
%     try
        tmpSummaryStruct = load(files{i},'SUMMARY_STRUCT');
        SUMMARY_STRUCT = tmpSummaryStruct.SUMMARY_STRUCT;
        tmpbgstruct = load(files{i},'bg_struct');
        bg_struct = tmpbgstruct.bg_struct;
        OUT_STRUCT = NewFeaturesPostProcessing_082421( bg_struct, SUMMARY_STRUCT );
        
        PIXPERMM(i)                = single(OUT_STRUCT.PIXPERMM)';
        MATFILENAME(i)             = files(i); %the name of the FINAL.mat file where we got the data
        VIDEONAME(i)             = OUT_STRUCT.VIDEONAME(1);
        %find datestring within videoname
        vn1 = VIDEONAME{i};
        [startIndex,endIndex] = regexp(vn1,'_[0-9]{6}_');
        datestring = vn1(startIndex+1:endIndex-1);
        DATE(i)                    = {datestring};
        VIDEOFRAME(i,:)            = OUT_STRUCT.VIDEOFRAME';
        BGVIDINDEX(i,:)            = OUT_STRUCT.BGVIDINDEX';
        
        bins = 1:49:((length(files)+1)*49); %indexing system for Spline points (49 rows per animal)
        startTostop = [bins(1:end-1)' bins(2:end)'-1];
        SPLINE_x(startTostop(i,1):startTostop(i,2),:) = single(OUT_STRUCT.SPLINE_x)';
        SPLINE_y(startTostop(i,1):startTostop(i,2),:) = single(OUT_STRUCT.SPLINE_y)';
        
        bins = 1:48:((length(files)+1)*48); %indexing system for Posture_Angle (48 rows per animal)
        startTostop = [bins(1:end-1)' bins(2:end)'-1];
        POSTURE_ANGLE(startTostop(i,1):startTostop(i,2),:) = single(OUT_STRUCT.POSTURE_ANGLE)';
        
        CENTROID_bbox_x(i,:)       = single(OUT_STRUCT.CENTROID(:,1))';
        CENTROID_bbox_y(i,:)       = single(OUT_STRUCT.CENTROID(:,2))';
        HEAD_tip_x(i,:)            = single(OUT_STRUCT.HEAD(:,1))';
        HEAD_tip_y(i,:)            = single(OUT_STRUCT.HEAD(:,2))';
        TAIL_tip_x(i,:)            = single(OUT_STRUCT.TAIL(:,1))';
        TAIL_tip_y(i,:)            = single(OUT_STRUCT.TAIL(:,2))';
        SPEED_bbox(i,:)            = single(OUT_STRUCT.SPEED)';
        ANGSPEED_bbox(i,:)         = single(OUT_STRUCT.ANGSPEED)';
        FORWARD_bbox(i,:)          = logical(OUT_STRUCT.FORWARD)';
        REVERSE_bbox(i,:)          = logical(OUT_STRUCT.REVERSE)';
        OMEGA(i,:)                 = logical(OUT_STRUCT.OMEGA)';
        MSD_bbox(i,:)              = single(OUT_STRUCT.MSD)';
        FULLYINLAWN_bbox(i,:)      = logical(OUT_STRUCT.FULLYINLAWN)';
        HEADPOKE_FWD_tip(i,:)      = logical(OUT_STRUCT.HEADPOKE_FWD)';
        HEADPOKE_REV_tip(i,:)      = logical(OUT_STRUCT.HEADPOKE_REV)';
        HEADPOKE_PAUSE_tip(i,:)    = logical(OUT_STRUCT.HEADPOKE_PAUSE)';
        RADIAL_DIST_tip(i,:)       = single(OUT_STRUCT.RADIAL_DIST)';
        EV_HO_DIST_tip(i,:)        = single(OUT_STRUCT.EV_HO_DIST)';
        HEADPOKE_ANGLE_tip(i,:)    = single(OUT_STRUCT.HEADPOKE_ANGLE)';
        LAWN_ENTRY_v0(i,:)         = logical(OUT_STRUCT.LAWN_ENTRY)';
        LAWN_EXIT_v0(i,:)          = logical(OUT_STRUCT.LAWN_EXIT)';
        IN_OR_OUT_v0(i,:)          = logical(fillmissing(OUT_STRUCT.IN_OR_OUT,'previous'))';
        
        Lawn_Entry(i,:)            = logical(OUT_STRUCT.Lawn_Entry)';
        Lawn_Exit(i,:)             = logical(OUT_STRUCT.Lawn_Exit)';
        In_Or_Out(i,:)             = logical(OUT_STRUCT.In_Or_Out)';
        HeadPokeIntervals(i,:)     = logical(OUT_STRUCT.HeadPokeIntervals)';
        HeadPokesAll(i,:)          = logical(OUT_STRUCT.HeadPokesAll)';
        HeadPokeFwd(i,:)           = logical(OUT_STRUCT.HeadPokeFwd)';
        HeadPokeRev(i,:)           = logical(OUT_STRUCT.HeadPokeRev)';
        HeadPokePause(i,:)         = logical(OUT_STRUCT.HeadPokePause)';
        HeadPokeDist(i,:)          = single(OUT_STRUCT.HeadPokeDist)';
        HeadPokeAngle(i,:)         = single(OUT_STRUCT.HeadPokeAngle)';
        HeadPokeSpeed(i,:)         = single(OUT_STRUCT.HeadPokeSpeed)';
        centMissing(i,:)           = logical(OUT_STRUCT.centMissing)';
        splineMissing(i,:)         = logical(OUT_STRUCT.splineMissing)';
        Head_cent_x(i,:)           = single(OUT_STRUCT.Head_cent(:,1))';
        Head_cent_y(i,:)           = single(OUT_STRUCT.Head_cent(:,2))';
        Head_cent_smth_x(i,:)      = single(OUT_STRUCT.Head_cent_smth(:,1))';
        Head_cent_smth_y(i,:)      = single(OUT_STRUCT.Head_cent_smth(:,2))';
        Midbody_cent_x(i,:)        = single(OUT_STRUCT.Midbody_cent(:,1))';
        Midbody_cent_y(i,:)        = single(OUT_STRUCT.Midbody_cent(:,2))';
        Midbody_cent_smth_x(i,:)   = single(OUT_STRUCT.Midbody_cent_smth(:,1))';
        Midbody_cent_smth_y(i,:)   = single(OUT_STRUCT.Midbody_cent_smth(:,2))';
        Tail_cent_x(i,:)           = single(OUT_STRUCT.Tail_cent(:,1))';
        Tail_cent_y(i,:)           = single(OUT_STRUCT.Tail_cent(:,2))';
        Tail_cent_smth_x(i,:)      = single(OUT_STRUCT.Tail_cent_smth(:,1))';
        Tail_cent_smth_y(i,:)      = single(OUT_STRUCT.Tail_cent_smth(:,2))';
        MovingForward(i,:)         = logical(OUT_STRUCT.MovingForward)';
        MovingReverse(i,:)         = logical(OUT_STRUCT.MovingForward)';
        Centroid_speed(i,:)        = single(OUT_STRUCT.Centroid_speed)';
        Head_speed(i,:)            = single(OUT_STRUCT.Head_speed)';
        Head_angspeed(i,:)         = single(OUT_STRUCT.Head_angspeed)';
        Midbody_speed(i,:)         = single(OUT_STRUCT.Midbody_speed)';
        Midbody_angspeed(i,:)      = single(OUT_STRUCT.Midbody_angspeed)';
        
        
        Tail_speed(i,:)            = single(OUT_STRUCT.Tail_speed)';
        Tail_angspeed(i,:)         = single(OUT_STRUCT.Tail_angspeed)';
        headAngVel_relMid(i,:)     = single(OUT_STRUCT.headAngVel_relMid)';
        headRadVel_relMid(i,:)     = single(OUT_STRUCT.headRadVel_relMid)';
        
        centerPt = OUT_STRUCT.Center_Point;
        Center_Point_x(i,1:length(centerPt)) = single(centerPt(:,1))';
        Center_Point_y(i,1:length(centerPt)) = single(centerPt(:,2))';
        
        bins = 1:360:((length(files)+1)*360); %indexing system for Lawn Boundary Points (360 rows per animal x # of bg_struct entries per SUMMARY_STRUCT)
        startTostop = [bins(1:end-1)' bins(2:end)'-1];
        lawnBoundaryPtsX = OUT_STRUCT.Lawn_Boundary_Pts_x';
        lawnBoundaryPtsY = OUT_STRUCT.Lawn_Boundary_Pts_y';
        Lawn_Boundary_Pts_x(startTostop(i,1):startTostop(i,2),1:size(lawnBoundaryPtsX,2)) = single(lawnBoundaryPtsX);
        Lawn_Boundary_Pts_y(startTostop(i,1):startTostop(i,2),1:size(lawnBoundaryPtsY,2)) = single(lawnBoundaryPtsY);
        
        Radial_Dist(i,:)           = single(OUT_STRUCT.Radial_Dist)';
        Lawn_Boundary_Dist(i,:)    = single(OUT_STRUCT.Lawn_Boundary_Dist)';
        HeadInLawn(i,:)            = logical(OUT_STRUCT.HeadInLawn)';
        MidbodyInLawn(i,:)         = logical(OUT_STRUCT.MidbodyInLawn)';
        TailInLawn(i,:)            = logical(OUT_STRUCT.TailInLawn)';
        radTrajAngle(i,:)          = single(OUT_STRUCT.radTrajAngle)';
        Head_grayscale(i,:)        = single(OUT_STRUCT.Head_grayscale)';
        Head_grayscale_v0(i,:)     = single(OUT_STRUCT.Head_grayscale_v0)';
        Centroid_grayscale(i,:)    = single(OUT_STRUCT.Centroid_grayscale)';
        Quirkiness(i,:)            = single(OUT_STRUCT.Quirkiness)';
        
        %new fields 08/24/21
        Centroid_Radial_Dist(i,:)  = single(OUT_STRUCT.Centroid_Radial_Dist)';
        Centroid_Lawn_Boundary_Dist(i,:) = single(OUT_STRUCT.Centroid_Lawn_Boundary_Dist)';
        CentroidInLawn(i,:)          = logical(OUT_STRUCT.CentroidInLawn)';
        Head_norm_grayscale(i,:)     = single(OUT_STRUCT.Head_norm_grayscale)';
        Centroid_norm_grayscale(i,:) = single(OUT_STRUCT.Centroid_norm_grayscale)';
        
        Grayscale_bounds(i,:)        = single(OUT_STRUCT.Grayscale_bounds)';
        mean_grayscale_profile(i,:)  = single(OUT_STRUCT.mean_grayscale_profile)';
        mean_norm_grayscale_profile(i,:) = single(OUT_STRUCT.mean_norm_grayscale_profile)';
        LBD_alignedTo_gs_profile(i,:) = single(OUT_STRUCT.LBD_alignedTo_gs_profile)'; 
        
%     catch
%         warning('Problem using function. Skipping this file. Check out why.');
%     end
    %     toc;
end

%derive pausing
PAUSE_bbox = abs(SPEED_bbox)<=0.02;
Pause = abs(Midbody_speed)<=0.02;



%save all variables to a -v7.3 .mat file, which can be read in by h5py
%package in python.

%change to outpath
cd(outpath);
save([strain_name '_' timenow '_newFeatures.mat'],...,
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
fid = H5F.create([strain_name '_' timenow '_Filenames.h5'],'H5F_ACC_TRUNC',...
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

end
