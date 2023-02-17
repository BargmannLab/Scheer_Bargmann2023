function SUMMARY_STRUCT = NewFeaturesPostProcessing_082421( bg_struct, SUMMARY_STRUCT )
%NewFeaturesPostProcessing_080421.m This function takes in an existing
%bg_struct and SUMMARY_STRUCT and derives several new features. It also
%inserts consensus NaNs to all fields to ensure missing data is the same
%across fields.

%   Procedures and new fields added (all have a mixture of Capital and Non-Capital letters):
%   0a) Define missing data based on when SPLINE data exists.
%   0b) Re-calculate the centroid path using the Midbody point along the
%   body spline instead of the centroid of the bounding box, then
%   re-derive CentSpeed, CentAngularSpeed using this new set of
%   coordinates.
%   1) calculate speed and angular speed for the Head, Midbody, and Tail
%   2) re-derive times when animal is moving Forward and Reverse
%   3) transform head position to Midbody-centric polar coordinates to
%   generate relative angular velocity and relative radial velocity
%   4) re-derive a host of lawn-related metrics:
%      Center_Point, Lawn_Boundary_Pts, Radial_Dist, Lawn_Boundary_Dist,
%      HeadInLawn, MidbodyInLawn, TailInLawn (booleans), Head_grayscale
%      (derived from the static original background with gaussian blur)
%   5) radTrajAngle (radial trajectory angle) : this is the
%   angle between an animal's centroid path and the radial vector pointing
%   to the center of the lawn. 0 = heading toward the center, 90 =
%   perpendicular to radial vector and parallel to the tangent line of the
%   closest point on the boundary, 180 = heading out of the center toward
%   the boundary.
%   6) re-calculate Lawn Leaving Events, Headpokes
%   7) quirkiness: this is the aspect ratio of the bounding box surrounding
%   the segmented animal.

pixpermm = SUMMARY_STRUCT.PIXPERMM;
centMissing = sum(isnan(SUMMARY_STRUCT.CENTROID),2)==2; % these are the missing data from the centroid path
splineMissing = sum(isnan(SUMMARY_STRUCT.SPLINE_x),2)==size(SUMMARY_STRUCT.SPLINE_x,2); % these are the missing data from splines. use this.

%following the Tierpsy methods, re-calculate the centroids for the head,
%body and tail segments of the 49 body points. Head_Tip is 1-3, Midbody is 17-33,
%Tail_Tip is 45-49

Centroid_smth = SUMMARY_STRUCT.CENTROID_SMTH;

Head_cent = [mean(SUMMARY_STRUCT.SPLINE_x(:,1:3),2) mean(SUMMARY_STRUCT.SPLINE_y(:,1:3),2)];
Midbody_cent = [mean(SUMMARY_STRUCT.SPLINE_x(:,17:33),2) mean(SUMMARY_STRUCT.SPLINE_y(:,17:33),2)];
Tail_cent = [mean(SUMMARY_STRUCT.SPLINE_x(:,45:49),2) mean(SUMMARY_STRUCT.SPLINE_y(:,45:49),2)];

%do a light moving average filter smoothing
Head_cent_smth = [movmean(Head_cent(:,1),3,'omitnan') movmean(Head_cent(:,2),3,'omitnan')];
Midbody_cent_smth = [movmean(Midbody_cent(:,1),3,'omitnan') movmean(Midbody_cent(:,2),3,'omitnan')];
Tail_cent_smth = [movmean(Tail_cent(:,1),3,'omitnan') movmean(Tail_cent(:,2),3,'omitnan')];

fps = 3;
StepSize = 3;

%Centroid
Xdif = CalcDif(Centroid_smth(:,1), StepSize) * fps;
Ydif = -CalcDif(Centroid_smth(:,2), StepSize) * fps;
Centroid_speed = (sqrt(Xdif.^2 + Ydif.^2)./pixpermm)'; %for head just keep it as absolute speed since there are a lot of head swings within forward and backward movement

%Head
Xdif = CalcDif(Head_cent_smth(:,1), StepSize) * fps;
Ydif = -CalcDif(Head_cent_smth(:,2), StepSize) * fps;
Head_speed = (sqrt(Xdif.^2 + Ydif.^2)./pixpermm)'; %for head just keep it as absolute speed since there are a lot of head swings within forward and backward movement
Head_angspeed = getAngularSpeed_NavinMethod(Head_cent_smth)';

%Midbody
Xdif = CalcDif(Midbody_cent_smth(:,1), StepSize) * fps;
Ydif = -CalcDif(Midbody_cent_smth(:,2), StepSize) * fps;
Midbody_speed = (sqrt(Xdif.^2 + Ydif.^2)./pixpermm)';

%derive speed, extract forward, reverse
coherence_thresh = 90; %in degrees -- this requires that centroid vector and head or tail vector must be within this angle range of each other to be considered coherent motion
speed_thresh = 0.02; %mm/sec -- this is the required speed to be considered moving
[~, ~, ~, ~, ~, moving_forward, moving_reverse] = getforwardreverse2(Midbody_cent_smth, Head_cent_smth, Tail_cent_smth, Midbody_speed, coherence_thresh, speed_thresh);

Midbody_speed(moving_reverse) = -1*Midbody_speed(moving_reverse); %here make speed negative if the animal is reversing.
Midbody_angspeed = getAngularSpeed_NavinMethod(Midbody_cent_smth)';

%Tail
Xdif = CalcDif(Tail_cent_smth(:,1), StepSize) * fps;
Ydif = -CalcDif(Tail_cent_smth(:,2), StepSize) * fps;
Tail_speed = (sqrt(Xdif.^2 + Ydif.^2)./pixpermm)'; %for head just keep it as absolute speed since there are a lot of head swings within forward and backward movement
Tail_angspeed = getAngularSpeed_NavinMethod(Tail_cent_smth)';

%%%%%%%%%%%%%%%%%%%%%%%
% Now, convert the head position into polar coordinates about the midbody
% position to facilitate computation of the radial and angular velocity of
% the head movements.

Head_relCent = abs(Head_cent_smth-Midbody_cent_smth);
[Head_theta,Head_rho] = cart2pol(Head_relCent(:,1),Head_relCent(:,2));
Head_theta = rad2deg(Head_theta);
headAngularVelocity_relMid = CalcDif(Head_theta, StepSize)' * fps;
headRadialVelocity_relMid = CalcDif(Head_rho, StepSize)' * fps;

%%%%%%%%%%%%%%%%%%%%%%%
%re-calculate lawn boundary distance and radial distance from center.
Center_Point = NaN(length(bg_struct),2); %note that these fields (Center Point and Lawn Boundary Pts) will always start with NaNs -- this is because SUMMARY_STRUCT begins 20 minutes in to the videos.
Lawn_Boundary_Pts_x = NaN(length(bg_struct),360);
Lawn_Boundary_Pts_y = NaN(length(bg_struct),360);

Radial_Dist = NaN(size(SUMMARY_STRUCT.RADIAL_DIST));
Lawn_Boundary_Dist = NaN(size(SUMMARY_STRUCT.EV_HO_DIST));
Centroid_Radial_Dist = NaN(size(SUMMARY_STRUCT.EV_HO_DIST));
Centroid_Lawn_Boundary_Dist = NaN(size(SUMMARY_STRUCT.EV_HO_DIST));
HeadInLawn = NaN(length(SUMMARY_STRUCT.RADIAL_DIST),1);
MidbodyInLawn = NaN(length(SUMMARY_STRUCT.RADIAL_DIST),1);
TailInLawn = NaN(length(SUMMARY_STRUCT.RADIAL_DIST),1);
CentroidInLawn = NaN(length(SUMMARY_STRUCT.RADIAL_DIST),1);
Midbody_radialVec = NaN(length(SUMMARY_STRUCT.RADIAL_DIST),2);
Head_norm_grayscale = NaN(size(SUMMARY_STRUCT.HEAD_GS));
Head_grayscale = NaN(size(SUMMARY_STRUCT.HEAD_GS));
Head_grayscale_v0 = SUMMARY_STRUCT.HEAD_GS;
Centroid_norm_grayscale = NaN(size(SUMMARY_STRUCT.HEAD_GS));
Centroid_grayscale = NaN(size(SUMMARY_STRUCT.HEAD_GS));

%loop over timepoints to extract all of the above quantities
BGVIDINDEX = fillmissing(SUMMARY_STRUCT.BGVIDINDEX,'nearest'); %sometimes we have missing values for the bg_struct, just restore to nearest non-missing.
lastBGVI = 0;

%Define the grayscale profile from the first video -- before it gets mussed
%up by animal movements.
[min_gs,max_gs,mean_interiorLawn_gs,mean_gs_profile_aligned,mean_norm_gs_profile_aligned,~,LBD_alignedTo_gs_profile,~] = ...
    extract_grayscale_minmax(bg_struct,1,pixpermm);

Grayscale_bounds = [min_gs mean_interiorLawn_gs max_gs]; %can be used to scale head_grayscale in postprocessing

for j = 1:length(BGVIDINDEX) 
    curr_BGVI = BGVIDINDEX(j);
%     disp(curr_BGVI);
    if curr_BGVI~=lastBGVI %update it if it changes throughout the loop.
%         disp('change bg_struct index.');
        lawnboundary = bg_struct(curr_BGVI).ev_ho_crp_rel;
        %interpolate the function to ensure its dimension is 1x360
        %(sometimes I have encountered 1x359 which causes problems)
        rs_x = interp1(1:length(lawnboundary),lawnboundary(:,1),linspace(1,length(lawnboundary),360));
        rs_y = interp1(1:length(lawnboundary),lawnboundary(:,2),linspace(1,length(lawnboundary),360));
        lawnboundary = [rs_x' rs_y'];
        
        Lawn_Boundary_Pts_x(curr_BGVI,:) = lawnboundary(:,1)';
        Lawn_Boundary_Pts_y(curr_BGVI,:) = lawnboundary(:,2)';
        [lawn_center_x, lawn_center_y, ~] = centroid(lawnboundary(:,1),lawnboundary(:,2)); %this calculates the center point of the lawn
        Center_Point(curr_BGVI,:)=[lawn_center_x lawn_center_y]; %catalog center point for future use
        curr_x_offset = bg_struct(curr_BGVI).region_rounded(1); %these should have been added to the lawn boundary and center points!
        curr_y_offset = bg_struct(curr_BGVI).region_rounded(2);
        
        % Subtract background illumination and blur out pixels above threshold
        orig_bg = bg_struct(curr_BGVI).orig_background;
        clean_background = bg_struct(curr_BGVI).clean_background;
        outer_boundary_mask = bg_struct(curr_BGVI).outer_boundary_mask;
        level = bg_struct(curr_BGVI).level;
        
        bg_bgsub = imcomplement(orig_bg.*outer_boundary_mask-clean_background.*outer_boundary_mask);
        bg_bgthresh = imcomplement(im2bw(bg_bgsub,level));
        PixelsToBlur = imdilate(bg_bgthresh,strel('disk',5));
        curr_bg = regionfill(orig_bg,PixelsToBlur);%8/24/21 now this is the negative image so grayscale values no longer need to be inverted in python post-processing!

    end
    
    HeadInLawn(j) = logical(inpolygon(Head_cent_smth(j,1),Head_cent_smth(j,2),lawnboundary(:,1),lawnboundary(:,2)));
    MidbodyInLawn(j) = logical(inpolygon(Midbody_cent_smth(j,1),Midbody_cent_smth(j,2),lawnboundary(:,1),lawnboundary(:,2)));
    TailInLawn(j) = logical(inpolygon(Tail_cent_smth(j,1),Tail_cent_smth(j,2),lawnboundary(:,1),lawnboundary(:,2)));
    CentroidInLawn(j) = logical(inpolygon(Centroid_smth(j,1),Centroid_smth(j,2),lawnboundary(:,1),lawnboundary(:,2)));
    
    if ~isnan(Head_cent_smth(j,1))
        Radial_Dist(j) = pdist2(Head_cent_smth(j,:),[lawn_center_x lawn_center_y])/pixpermm; %find head distance from the center of the lawn
        [~,dist,~] = distance2curve(lawnboundary,Head_cent_smth(j,:)); %find head distance to the nearest point on event horizon
        if ~HeadInLawn(j)
            dist = -1*dist;
        end
        Lawn_Boundary_Dist(j) = dist/pixpermm;
        Midbody_radialVec(j,:) = [lawn_center_x-Midbody_cent_smth(j,1) lawn_center_y-Midbody_cent_smth(j,2)];
        
        head_x = round(Head_cent_smth(j,1))+curr_x_offset; head_y = round(Head_cent_smth(j,2))+curr_y_offset;
        Head_grayscale(j) = curr_bg(head_y,head_x); %high numbers are darker pixels
        %look up based on lawn boundary distance
        [~,minIdx] = min(abs(Lawn_Boundary_Dist(j)-LBD_alignedTo_gs_profile)); %find the closest index in the LBD profile to the current lawn boundary distance to extract the corresponding bacterial density
        Head_norm_grayscale(j) = mean_norm_gs_profile_aligned(minIdx);
    end
    
    if ~isnan(Centroid_smth(j,1))
        Centroid_Radial_Dist(j) = pdist2(Centroid_smth(j,:),[lawn_center_x lawn_center_y])/pixpermm; %find head distance from the center of the lawn
        [~,dist,~] = distance2curve(lawnboundary,Centroid_smth(j,:)); %find head distance to the nearest point on event horizon
        if ~CentroidInLawn(j)
            dist = -1*dist;
        end
        Centroid_Lawn_Boundary_Dist(j) = dist/pixpermm;
        
        cent_x = round(Centroid_smth(j,1))+curr_x_offset; cent_y = round(Centroid_smth(j,2))+curr_y_offset;
        Centroid_grayscale(j) = curr_bg(cent_y,cent_x); %high numbers are darker pixels
        %look up based on lawn boundary distance
        [~,minIdx] = min(abs(Centroid_Lawn_Boundary_Dist(j)-LBD_alignedTo_gs_profile)); %find the closest index in the LBD profile to the current lawn boundary distance to extract the corresponding bacterial density
        Centroid_norm_grayscale(j) = mean_norm_gs_profile_aligned(minIdx);
    end
    
    lastBGVI = curr_BGVI;
end


%compute midpoint trajectory angle to the center point vector
Midbody_timeVec = [(CalcDif(Midbody_cent_smth(:,1), StepSize)*fps)' (CalcDif(Midbody_cent_smth(:,2), StepSize)*fps)'];
Midbody_timeVec_norm = cell2mat(cellfun(@(x) x/norm(x), num2cell(Midbody_timeVec,2),'UniformOutput',false));
Midbody_radialVec_norm = cell2mat(cellfun(@(x) x/norm(x), num2cell(Midbody_radialVec,2),'UniformOutput',false));
%use the trick to calculate angles between vectors: angle = atan2(norm(cross(a,b)), dot(a,b))
a = [Midbody_timeVec_norm zeros(size(Midbody_timeVec_norm,1),1)]; %pad with zeros to use cross and dot
b = [Midbody_radialVec_norm zeros(size(Midbody_radialVec_norm,1),1)];
radTrajAngle = cell2mat(cellfun(@(x,y)(atan2d(norm(cross(x,y)),dot(x,y))),num2cell(a,2),num2cell(b,2),'UniformOutput',false));

%%%%%%%%%%%%%%%%%%%%%
mergeStruct = @(x,y) cell2struct([struct2cell(x);struct2cell(y)],[fieldnames(x);fieldnames(y)]); %merge new fields into SUMMARY_STRUCT
% Re-calculate lawn leaving events:
EXIT_STRUCT = get_enter_exit_events_from_summary( BGVIDINDEX, SUMMARY_STRUCT.SPLINE_x, SUMMARY_STRUCT.SPLINE_y, Lawn_Boundary_Pts_x, Lawn_Boundary_Pts_y );
SUMMARY_STRUCT = mergeStruct(SUMMARY_STRUCT,EXIT_STRUCT);
% Re-calculate headpoke events:
POKE_STRUCT = get_head_pokes_from_summary( BGVIDINDEX, pixpermm, SUMMARY_STRUCT.In_Or_Out, Lawn_Boundary_Pts_x, Lawn_Boundary_Pts_y, Lawn_Boundary_Dist, Head_cent_smth, Midbody_cent_smth, moving_forward, moving_reverse, Midbody_speed );
SUMMARY_STRUCT = mergeStruct(SUMMARY_STRUCT,POKE_STRUCT);

%%%%%%%%%%%%%%%%%%%%%%%
% Now calculate the quirkiness of the bounding box (an indirect measure of
% posture / curvature)
% Q = sqrt(1 - (a^2)/(A^2)), where a is the smaller and A is the larger
% side of the rectangular bounding box of the animal, respectively.
bbox = SUMMARY_STRUCT.BBOX; %x,y,w,h
a = min(bbox(:,3:4),[],2); % the shorter side length
A = max(bbox(:,3:4),[],2); % the longer side length
Quirkiness = sqrt(1-((a.^2)./(A.^2)));

% add fields to SUMMARY STRUCT (all will be lowercase - Not all caps)
SUMMARY_STRUCT.BGVIDINDEX = BGVIDINDEX; %update to the filled in version.
SUMMARY_STRUCT.centMissing = centMissing;
SUMMARY_STRUCT.splineMissing = splineMissing;
SUMMARY_STRUCT.Head_cent = Head_cent;
SUMMARY_STRUCT.Head_cent_smth = Head_cent_smth;
SUMMARY_STRUCT.Midbody_cent = Midbody_cent;
SUMMARY_STRUCT.Midbody_cent_smth = Midbody_cent_smth;
SUMMARY_STRUCT.Tail_cent = Tail_cent;
SUMMARY_STRUCT.Tail_cent_smth = Tail_cent_smth;
%add forward and reverse as calculated using the new head, midbody, tail
%segments:
SUMMARY_STRUCT.MovingForward = moving_forward;
SUMMARY_STRUCT.MovingReverse = moving_reverse;
SUMMARY_STRUCT.Centroid_speed = Centroid_speed;
SUMMARY_STRUCT.Head_speed = Head_speed;
SUMMARY_STRUCT.Head_angspeed = Head_angspeed;
SUMMARY_STRUCT.Midbody_speed = Midbody_speed;
SUMMARY_STRUCT.Midbody_angspeed = Midbody_angspeed;
SUMMARY_STRUCT.Tail_speed = Tail_speed;
SUMMARY_STRUCT.Tail_angspeed = Tail_angspeed;
%include new fields relating to head movement relative to midbody point
SUMMARY_STRUCT.headAngVel_relMid = headAngularVelocity_relMid;
SUMMARY_STRUCT.headRadVel_relMid = headRadialVelocity_relMid;

%include new features relating to lawn boundary
SUMMARY_STRUCT.Center_Point = Center_Point;
SUMMARY_STRUCT.Lawn_Boundary_Pts_x = Lawn_Boundary_Pts_x;
SUMMARY_STRUCT.Lawn_Boundary_Pts_y = Lawn_Boundary_Pts_y;
SUMMARY_STRUCT.Radial_Dist = Radial_Dist;
SUMMARY_STRUCT.Lawn_Boundary_Dist = Lawn_Boundary_Dist;
SUMMARY_STRUCT.HeadInLawn = HeadInLawn;
SUMMARY_STRUCT.MidbodyInLawn = MidbodyInLawn;
SUMMARY_STRUCT.TailInLawn = TailInLawn;
SUMMARY_STRUCT.radTrajAngle = radTrajAngle;
SUMMARY_STRUCT.Head_grayscale = Head_grayscale;
SUMMARY_STRUCT.Head_grayscale_v0 = Head_grayscale_v0; %old way
SUMMARY_STRUCT.Centroid_grayscale = Centroid_grayscale;

%include Quirkiness of the bounding box
SUMMARY_STRUCT.Quirkiness = Quirkiness;

%new fields as of 8/24/21
SUMMARY_STRUCT.Centroid_Radial_Dist = Centroid_Radial_Dist;
SUMMARY_STRUCT.Centroid_Lawn_Boundary_Dist = Centroid_Lawn_Boundary_Dist;
SUMMARY_STRUCT.CentroidInLawn = CentroidInLawn;
SUMMARY_STRUCT.Head_norm_grayscale = Head_norm_grayscale;
SUMMARY_STRUCT.Centroid_norm_grayscale = Centroid_norm_grayscale;

SUMMARY_STRUCT.Grayscale_bounds = Grayscale_bounds;
SUMMARY_STRUCT.mean_grayscale_profile = mean_gs_profile_aligned;
SUMMARY_STRUCT.mean_norm_grayscale_profile = mean_norm_gs_profile_aligned;
SUMMARY_STRUCT.LBD_alignedTo_gs_profile = LBD_alignedTo_gs_profile;

end