function track_QC_010219(camera_params_file)
%This function opens each .mat file in the folder and decides whether to
%include it in subsequent analyses based on the following criteria:
% 1. there was a worm in the video
% 2. the worm was visible for at least half of the video
% 3. the worm gets into the lawn within the first 20 minutes of the video
% 4. there aren't too many tracks after removing garbage tracks
% 5. the plate wasn't bumped more than 10 pixels in either direction during
% the video

%what's new in this version:
% for at least a cumulative minute's worth of time
% if these criteria are met, we copy the .mat to a subfolder called "passQC"

timenow = datestr(now,'mm_dd_yy');
lawn_string = ['010219_QCedit_' timenow];

if nargin<1
   camera_params_file = 'camera_params_NORMAL_LED_setup_04_22_2019.mat'; 
end
% '/Users/elias/Dropbox/ROCKEFELLER/BARGMANN_LAB_LYFE/matlab_codes/LAWN-LEAVING-REPO/tracking_params'
% camera_params_path = which(['LAWN-LEAVING-REPO/tracking_params/' camera_params_file]);
camera_params_path = which(['tracking_params\' camera_params_file]);
load(camera_params_path, 'camera_params'); %load in data

[~, pathname, ~] = uigetfile({'*'});
cd(pathname);
files = dir('*FINAL.mat');

foldername = '';
mkdir([foldername '_passQC']);
subdir = [pwd '\' foldername '_passQC'];

for i = 1:length(files)
    
    keep = true; %boolean whether to keep this file
    resave = false;
    
    disp(files(i).name);

    %just load TRACKS_slim
    TRACKS_slim = load(files(i).name,'TRACKS_slim');
    TRACKS_slim = TRACKS_slim.TRACKS_slim; %always a struct within a struct


    
    %check if there are any tracks that need to be removed -- such as a
    %piece of dust, etc. this check should also be performed by
    %tracks_postprocessing2.m but I will put it here for now.
    bbox = {TRACKS_slim.bbox}';
    area = cell2mat(cellfun(@(x) mean(x(:,3).*x(:,4),1),bbox,'UniformOutput',false)); %get the average area of the bounding box for the worm for each track
    
    area_cutoff = 750;
    numtracks_cutoff = 150;
    if sum(area>area_cutoff)~=length(area)
        TRACKS_slim = TRACKS_slim(area>area_cutoff);
        
        if length(TRACKS_slim)>numtracks_cutoff %if you still have a ton of tracks after removing obvious crap, the whole thing is probably crap
            keep = false;
            disp('TOO MANY TRACKS! DISCARD!');
        else
            bg_struct = load(files(i).name,'bg_struct');
            bg_struct = bg_struct.bg_struct;
            videoname = bg_struct(1).videoname;
            %%%%%%%%% recalculate lawn entry and exit, remake plots
            cameracode = videoname(1:2);
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
                case 's2'
                    level = camera_params.s2.level;
                    pixpermm = camera_params.s2.pixpermm;
                case 's3'
                    level = camera_params.s3.level;
                    pixpermm = camera_params.s3.pixpermm;
                case 's4'
                    level = camera_params.s4.level;
                    pixpermm = camera_params.s4.pixpermm;
                case 's5'
                    level = camera_params.s5.level;
                    pixpermm = camera_params.s5.pixpermm;
                case 's6'
                    level = camera_params.s6.level;
                    pixpermm = camera_params.s6.pixpermm;
                case 'c1'
                    level = camera_params.c1.level;
                    pixpermm = camera_params.c1.pixpermm;
                case 'c2'
                    level = camera_params.c2.level;
                    pixpermm = camera_params.c2.pixpermm;
                case 'c3'
                    level = camera_params.c3.level;
                    pixpermm = camera_params.c3.pixpermm;
                case 'c4'
                    level = camera_params.c4.level;
                    pixpermm = camera_params.c4.pixpermm;
                case 'c5'
                    level = camera_params.c5.level;
                    pixpermm = camera_params.c5.pixpermm;
                case 'c6'
                    level = camera_params.c6.level;
                    pixpermm = camera_params.c6.pixpermm;
                otherwise
                    error('Camera data cannot be found!');
            end
            
            [TRACKS_slim, EXIT_STRUCT, POKE_STRUCT, SUMMARY_STRUCT] = tracks_postprocessing_resave( TRACKS_slim, [], bg_struct, pixpermm, [videoname(1:end-4) '_' lawn_string]);
            
            %%%%%%%%%
            resave = true;
        end
    end
    
    if isempty(TRACKS_slim) %if there are no tracks, don't keep it
        keep = false;
    else %otherwise check other ways
        totalVisible = 0;
        totalInLawn = 0;
        totalInLawnbefore20 = 0;
        totalFrames = 10800; %disregard frames after 1 hour for the two hour videos I used to take
        for j = 1:length(TRACKS_slim)
            fa = TRACKS_slim(j).framesActive;
            fa_1 = fa(1);
            fa_end = fa(end);
            if fa_1 > totalFrames
                continue;
            end
            
            totalVisible = totalVisible + TRACKS_slim(j).totalVisibleCount;
            
            startidx = 1;
            stopidx = length(TRACKS_slim(j).centroidinlawn);
            if fa_1 <= totalFrames && fa_end > totalFrames
                stopidx = find(fa==10800);
                totalVisible = totalVisible - (fa_end-totalFrames); %remove overflow
            end
            totalInLawn = totalInLawn + nansum(TRACKS_slim(j).centroidinlawn(startidx:stopidx));
            
            if fa_1<3600 %20 minutes
                if fa_end > 3600
                    stopidx20 = find(fa==3600);
                else
                    stopidx20 = length(TRACKS_slim(j).centroidinlawn);
                end
                totalInLawnbefore20 = totalInLawnbefore20 + nansum(TRACKS_slim(j).centroidinlawn(startidx:stopidx20));
            end
        end
        %MUST BE...
        if totalVisible/totalFrames<0.50 % 50% visible within the hour
            disp('was not visible at least 50%.');
            keep = false;
        end
        
        if totalInLawnbefore20<180      % in the lawn for at least a minute within first 20 minutes
            disp('was not in lawn for at least 1 minute before 20 minutes.');
            keep = false;
        end
        
        %throw out any data that came from a video that was bumped more
        %than 10 pixels in either direction -- this data can be unreliable.
        mvmnt_cutoff = 10;
        xshift = {TRACKS_slim.x_shift}';
        xshiftmorethanx = cellfun(@(x) sum(x>mvmnt_cutoff)>0,xshift,'UniformOutput',true);
        yshift = {TRACKS_slim.y_shift}';
        yshiftmorethanx = cellfun(@(x) sum(x>mvmnt_cutoff)>0,yshift,'UniformOutput',true);
        if sum(xshiftmorethanx)>0 || sum(yshiftmorethanx)>0
            disp('plate was bumped more than 10 pixels in either direction!');
            keep = false;
        end
        
    end
    
    if keep
        if ~resave
%             command = ['copy ' files(i).name ' ' subdir];
            command = ['copy  "' files(i).name '" "' subdir '"'];
            status = dos(command);
            if status ~= 0
                disp('not copied successfully!');
            end
            
        else
            bg_struct = load(files(i).name,'bg_struct');
            bg_struct = bg_struct.bg_struct;
            cd(subdir);
            save([files(i).name(1:end-25) '_QC_EDIT_' timenow '_FINAL.mat'],'TRACKS_slim','EXIT_STRUCT','POKE_STRUCT','SUMMARY_STRUCT','bg_struct'); %just save the same tracks that you loaded -- which includes head_pokes and leaving_events
            cd(pathname);
        end
        
    else
        disp('discard!');
    end
    
end

end

