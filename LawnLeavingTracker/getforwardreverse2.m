function [head_vector, cent_vector, tail_vector, speed, speed_smooth, moving_forward, moving_reverse] = getforwardreverse2( cent,head,tail,speed,coherence_thresh,speed_thresh )
%GETFORWARDREVERSE.M This function takes in the centroid, head and tail
%positions and extracts the bouts of forward and reverse movement from
%them. Also uses speed to determine whether the animal is moving enough to
%discern these states.

smth_window = 3;
c_smth = [movmean(cent(:,1),smth_window,'omitnan') movmean(cent(:,2),smth_window,'omitnan')];
h_smth = [movmean(head(:,1),smth_window,'omitnan') movmean(head(:,2),smth_window,'omitnan')];
t_smth = [movmean(tail(:,1),smth_window,'omitnan') movmean(tail(:,2),smth_window,'omitnan')];

speed_smooth = movmean(speed,smth_window,'omitnan');

int = 3;
c_smth2 = c_smth(1+int:end,:); %interval = 3 frames = 1 second
h_smth2 = h_smth(1+int:end,:);
t_smth2 = t_smth(1+int:end,:);

cent_vector =  c_smth2 - c_smth(1:end-int,:) ; %don't pad with zeros yet
head_vector =  h_smth2 - h_smth(1:end-int,:) ;
tail_vector =  t_smth2 - t_smth(1:end-int,:) ;

c_h_vect_angles = zeros(size(cent_vector,1),1);
c_t_vect_angles = zeros(size(cent_vector,1),1);

cent_vect_head_angles = zeros(size(cent_vector,1),1);
cent_vect_tail_angles = zeros(size(cent_vector,1),1);

for j = 1:size(cent_vector,1)
    c = [cent_vector(j,:) 0]; %time vectors
    h = [head_vector(j,:) 0];
    t = [tail_vector(j,:) 0];
    
    c_h_vect_angles(j) = atan2d(norm(cross(c,h)),dot(c,h));%angles between time vectors
    c_t_vect_angles(j) = atan2d(norm(cross(c,t)),dot(c,t));
    
    he = h_smth(j,:); %actual coordinates
    ce = c_smth(j,:);
    ta = t_smth(j,:);
    center_head = [he-ce 0]; %vectors from centroid to head or centroid to tail (single frame)
    center_tail = [ta-ce 0];
    cent_vect_head_angles(j) = atan2d(norm(cross(c,center_head)),dot(c,center_head));
    cent_vect_tail_angles(j) = atan2d(norm(cross(c,center_tail)),dot(c,center_tail));
end
%now pad everything with 3 zeros at the beginning to match speed and
%angular speed from other calculations.
cent_vector = [zeros(int,2); cent_vector];
head_vector = [zeros(int,2); head_vector];
tail_vector = [zeros(int,2); tail_vector];
c_h_vect_angles = [zeros(int,1) ; c_h_vect_angles];
c_t_vect_angles = [zeros(int,1) ; c_t_vect_angles];

cent_vect_head_angles = [zeros(int,1) ; cent_vect_head_angles];
cent_vect_tail_angles = [zeros(int,1) ; cent_vect_tail_angles];

moving = speed_smooth>speed_thresh; %this is the speed threshold for movement
c_h_agree = c_h_vect_angles<coherence_thresh; %centroid time vector is consistent with head time vector
c_t_agree = c_t_vect_angles<coherence_thresh;
cent_head_agree = cent_vect_head_angles < cent_vect_tail_angles; %centroid time vector is pointing towards head
cent_tail_agree = cent_vect_tail_angles < cent_vect_head_angles; % " towards tail
moving_forward = moving & c_h_agree & cent_head_agree;
moving_reverse = moving & c_t_agree & cent_tail_agree;

%clean up these intervals so there aren't frames when it stops going
%forward for a single frame or vice versa.
moving_forward = logical(hampel(double(moving_forward),3)); %remove outliers where the animal stops moving spontaneously for 1 frame
moving_reverse = logical(hampel(double(moving_reverse),3)); 

%change the sign of speed so that forward movement is + and reverse
%movement is -
speed(moving_reverse) = -1*speed(moving_reverse);
speed_smooth = movmean(speed,smth_window,'omitnan'); %re-do smoothing procedure afte sign change

end

