function [msd, msd_smth] = get_msd( c_smth, pixpermm, window )
%GET_MSD.m This function takes in the smoothed centroid path and calculates
%the mean squared displacement of the animal's path in the lawn.

c_smth2 = c_smth(1+window:end,:);
cent_vector = [zeros(window,2); c_smth2-c_smth(1:end-window,:) ]./pixpermm;
msd = cent_vector(:,1).^2 + cent_vector(:,2).^2;
msd_smth = movmean(msd,15,'omitnan');
end

