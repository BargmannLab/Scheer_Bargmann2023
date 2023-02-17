function statemap = getState2(SpeedBins,AngSpeedBins,ratio_cutoff,x_offset) % converts sequence of Speed/AngSpeed into a binary of 1's (dwell bins) and 2's (roam bins)

for(j=1:length(SpeedBins))
    %         if j == length(SpeedBins)
    %             disp('debug');
    %         end
    nBins = length(SpeedBins(j).Speed);
    statemap(j).state = [];
    statemap(j).actualratio = [];
    for (i=1:nBins)
        actualRatio = ((AngSpeedBins(j).AngSpeed(i) - x_offset)/(SpeedBins(j).Speed(i))); %ATTENTION HERE FOR X OFFSET 01-31-2018
        statemap(j).actualratio(i) = actualRatio;
        if((isnan(actualRatio)) == 1)
            statemap(j).state(i) = NaN;
%             if(i==1)
%                 statemap(j).state(i) = NaN;
%             else
%                 statemap(j).state(i) = statemap(j).state(i-1);
%             end
        else
            if (actualRatio >= ratio_cutoff)
                statemap(j).state(i) = 1;
            else
                statemap(j).state(i) = 2;
            end
        end
    end
end

end
