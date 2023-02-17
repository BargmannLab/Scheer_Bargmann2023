function binnedAngSpeed = binAngSpeed(allTracks, binSize) %binSize is in frames
	for (i=1:length(allTracks)) 
        angspeedData = allTracks(i).angspeed;
%         angspeedData(~onfood) = NaN; %disqualify off food frames
        numbBins = (length(angspeedData))/binSize;
        for (j=1:numbBins)
            startInd = (j*binSize) - (binSize -1);
            endInd = (j*binSize);
            currentData = angspeedData(startInd:endInd);
            binnedAngSpeed(i).AngSpeed(j) = nanmean(abs(currentData));
        end
    end
    
end
