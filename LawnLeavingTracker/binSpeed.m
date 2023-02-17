function binnedSpeed = binSpeed(allTracks, binSize) % binSize is in Frames here
	for (i=1:length(allTracks)) 
        speedData = allTracks(i).speed;
        numbBins = (length(speedData))/binSize;
        for (j=1:numbBins)
            startInd = (j*binSize) - (binSize -1);
            endInd = (j*binSize);
            currentData = speedData(startInd:endInd);
            binnedSpeed(i).Speed(j) = nanmean(currentData); %changed to mean from nanmean to be strict about onfood
        end
    end
end
