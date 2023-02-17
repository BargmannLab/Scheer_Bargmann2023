function statemap = getStateAuto2(trackData,binSize,ratio_cutoff,x_offset) % binSize in Frames here
binnedSpeed = binSpeed(trackData,binSize);
binnedAngSpeed = binAngSpeed(trackData,binSize);
statemap = getState2(binnedSpeed,binnedAngSpeed,ratio_cutoff,x_offset); 
end
