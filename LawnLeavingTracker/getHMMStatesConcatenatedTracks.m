function [expNewSeq_out, expStates_out, estTR, estE, actualratio_out] = getHMMStatesConcatenatedTracks(speed,angspeed,binSize,onfood,ratio_cutoff,trans,emis,x_offset,train)
%function [estTR estE] = getHMMStates(finalTracks,binSize, trans,emis)
%does R/D classification on the entire track regardless of whether it was
%in or out of the lawn.

finalTracks = struct();
finalTracks.speed = abs(speed);%make sure this is absolute speed for R/D
finalTracks.angspeed = angspeed;
finalTracks.frames = 1:size(speed,1);

% call this for every contiguous bit of inlawn track
statemap = getStateAuto2(finalTracks,binSize,ratio_cutoff,x_offset);

statesHMM = [];
for (i=1:length(statemap)) % clean up first couple entries
    if ~isempty(statemap(i).state)
        if length(statemap(i).state) == 1 %don't include these
        else
            tmp = statemap(i).state(2:(length(statemap(i).state))); %why do 2->end?
%             tmp = statemap(i).state; %just use the whole thing 02-05-2018
            %if you are missing data, fill it in
            tmp = fillmissing(tmp,'nearest');
            newseq(i).states = tmp;
        end

    end
end
if ~exist('newseq','var') %if we never encountered a track bit longer than a single bin, return NaNs as before
    expNewSeq_out = NaN(size(onfood));
    expStates_out = NaN(size(onfood));
    estTR = NaN;
    estE = NaN;
    actualratio_out = NaN(size(onfood));
    return;
end

seqs = struct2cell(newseq);


%%%%TRAIN THE MODEL
if train
    [estTR,estE] = hmmtrain(seqs,trans,emis); %might not need to do this every time, just use what you find from a big N2 population
else
    estTR = trans;
    estE = emis;
end
%%%%Apply Viterbi algorithm
for i = 1:length(newseq)
        statesHMM(i).states = hmmviterbi(newseq(i).states,estTR,estE);
        statesHMM(i).actualratio = statemap(i).actualratio(2:end);
end


%%%Create expStates structure - most probable state path by HMM
%%%Re-normalize to track length, given binSize
for i = 1:length(newseq)
    if ~isempty(statesHMM(i).states)
        speedData = finalTracks(i).speed;
        numbBins = (length(speedData))/binSize;
        numbBins = floor(numbBins);
        expStates(i).states(1:(binSize)) = statesHMM(i).states(1);
        expStates(i).actualratio(1:(binSize)) = statesHMM(i).actualratio(1);
        for (j = 2:numbBins)
            startPl = (j * binSize) - (binSize-1);
            stopPl  = (j * binSize);
            expStates(i).states(startPl:stopPl) = statesHMM(i).states(j-1);
            expStates(i).actualratio(startPl:stopPl) = statesHMM(i).actualratio(j-1);
        end
        numFrames = length(finalTracks(i).frames);
        leftOver_Frames = mod(numFrames,binSize);
        if (leftOver_Frames>0)
            currentExpStateLength = length(expStates(i).states);
            newStart = currentExpStateLength+1;
            %             newStop = newStart + (leftOver_Frames -4);
            newStop = newStart + leftOver_Frames-1;
            expStates(i).states(newStart:newStop) = statesHMM(i).states(end);
            expStates(i).actualratio(newStart:newStop) = statesHMM(i).actualratio(end);
            %         else
            %             expStates(i).states = expStates(i).states(1:(numFrames-3));
        end
    end
end

%%%Create expNewSeq structure - just the binary data from 2D scatter
% for(i= 1:length(finalTracks))
for i = 1:length(newseq)
    if ~isempty(statesHMM(i).states)
        speedData = finalTracks(i).speed;
        numbBins = (length(speedData))/binSize;
        numbBins = floor(numbBins);
        expNewSeq(i).states(1:(binSize)) = newseq(i).states(1);
        %         expNewSeq(i).actualratio(1:(binSize)) = statesHMM(i).actualratio(1);
        for (j = 2:numbBins)
            startPl = (j * binSize) - (binSize-1);
            stopPl  = (j * binSize);
            expNewSeq(i).states(startPl:stopPl) = newseq(i).states(j-1);
            %             expNewSeq(i).actualratio(startPl:stopPl) = statesHMM(i).actualratio(j-1);
        end
        numFrames = length(finalTracks(i).frames);
        leftOver_Frames = mod(numFrames,binSize);
        if (leftOver_Frames>0)
            currentExpStateLength = length(expNewSeq(i).states);
            newStart = currentExpStateLength+1;
            %             newStop = newStart + (leftOver_Frames -4);
            newStop = newStart + leftOver_Frames-1;
            expNewSeq(i).states(newStart:newStop) = newseq(i).states(end);
            %             expNewSeq(i).actualratio(newStart:newStop) = statesHMM(i).actualratio(end);
            %         else
            %             expStates(i).states = expStates(i).states(1:(numFrames-3));
        end
    end
end

expStates_out = NaN(size(onfood));
expNewSeq_out = NaN(size(onfood));
actualratio_out = NaN(size(onfood));

for i = 1:length(newseq)
    if ~isempty(expStates(i).states)
        expStates_out(finalTracks(i).frames) = expStates(i).states';
        expNewSeq_out(finalTracks(i).frames) = expNewSeq(i).states';
        actualratio_out(finalTracks(i).frames) = expStates(i).actualratio';
    end
end

