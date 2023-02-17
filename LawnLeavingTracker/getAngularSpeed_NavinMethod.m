function AngSpeed = getAngularSpeed_NavinMethod(centroid)
%GETANGULARSPEED2.m extracts the angular speed in deg/sec from the centroid
%path. This method smoothly interpolates the centroid path at 5x the
%sampled rate to overcome aliasing problems. Then uses LineCurvature to
%extract theta at every point (distance/tangent circle radius), this can
%then be smoothly differentiated to come up with angular speed.

% SmoothWinSize = 1*3;
StepSize = 1*3;

cent_x = centroid(:,1);
cent_y = centroid(:,2);

%fill in any missing data
cent_x = fillmissing(cent_x,'linear');
cent_y = fillmissing(cent_y,'linear');

% Calculate Direction & Speed
Xdif = CalcDif(cent_x, StepSize) * 3;
Ydif = -CalcDif(cent_y, StepSize) * 3;    % Negative sign allows "correct" direction

% direction 0 = Up/North
ZeroYdifIndexes = Ydif == 0;
Ydif(ZeroYdifIndexes) = eps;     % Avoid division by zero in direction calculation

Direction = atan(Xdif./Ydif)*360/(2*pi);	    % In degrees, 0 = Up ("North")

NegYdifIndexes = find(Ydif < 0); % this code puts everything around a circle in -180 to +180 degrees
Index1 = find(Direction(NegYdifIndexes) <= 0);
Index2 = find(Direction(NegYdifIndexes) > 0);
Direction(NegYdifIndexes(Index1)) = Direction(NegYdifIndexes(Index1)) + 180;
Direction(NegYdifIndexes(Index2)) = Direction(NegYdifIndexes(Index2)) - 180;

% Calculate angular speed
AngSpeed = CalcAngleDif(Direction, StepSize)*3;		% in deg/sec

end

function Dif = CalcDif(Data, StepSize)			% StepSize MUST be > 0

% This function calculates the (approx.) derivative of the vector *Data*.
%
% Alignment issues:
% The function returns a vector (Dif) that is aligned with the input vector (Data).
% This means that the i-th index of Dif is the value of the derivative
% at the i-th index of Data. This value is calculated by taking the difference
% between the values located StepSize/2 samples on either side of Data(i),
% and dividing by StepSize.

% Edge effects: See code (too long to explain & not that important).

Len = length(Data);
HalfStepHi = ceil(StepSize/2);
HalfStepLo = floor(StepSize/2);

Dif(1) = Data(2) - Data(1);
for i = 2:HalfStepHi
    Dif(i) = (Data(2*i-1) - Data(1)) / (2*i-2);
end
Dif(HalfStepHi+1:Len-HalfStepLo) = (Data(StepSize+1:Len) - Data(1:Len-StepSize))/StepSize;
for i = 1:HalfStepLo-1
    Dif(Len-HalfStepLo+i) = (Data(Len) - Data(Len-2*HalfStepLo+2*i))/(2*HalfStepLo-2*i);
end
Dif(Len) = Data(Len) - Data(Len-1);
end

function Dif = CalcAngleDif(Data, StepSize)			% StepSize MUST be > 0

% This function is identical to CalcDif, except for the fact that it accounts for
% the periodic nature of Data about +/-180 deg.
% For detailed documentation see CalcDif.

Len = length(Data);
HalfStepHi = ceil(StepSize/2);
HalfStepLo = floor(StepSize/2);

Dif(1) = GetAngleDif(Data(1), Data(2));
for i = 2:HalfStepHi
    Dif(i) = GetAngleDif(Data(1), Data(2*i-1)) / (2*i-2);
end
Dif(HalfStepHi+1:Len-HalfStepLo) = GetAngleDif(Data(1:Len-StepSize), Data(StepSize+1:Len)) / StepSize;
for i = 1:HalfStepLo-1
    Dif(Len-HalfStepLo+i) = GetAngleDif(Data(Len-2*HalfStepLo+2*i), Data(Len)) / (2*HalfStepLo-2*i);
end
Dif(Len) = GetAngleDif(Data(Len-1), Data(Len));

return;

end

function Dif = GetAngleDif(x,y)

% This function subtracts x from y, taking into consideration
% "wrap-around" issues with a period of 360

Dif = y - x; 

Index = find(Dif > 180);
Dif(Index) = 360 - Dif(Index);

Index = find(Dif < -180);
Dif(Index) = -360 - Dif(Index);

% after wrapping, correct the few stragglers
Index = find(Dif > 180);
Dif(Index) = 180;

Index = find(Dif < -180);
Dif(Index) = -180;

return;
end


