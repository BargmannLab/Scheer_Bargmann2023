function Dif = CalcDif(Data, StepSize)          % StepSize MUST be > 0
 
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