function vector_ints = get_intervals( vector, split_number )
%GET_INTERVALS.m This function takes in a vector and a value on which to split the vector into chunks

if length(vector)==1
    vector_ints = [];
    return;
end

a = find(vector==split_number);
x = diff(vector==split_number);
if ~isempty(a)
    split_starts = a((diff(a)~=1)');
    split_starts = unique([split_starts;a(end)]);%add back last zero member, sometimes lost
    a_diff = (diff(a)==1)';
    f = find([false,a_diff]~=[a_diff,false]);
    g = find(f(2:2:end)-f(1:2:end-1));
    split_ends = a(f(2*g-1));
    splits = unique(sort([1;split_starts;split_ends;length(vector)])); %these are all breakpoints
    split_int = [splits(1:end-1)+1 splits(2:end)-1];
    
    %if the first or last element of the vector is not a changepoint,
    %make sure the first index is 1 and the last index is the last number in the array
    if ~abs(x(1))==1 
        split_int(1) = 1;
    end
    if ~abs(x(end))==1
        split_int(end) = length(vector); 
    end
    
    split_int(split_int(:,1)-split_int(:,2)>0,:) = []; %get rid of impossible intervals
    % and intervals to keep, those containing nonzero, nonnan entries
    vector_ints = [];
    for n = 1:size(split_int,1)
        c = vector(split_int(n,1):split_int(n,2));
        if sum(c==0)~=length(c)
            vector_ints = [vector_ints ; split_int(n,:)];
        end
    end
        
else %no need to split this track further
    vector_ints = [1 length(vector)];
end

end

