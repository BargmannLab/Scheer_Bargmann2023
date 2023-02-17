function [ movienames, curr_file ] = getnextvideo( movienames )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
curr_file = movienames{1};
movienames = movienames(2:end);
end

