function copyfiles_recursively( topdir, outpath, string )
%little script to copy all files matching a regular expression to an
%outpath

D = rdir([topdir '\**\' string]);

for i = 1:length(D)
    copyfile(D(i).name,outpath);
end

end

