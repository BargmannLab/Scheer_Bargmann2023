# LawnLeavingTracker README

This folder contains the code for LawnLeavingTracker: a MATLAB-based set of scripts that can find and track single *C.elegans* animals on small lawns of *E.coli* seeded on agar plates. This code is optimized to find the edges of these bacterial lawns and identify and classify different types on animal interactions with the lawn boundary: head pokes, in which an animal pokes its head outside the bacterial lawn but does not leave, and lawn leaving, when animals poke their head out of the lawn followed by full body exploration of the agar area outside the bacterial food. The code also extracts the fully body shape and position of the animal by extracting a 49-point spline so that the movements of different points along the body can be tracked together. This allows determination of many other features including speed, angular speed, etc.

**Workflow**
1) run either *tracking_enter_exit_010219* for a single folder of videos or *tracking_enter_exit_BATCH_061919_freehand* for a folder of folders of videos, each cropped around the arena containing the worm.
These codes generate a FINAL.mat file per folder of videos after tracking completes.

2) run *copyfiles_recursively* to move these FINAL.mat files to a new folder so we can collect all data for a given genotype/condition.

3) inside that directory, run *track_QC_010219* to exclude animals that do not meet criteria for post-processing. The FINAL.mat files that pass QC will be in a subfolder called \_passQC

4) run *mergedata_newFeatures_082421* inside \_passQC to merge all data and derive some new behavioral features.
This code will generate two files per genotype/condition: \_newFeatures.mat and \_Filenames.h5

5) Compile all of these files in a folder. Then proceed to LawnLeavingAnalysis.

**This code draws inspiration and borrows from the following codebases:**

Schwarz RF, Branicky R, Grundy LJ, Schafer WR, Brown AE. Changes in Postural Syntax Characterize Sensory Modulation and Natural Variation of C. elegans Locomotion. PLoS Comput Biol. 2015 Aug 21;11(8):e1004322. doi: 10.1371/journal.pcbi.1004322. PMID: 26295152; PMCID: PMC4546679.
https://github.com/aexbrown/Behavioural_Syntax

Hums I, Riedl J, Mende F, Kato S, Kaplan HS, Latham R, Sonntag M, Traunmüller L, Zimmer M. Regulation of two motor patterns enables the gradual adjustment of locomotion strategy in Caenorhabditis elegans. Elife. 2016 May 25;5:e14116. doi: 10.7554/eLife.14116. PMID: 27222228; PMCID: PMC4880447.
https://github.com/openworm/SegWorm

López-Cruz A, Sordillo A, Pokala N, Liu Q, McGrath PT, Bargmann CI. Parallel Multimodal Circuits Control an Innate Foraging Behavior. Neuron. 2019 Apr 17;102(2):407-419.e8. doi: 10.1016/j.neuron.2019.01.053. Epub 2019 Feb 26. PMID: 30824353; PMCID: PMC9161785.
https://github.com/navinpokala/BargmannWormTracker

Brown AE, Yemini EI, Grundy LJ, Jucikas T, Schafer WR. A dictionary of behavioral motifs reveals clusters of genes affecting Caenorhabditis elegans locomotion. Proc Natl Acad Sci U S A. 2013 Jan 8;110(2):791-6. doi: 10.1073/pnas.1211447110. Epub 2012 Dec 24. PMID: 23267063; PMCID: PMC3545781.

**The following MATLAB Toolboxes are required:**
- Signal Processing Toolbox
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox
- Computer Vision Toolbox

**The following codes from the MATLAB file exchange were used:**

- John D'Errico (2023). arclength (https://www.mathworks.com/matlabcentral/fileexchange/34871-arclength), MATLAB Central File Exchange. Retrieved Jan 15, 20219.

- Manuel Guizar (2023). Efficient subpixel image registration by cross-correlation (https://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation), MATLAB Central File Exchange. Retrieved Dec 19, 2017.

- Brandon Kuczenski (2023). hline and vline (https://www.mathworks.com/matlabcentral/fileexchange/1039-hline-and-vline), MATLAB Central File Exchange. Retrieved June 16, 2018.

- Daniel Kovari (2023). imrect2 (https://www.mathworks.com/matlabcentral/fileexchange/53758-imrect2), MATLAB Central File Exchange. Retrieved Nov 17, 2020.

- NS (2023). Curve intersections (https://www.mathworks.com/matlabcentral/fileexchange/22441-curve-intersections), MATLAB Central File Exchange. Retrieved Jan 7, 2018.

- Dirk-Jan Kroon (2023). 2D Line Curvature and Normals (https://www.mathworks.com/matlabcentral/fileexchange/32696-2d-line-curvature-and-normals), MATLAB Central File Exchange. Retrieved Sep 13, 2017.

- Jos (10584) (2023). PADCAT (https://www.mathworks.com/matlabcentral/fileexchange/22909-padcat), MATLAB Central File Exchange. Retrieved Jan 9, 2018.

- Xavier Xavier (2023). Range intersection (https://www.mathworks.com/matlabcentral/fileexchange/31753-range-intersection), MATLAB Central File Exchange. Retrieved Jan 23, 2018.









