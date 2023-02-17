# LawnLeavingTracker README

This repository contains two code bases:
1) LawnLeavingTracker: a MATLAB-based set of scripts that can find and track single *C.elegans* animals on small lawns of *E.coli* seeded on agar plates. This code is optimized to find the edges of these bacterial lawns and identify and classify different types on animal interactions with the lawn boundary: head pokes, in which an animal pokes its head outside the bacterial lawn but does not leave, and lawn leaving, when animals poke their head out of the lawn followed by full body exploration of the agar area outside the bacterial food. The code also extracts the fully body shape and position of the animal by extracting a 49-point spline so that the movements of different points along the body can be tracked together. This allows determination of many other features including speed, angular speed, etc. This code draws inspiration and borrows from the following codebases:

Schwarz RF, Branicky R, Grundy LJ, Schafer WR, Brown AE. Changes in Postural Syntax Characterize Sensory Modulation and Natural Variation of C. elegans Locomotion. PLoS Comput Biol. 2015 Aug 21;11(8):e1004322. doi: 10.1371/journal.pcbi.1004322. PMID: 26295152; PMCID: PMC4546679.

Hums I, Riedl J, Mende F, Kato S, Kaplan HS, Latham R, Sonntag M, Traunmüller L, Zimmer M. Regulation of two motor patterns enables the gradual adjustment of locomotion strategy in Caenorhabditis elegans. Elife. 2016 May 25;5:e14116. doi: 10.7554/eLife.14116. PMID: 27222228; PMCID: PMC4880447.

López-Cruz A, Sordillo A, Pokala N, Liu Q, McGrath PT, Bargmann CI. Parallel Multimodal Circuits Control an Innate Foraging Behavior. Neuron. 2019 Apr 17;102(2):407-419.e8. doi: 10.1016/j.neuron.2019.01.053. Epub 2019 Feb 26. PMID: 30824353; PMCID: PMC9161785.
https://github.com/navinpokala/BargmannWormTracker

Brown AE, Yemini EI, Grundy LJ, Jucikas T, Schafer WR. A dictionary of behavioral motifs reveals clusters of genes affecting Caenorhabditis elegans locomotion. Proc Natl Acad Sci U S A. 2013 Jan 8;110(2):791-6. doi: 10.1073/pnas.1211447110. Epub 2012 Dec 24. PMID: 23267063; PMCID: PMC3545781.





