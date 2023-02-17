# LawnLeavingAnalysis README

This folder contains Jupyter notebooks that perform post-processing analysis in Python to quantify behavioral metrics relating to behavioral arousal states and lawn leaving decisions.

### **Requirements:**
All packages specified in pkg_requirements.txt

This code relies on the SSM package developed by the Linderman lab: https://github.com/lindermanlab/ssm (follow installation instructions there).

You must have a folder with matched \_newFeatures.mat and \_Filenames.h5 files (1 per condition/genotype).Then you can proceed to Data_Preprocessing.ipynb

Or you can download pre-processed data in pickle file format from the Dryad repository accompanying this publication and proceed directly to generating figures: ***(insert link here)***.

To run this code, make sure to change the paths to all data files, pickles, .csvs etc. where they are specified in the notebooks.

### **This code generates behavioral metrics per genotype/condition including the following most important ones(others are specified within preProcessing.py):**

*Radial_Dist*, the distance from the animal's nose tip from the center of the bacterial lawn. (mm/sec)

*Lawn_Boundary_Dist*, the distance from the animal's nose tip to the closest point on the lawn boundary.(mm/sec)

*Lawn_Entry*, boolean. True = frame when the animal began to enter the lawn.

*Lawn_Exit*, boolean. True = frame when the animal began to exit the lawn.

*HeadPokeFwd*, boolean. True = frame when the animal reached the maximum *Radial_dist* displacement outside the lawn during a head poke forward.

*HeadPokePause*, same for head poke pause.

*HeadPokeRev*, same for a head poke reversal.

*HeadPokesAll*, boolean. True = frame when the animal reached the maximum *Radial_dist* displacement for any type of head poke.

*Midbody_cent_x*, x coordinate of the Midbody point. (pixels)

*Midbody_cent_y*, y coordinate of the Midbody point. (pixels)

*Head_cent_x*, x coordinate of the Head point. (pixels)

*Head_cent_y*, y coordinate of the Head point. (pixels)

*Tail_cent_x*, x coordinate of the Tail point. (pixels)

*Tail_cent_y*, y coordinate of the Tail point. (pixels)

*Midbody_speed*, signed speed of the Midbody. Positive values indicate forward movement; negative, reverse.

*Head_speed*, signed speed of the Head. Positive values indicate forward movement; negative, reverse.

*Tail_speed*, signed speed of the Tail. Positive values indicate forward movement; negative, reverse.

*Quirkiness*, Q = sqrt(1 - (a^2/A^2)), where a is minor axis of the bounding box, A is major axis of bounding box.

*headAngVel_relMid*, the derivative of the angular displacement of the head relative to the midbody across frames.

*headRadVel_relMid*, the derivative of displacement of the head relative to the midbody across frames.

*Omega*, boolean. True = frame when animal posture was unsegmentable due to self-intersection.

*Centroid_x*, x coordinate of the Centroid of the bounding box around animal. (pixels)

*Centroid_y*, y coordinate of the Centroid of the bounding box around animal. (pixels)

*Centroid_speed*, speed of the Centroid. Values always greater than 0 (absolute speed, mm/sec)

*Centroid_angspeed*, angular speed of the Centroid. Calculated as the arc-cosine of the dot product between vectors formed from 3 consecutive Centroid positions across frames. (deg/sec)

*Midbody_angspeed*, same but calculated for Midbody position.

*Head_angspeed*, same but calculated for Head position.

*Tail_angspeed*, same but calculated for Tail position.

*MovingReverse*, boolean. True = animal is moving reverse.

*MovingForward*, boolean. True = animal is moving forward.

*Pause*, boolean. True = animal pausing (Midbody speed < 0.02 mm/sec)

*Midbody_absSpeed*, absolute value of Midbody_speed

*Midbody_fspeed*, Midbody_speed, where negative values are replaced with 0s.

*Midbody_rspeed*, Midbody_speed, where positive values are replaced with 0s. Inverted so it is also a positive number.

### **And versions of the same features binned into contiguous 10-second bins. A few other key features (which are only defined in the 10-second binned data format):**

*InLawnRunMask*, a boolean mask. True = a contiguous intervals in which the animal was inside the lawn.

*bin_LawnExit_mostRecent*, boolean. True = animal did a lawn leaving event within this 10-second interval.

*bin_HeadPokeFwd*, the mean number of head poke forwards in this 10-second interval.

*bin_HeadPokeRev*, same for head poke reversal.

*bin_HeadPokePause*, same for head poke pause.

*bin_HeadPokesAll*, same for all types of head pokes.

*bin_MovingForward*, fraction of time animal is moving forward within 10-second interval.

*bin_MovingReverse*, same for reverse.

*bin_Pause*, same for pause.

*RD_states_Matrix_exog_Cent*, These are the Roaming and Dwelling state calls based on a 2-state Hidden Markov Model found in *PD1074_od2_LL_Data_Centroid_RoamingDwellingHMM_081721.pkl*

*arHMM_MLstates*, These are the AR-HMM state calls based on a 4-state Autoregressive Hidden Markov Model found in *HMM_OnLawnOnly_ForwardFeaturesOnly_noQuirk_AR_062321.pkl*

### Additional files:
*PD1074_scalers_062321.pkl* contains StandardScalers (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) trained on wild type PD1074 data used to z-normalize new data before running it through AR-HMM segmentation steps.

*PD1074_od2_LL_Data_Centroid_RoamingDwellingHMM_081721.pkl* contains the Roaming and Dwelling HMM used in the paper.

*HMM_OnLawnOnly_ForwardFeaturesOnly_noQuirk_AR_062321.pkl* contains the AR-HMM used in the paper. After loading: arHMM_model = arHMMs_ALL[4][2] (see code): num states =4, KAPPA = 25,000

*presentation_smallerfonts.mplstyle* a style-sheet to make matplotlib figures.

*LIGHT.csv* a boolean vector representing the light OFF and light ON periods of stimulation used during optogenetics experiments.



