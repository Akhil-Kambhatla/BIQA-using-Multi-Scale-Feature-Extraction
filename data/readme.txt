LABORATORY FOR IMAGE AND VIDEO ENGINEERING
in collaboration with
CENTER FOR PERCEPTUAL SYSTEMS
at The University of Texas at Austin


-----------COPYRIGHT NOTICE STARTS WITH THIS LINE------------
Copyright (c) 2003 The University of Texas at Austin
All rights reserved.

Permission is hereby granted, without written agreement and without
license or royalty fees, to use, copy, modify, and distribute this
database (the images, the results and the source files) and its 
documentation for any purpose, provided that the copyright 
notice in its entirity appear in all copies of this 
database, and the original source of this database, Laboratory for 
Image and Video Engineering (LIVE, http://live.ece.utexas.edu) and 
Center for Perceptual Systems (CPS, http://www.cps.utexas.edu) at the 
University of Texas at Austin (UT Austin, http://www.utexas.edu), 
is acknowledged in any publication that reports research using this database.
The database is to be cited in the bibliography as:

H. R. Sheikh, Z. Wang, L. Cormack and A. C. Bovik, "LIVE Image Quality 
Assessment Database", http://live.ece.utexas.edu/research/quality.

IN NO EVENT SHALL THE UNIVERSITY OF TEXAS AT AUSTIN BE LIABLE TO ANY PARTY 
FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES 
ARISING OUT OF THE USE OF THIS DATABASE AND ITS DOCUMENTATION, EVEN IF 
THE UNIVERSITY OF TEXAS AT AUSTIN HAS BEEN ADVISED OF THE POSSIBILITY OF 
SUCH DAMAGE.

THE UNIVERSITY OF TEXAS AT AUSTIN SPECIFICALLY DISCLAIMS ANY WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE DATABASE
PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF
TEXAS AT AUSTIN HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
ENHANCEMENTS, OR MODIFICATIONS.

The following input images are from the CD "Austin and Vicinity" by Visual Delights Inc. 
(http://www.visualdelights.net) coinsinfountain.bmp, dancers.bmp, 
flowersonih35.bmp, studentsculpture.bmp, carnivaldolls.bmp, cemetry.bmp, 
manfishing.bmp, churchandcapitol.bmp, building2.bmp
These images were modified from the original (resized) and then compressed 
to obtain images in the database. Permission to release these images and their 
distorted versions was graciously granted by Visual Delights Inc. These images may
not be used outside the scope of this database without their prior permission.
The rest of the images were public domain Kodak PhotoCD images obtained from the Internet.
-----------COPYRIGHT NOTICE ENDS WITH THIS LINE------------

Please contact Hamid Rahim Sheikh (sheikh@ece.utexas.edu) if you have any questions.
This investigators on this research were:
Hamid Rahim Sheikh (sheikh@ece.uetxas.edu) -- Department of ECE at UT Austin
Dr. Alan C. Bovik (bovik@ece.utexas.edu) -- Department of ECE at UT Austin
Dr. Lawrence Cormack (cormack@psy.utexas.edu) -- Department of Psychology at UT Austin
Dr. Zhou Wang (zhouwang@ieee.org)

-------------------------------------------------------------------------

The subjective experiment release comes with the following files:

* This readme file containing copyright information and usage information.
* Information file
* Two subjective score files containing raw subject scores in text format
* Two Matlab .mat files containing processed scores and mean scores as well
  as standard deviations obtained from processed scores.
* Three Matlab source files describing how the processed 
  scores were obtained from the raw scores
* individual subject score files (raw data)
* alot of images in bmp format. Images of the filename img%%.bmp are the ones 
  used in the testing. the rest were used to derive this database

DETAILS OF THE SUBJECTIVE TESTS
29 input images were used to create a database whose results are being provided. 
The JPEG2000 codec used was Kakadu version 2.2 that comes with the book: 
"JPEG2000 Image compression fundamentls, standards and practice" by David 
Taubman and Michael Marcellin, 2002 Kluwer Academic Publishers.

The command for generation of the database was: 
kdu_compress -i  source_filename -o destination_filename -rate bitrate -no_weights
The source, destination and the bitrates are given in the info file.

The testing procedure was as follows:
Twenty-nine high-resolution 24-bits/pixel RGB color images (typically 768 x 512) 
were compressed using JPEG2000 with different compression ratios to yield a database 
of 198 images, 29 of which were the original (uncompressed) images. The study was 
conducted in two sessions, with the original images included in both. Study 1 
contained images img1.bmp to img116.bmp, and study 2 contained images img117.bmp 
to img227.bmp. The bit rates were chosen such that the resulting distribution of 
quality scores for the compressed images was roughly uniform over the entire range. 
Each observer was shown the images randomly. Observers were asked to provide their 
perception of quality on a continuous linear scale that was divided into five equal 
regions marked with adjectives ”Bad”, ”Poor”, ”Fair”, ”Good” and ”Excellent”. The 
scale was then converted into 1-100 linearly. The testing was done in two sessions 
with about half of the images in each session. No viewing distance restrictions 
were imposed, display device configurations were identical and ambient illumination 
levels were normal indoor illumination. Subjects were asked to comfortably view the 
images and make their judgements. A short training preceded the session. 

INFORMATION FILE
The information file contains a list which describes how the database was created. 
Each line is separate entry in the image database: 
<Source image> <Destination image> <Bit rate>
A bit rate of 0.00000 means a loss-less copy of the source file.

SUBJECTIVE SCORE FILES	
Each image in the database is followed by the scores given to it by the different 
subjects. A score of 0 means that the subject skipped the image.

MATLAB .MAT FILES
These files (one for each session) have the following variables: 
mmt is the mean processed score for the image
mst is the standard deviation of the processed scores for the image
br is the bit rate used for that image. br==0 means LOSSLESS!
scores(i,:) is the array of processed score for image i. A score of zero implies 
that it was either skipped, or removed in the outlier removal step. "scores" only 
contains processed scores, which means that some subjects may have been removed. 
You may want to play with loaddatasort*.m to modify how the outlier detection is 
performed and how subjects are removed.

MATLAB SOURCE FILES
These files describe how the processed scores were obtained from the raw scores.

INDIVIDUAL SUBJECT SCORE FILES
Each line is an entry for an image (except for first line) of the form 
Image_number raw_score
where Image_number corresponds to filename img[Image_number].bmp