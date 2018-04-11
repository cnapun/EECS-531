# OVERVIEW

This dataset is constructed for EECS531-A4 at CWRU. It consists of 6 image sequences (in './image’) and its corresponding optical flows(‘./flow’). For each image sequence, there are 9 frames and all the optical flows are computed with reference to the first frame. The ‘u’ and ‘v’ components of a optical flow field are stored separately in TXT format where 'u' is the motion in horizontal direction and 'v' in vertical direction. For example, the 'seq<i>/flow<j>.u' and 'seq<i>/flow<j>.v' are the optical flow field of the 'seq<i>/frame1.png' relative to 'seq<i>/frame<j>.png'. 

# CONTENT

    - image
        - seq1
            - frame1.png
            - ...
            - frame9.png
        - ...
        - seq6
    - flow
        - seq1
            - flow2.u
            - flow2.v
            - ...
            - flow9.u
            - flow9.v
        - ...
        - seq6 


# HOW TO LOAD FLOW DATA
Load the ground truth optical flow in Matlab


    % both u and v are 256x256 matrix 
    u = dlmread('./data/flow/seq1/flow2.u');
    v = dlmread('./data/flow/seq1/flow2.v');


Load the ground truth optical flow in python
    
    % both u and v are 256x256 numpy matrix 
    u = numpy.loadtxt('./data/flow/seq1/flow2.u', dtype='double', delimiter=',');
    v = numpy.loadtxt('./data/flow/seq1/flow2.v', dtype='double', delimiter=',');

----------------
Copyright (c) 2018, Jing Chen 
All rights reserved.


