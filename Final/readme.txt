For this project, I have explored the problems of deconvolution and image recovery from subsamplings of convolutions. The three notebooks TV, Tikhonov, and Gamma show a mathematical justification of each model, and the file img_utils.py contains the code for these models.

Results and analysis can be found in the Results notebook, and some alternative methods of determining optimal parameters can be found in ResultsBayesOpt

Implementation of Tikhonov regularization was done without any external aid, MATH 473 lecture slides were used to understand TV regularization and Split Bregman, and "Hypermodels in the Bayesian imaging framework" was followed for the implementation of the Gamma prior.

Calvetti, Daniela, and Erkki Somersalo. "Hypermodels in the Bayesian imaging framework." Inverse Problems 24, no. 3 (2008): 034013.