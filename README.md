# Deepfake Detection

Deepfakes, videos or images containing a modified or swapped face produced by a deep neural network, are an increasing and unprecedented threat. Online deepfake content is doubling every six months. As of June 2020, approximately 50,000 deepfake videos were identified by \cite{ajder2019state}. An overwhelming 63\% of all deepfake videos online on June 2020 contained pornographic content. The victims suffer loss of professional and educational opportunities, stalking, psychological damage and defamation \citep{citron2014criminalizing}. An additional threat of deepfakes presents itself as undetected political deepfake material, which is capable of severe damage in election campaigns as well as in financial markets (Nguyen et al., \citeyear{nguyen2019deep}). The increasing prevalence of deepfakes results in a society suspicious of online visual content. If deepfake creation becomes more accessible and harder to detect, all online images and videos could lose credibility. 

This GitHub supports our Capstone Project titled **'Deepfake Detection: Building the Optimal Model'**, submitted for the MSc Data Science from the London School of Economics, carried out in conjunction with SAMSUNG. 

## Objectives
The paper's overarching purpose is to develop an optimal model for deepfake detection, maximising its classification accuracy. 

1. Reproduce the results of *FaceForensics++* (https://github.com/ondyari/FaceForensics)
2. Determine whether data augmentation and its optimisation as well as the correct hyperparameter selection can improve model performance and generalisation.
3. Determine whether temporal model architectures produce superior results when compared to frame-based architectures and their corresponding aggregation techniques.

## Objective 1.
To achieve this, the results were reproduced within /training/baseline_netorks.ipynb.

## Objective 2.
For hyperparameters, most analysis is performed within /hyperparameters/. For data augmentation, most analysis is performed within /augmentations/. Simple features, a peculiar type of augmentation based on https://github.com/cc-hpc-itwm/DeepFakeDetection is within /simplefeatures/.

## Objective 3.
All code for temporal models is within /transformer/.

## Overarching purpose
The final model is a combination of different configs within /configs/, and finalised within /ensembles/. The /predictions/ notebook is used to obtain video accuracies for almost all methods. The final pipeline looks as follows:



