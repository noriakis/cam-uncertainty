### Introducing uncertainty to class activation mapping in convolutional neural networks
Jupyter notebooks are for reproducing the visualization involving the general image classification.  

- models: directory storing models.
- weights: directory storing model weights.
	- best_weights_ECG.h5 (for 1DCNN model)
	- best_weights_CXP.h5 (for DenseNet121 with dropout)
- images: directory storing the numpy array of tested images (resized to 224 x 224) and the resulting figure.

#### The dataset used:
- [PTB-XL](https://physionet.org/content/ptb-xl/)
- Chest-Xray14 (with the annotation provided by RSNA)
- ImageNet

#### 2DCNN
Using Monte-Carlo dropout to improve the localization performance of Grad-CAM and Score-CAM.

##### Standard CNN with MC dropout
![Example CAM image (chest-xray image)](images/example_cxp.png)

#### 1DCNN
Improved explainability of classification of electrocardiogram using Score-CAM and Bayesian CNN.
The blue point indicates the CAM with high confidence, while red indicates low confidence. The pink background indicates CAM aboe 90th percentile of all CAM, and the green background indicates CAM below 10th percentile of all coefficient of variance.
![Example CAM image (ECG)](images/example_ecg.png)