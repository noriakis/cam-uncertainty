### Introducing uncertainty to class activation mapping in convolutional neural networks
- models: directory storing models (for 2DCNN).
- weights: directory storing model weights (for 1DCNN).
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