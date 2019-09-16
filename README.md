# APTOS 2019 Blindness Detection [81/2943 ranking]

This project uses the following libraries:
```
keras
tensorflow-gpu
efficientnet==0.0.4
pandas
numpy
scipy
imgaug
cv2 (openCV)
tqdm
sklearn
```

# About this project

## What is this repository?
This repo includes my entire approach to Kaggle's APTOS 2019 Blindness Detection challenege. Not only does this include all the code necessary to reproduce my score, but also the trained model weights!


- [ ] TODO: update repo for gold zone score

## What is Diabetic Retinopathy (DR)?
It is a disease caused by damage to the retina. If not treated, it can progress to blindness. Diagnosed by screening fundus imaging, the comparison is as follows:
![alt text](https://www.eatonrapidseyecare.com/wp-content/uploads/2017/08/Diabetic-Retinopathy_SS-Graphic-732x293.jpg)

Current method of diagnosis is defined by classes 0 to 4, with 0 indicating no presence of disesase with 4 indicating severe progression.

## Why does this matter?
DR causes blindness! Fortunately, detecting the disease before any further progression can prevent blindness. While millions are affected by DR, lack of access of medical diagnosis makes prevention difficult. A deep learning based detection, when tested to work well, can help diagnosis much more people and prevent those suffering from DR turning blind.

## My approach:
  - Since we are free to explore external dataset with Kaggle having hosted same competition from APTOS before, I experimented with pretraining model on 2015 competition data first, then finetuning in on 2019 data
  - Other experiments such as snapshot ensembling, cyclic LR with Adam, and augmentations followed
  - Best choice an ensemble of my own approach with a public kernel approach (thanks drHB!)
  - Ranked 81st out of approximately 3000 applicants, but actually ranked 129th during public LB, hence it indicates my model was robust to generalization.
  - Technical details in the Wiki.

## Challenges:
- Lack of correlation between CV and public LB, and later between private and public LB
- External data handling
- Augmentation/ preprocess handling
