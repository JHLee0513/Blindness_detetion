# APTOS 2019 Blindness Detection [81/2943 ranking]

# Some TLDRs

* What is this repository?
This repo includes my entire approach to Kaggle's APTOS 2019 Blindness Detection challenege. Not only does this include all teh code necessary to reproduce my score, but also the trained model weights!

- [] TODO: cleanup
- [] TODO: model weights
- [] TODO: update repo for gold zone score

* What is Diabetic Retinopathy (DR)?
It is a disease caused by damage to the retina. If not treated, it can progress to blindness. Current method of diagnosis is defined by classes 0 to 4, with 0 indicating no presense of disesase with 4 indicating severe progression.

* why does this matter?
DR causes blindness! What makes this challenge more meaning, also, is that detecting the disease before any further progression can prevent blindness. While millions are affected by DR, lack of access of medical diagnosis makes prevention difficult. A deep learning based detection, when tested to work well, can help diagnosis much more people and prevent those suffering from DR turning blind.

# Data pipeline:

## Preprocessing (optional)



## Modeling

## Inference (performed in Kaggle kernels)

Test data is preprocess the same way as training data and fed into the model for detection. While the model outputs regression values, they are rounded and clipped to respective classes [0,4].
