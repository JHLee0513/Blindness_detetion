# Blindness_detetion
The Readme will serve as the solution writeup.

THe kernel is no internet! save .h5 model instead of weights

# Some TLDRs

* What is Diabetic Retinopahty?

* why does this matter?
Well, detecting this condition early on will greatly help with prevention of blindness cause by this condition. Furthermore, *data* shows that it is difficult to diagnose them early on, which this application should eventually work towards to help. 

* How does this writeup work towards the solution?
This experiment involves CNN manipulation, pre-preocessing, post-processing, and overall problem tuning.

# log

1. Baseline.py

  intputs = 256x256x3 
  x = DenseNet 121. outputs
  x = Dense(1024, activation = 'relu', use_bias = True) (x)
  x = Dense(1024, activation = 'relu', use_bias = True) (x)
  x = Dense(1024, activation = 'relu', use_bias = True) (x)
  x = Dense(5, activation = 'softmax') (x)
  model = Model(inputs, x)

  model.compile(loss='categorical_crossentropy', optimizer = 'SGD',
             metrics= ['categorical_accuracy'])
  augmentations:
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     rescale = 1./255
                     
  cycle: 30 epochs per cycle, CyclicLR(mode='exp_range', base_lr = 0.0005, max_lr = 0.01, step_size = cycle)
  60 epochs total
  
  Validation: ~0.80
  LB:0.704
  
2. 
  Same model, same training, new preprocessing
  
3. 
intputs = 256x256x3 
  x = DenseNet 121. outputs
  x = Dense(1024, activation = 'relu', use_bias = True) (x)
  x = Dense(1024, activation = 'relu', use_bias = True) (x)
  x = Dense(1024, activation = 'relu', use_bias = True) (x)
  x = Dense(5, activation = 'softmax') (x)
  model = Model(inputs, x)

  model.compile(loss='categorical_crossentropy', optimizer = 'SGD',
             metrics= ['categorical_accuracy'])
  augmentations:
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     rescale = 1./255
                     
  cycle: 30 epochs per cycle, CyclicLR(mode='exp_range', base_lr = 0.0005, max_lr = 0.01, step_size = cycle)
  60 epochs total
  
  CV: 0.820
  LB: 0.716
  
  4. DenseNet 169
  no FC layers (did not know densenet does not have fc in the paper)
  loss: CE, optimizer: SGD with momentum
  Ben's pre-processing idea with RGB
  
  5. Auto cropping
  
  6. idea: edge detection preprocessing to force cnn to look more at shapes not texture
  
  7. Classificaitoion --> regression
  
  
