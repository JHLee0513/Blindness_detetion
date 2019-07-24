# Blindness_detetion
Kaggle challenge:

## log

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
  
  Validation:
  LB:
  
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
  
  4. DenseNet 169
  no FC layers (did not know densenet does not have fc in the paper)
  loss: CE, optimizer: SGD with momentum
  Ben's pre-processing idea with RGB
  
  5. Auto cropping
  
  6. Different encoder
  
  7. Classificaitoion --> regression
  
  
