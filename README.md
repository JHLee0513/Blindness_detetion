# Blindness_detetion
Kaggle challenge:

## log

1.
  DenseNet 121
  x = Dense(512, activation = 'relu', use_bias = True) (x)
  x = Dense(512, activation = 'relu', use_bias = True) (x)
  x = Dense(512, activation = 'relu', use_bias = True) (x)
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
                     
  #cycle = 2560/batch * 20
  #cyclic = CyclicLR(mode='exp_range', base_lr = 0.0005, max_lr = 0.01, step_size = cycle)
  40 epochs total
  
2. 
