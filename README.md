# Assignment3

#### Final validation accuracy for base network is 81.97

#### Our model 

model = Sequential()


model.add(SeparableConv2D(48, 3, 3, border_mode='same', input_shape=(32, 32, 3)))

model.add(Activation('relu'))

model.add(SeparableConv2D(48, 3, 3,use_bias=False)) #rf = 3 , 30x30x48

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))


model.add(SeparableConv2D(96, 3, 3,use_bias=False)) #rf = 5, 28x28x96

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))


model.add(MaxPooling2D(pool_size=(2, 2))) #rf = 6, 14x14x96


model.add(SeparableConv2D(192,3,3,use_bias=False)) #rf = 10, 12x12x192

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))

model.add(SeparableConv2D(96,1,1,use_bias=False)) #rf =10 , 12x12x96

model.add(MaxPooling2D(pool_size=(2, 2))) #rf = 12, 6x6x96

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))

model.add(SeparableConv2D(192,3,3,use_bias=False)) #rf = 20,4x4x192 

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))

model.add(MaxPooling2D(pool_size=(2, 2))) #rf = 24, 2x2x192

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))

model.add(SeparableConv2D(96,1,1,use_bias=False)) #rf = 24, 2x2x96

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))

model.add(SeparableConv2D(96,2,2,use_bias=False)) #rf = 32,1x1x96

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.1))

model.add(SeparableConv2D(10,1,1)) #rf = 32, 1x1x10


model.add(Flatten())

model.add(Activation('softmax'))

#### logs

Epoch 1/50
390/390 [==============================] - 30s 76ms/step - loss: 1.3754 - acc: 0.5019 - val_loss: 1.4421 - val_acc: 0.5382

Epoch 2/50
390/390 [==============================] - 22s 57ms/step - loss: 0.9743 - acc: 0.6552 - val_loss: 1.1258 - val_acc: 0.6181

Epoch 3/50
390/390 [==============================] - 22s 57ms/step - loss: 0.8372 - acc: 0.7055 - val_loss: 0.8959 - val_acc: 0.6871

Epoch 4/50
390/390 [==============================] - 22s 57ms/step - loss: 0.7632 - acc: 0.7321 - val_loss: 0.8576 - val_acc: 0.7049

Epoch 5/50
390/390 [==============================] - 22s 57ms/step - loss: 0.7005 - acc: 0.7576 - val_loss: 0.8945 - val_acc: 0.6875

Epoch 6/50
390/390 [==============================] - 22s 57ms/step - loss: 0.6585 - acc: 0.7707 - val_loss: 0.7528 - val_acc: 0.7396

Epoch 7/50
390/390 [==============================] - 22s 57ms/step - loss: 0.6280 - acc: 0.7814 - val_loss: 0.6714 - val_acc: 0.7690

Epoch 8/50
390/390 [==============================] - 22s 57ms/step - loss: 0.5952 - acc: 0.7929 - val_loss: 0.7462 - val_acc: 0.7432

Epoch 9/50
390/390 [==============================] - 22s 57ms/step - loss: 0.5714 - acc: 0.8022 - val_loss: 0.8171 - val_acc: 0.7199

Epoch 10/50
390/390 [==============================] - 22s 57ms/step - loss: 0.5523 - acc: 0.8060 - val_loss: 0.6374 - val_acc: 0.7784

Epoch 11/50
390/390 [==============================] - 22s 57ms/step - loss: 0.5370 - acc: 0.8110 - val_loss: 0.7059 - val_acc: 0.7642

Epoch 12/50
390/390 [==============================] - 22s 57ms/step - loss: 0.5113 - acc: 0.8207 - val_loss: 0.6230 - val_acc: 0.7813

Epoch 13/50
390/390 [==============================] - 22s 57ms/step - loss: 0.5048 - acc: 0.8227 - val_loss: 0.6658 - val_acc: 0.7678

Epoch 14/50
390/390 [==============================] - 22s 57ms/step - loss: 0.4936 - acc: 0.8274 - val_loss: 0.6497 - val_acc: 0.7795

Epoch 15/50
390/390 [==============================] - 22s 57ms/step - loss: 0.4698 - acc: 0.8351 - val_loss: 0.6351 - val_acc: 0.7870

Epoch 16/50
390/390 [==============================] - 22s 57ms/step - loss: 0.4639 - acc: 0.8358 - val_loss: 0.6584 - val_acc: 0.7835

Epoch 17/50
390/390 [==============================] - 22s 57ms/step - loss: 0.4536 - acc: 0.8415 - val_loss: 0.5941 - val_acc: 0.7970

Epoch 18/50
390/390 [==============================] - 22s 57ms/step - loss: 0.4436 - acc: 0.8447 - val_loss: 0.6361 - val_acc: 0.7867

Epoch 19/50
390/390 [==============================] - 22s 57ms/step - loss: 0.4311 - acc: 0.8494 - val_loss: 0.7044 - val_acc: 0.7665

Epoch 20/50
390/390 [==============================] - 22s 57ms/step - loss: 0.4209 - acc: 0.8517 - val_loss: 0.6341 - val_acc: 0.7906

Epoch 21/50
390/390 [==============================] - 22s 57ms/step - loss: 0.4158 - acc: 0.8542 - val_loss: 0.5614 - val_acc: 0.8118

Epoch 22/50
390/390 [==============================] - 22s 57ms/step - loss: 0.4036 - acc: 0.8586 - val_loss: 0.5979 - val_acc: 0.8011

Epoch 23/50
390/390 [==============================] - 22s 57ms/step - loss: 0.3940 - acc: 0.8615 - val_loss: 0.6743 - val_acc: 0.7811

Epoch 24/50
390/390 [==============================] - 22s 57ms/step - loss: 0.3917 - acc: 0.8627 - val_loss: 0.7284 - val_acc: 0.7631

Epoch 25/50
390/390 [==============================] - 22s 57ms/step - loss: 0.3795 - acc: 0.8659 - val_loss: 0.5864 - val_acc: 0.8100

Epoch 26/50
390/390 [==============================] - 22s 57ms/step - loss: 0.3799 - acc: 0.8664 - val_loss: 0.8823 - val_acc: 0.7231

Epoch 27/50
390/390 [==============================] - 22s 57ms/step - loss: 0.3727 - acc: 0.8661 - val_loss: 0.5986 - val_acc: 0.8025

Epoch 28/50
390/390 [==============================] - 22s 57ms/step - loss: 0.3625 - acc: 0.8729 - val_loss: 0.5445 - val_acc: 0.8187

Epoch 29/50
390/390 [==============================] - 22s 57ms/step - loss: 0.3573 - acc: 0.8728 - val_loss: 0.5717 - val_acc: 0.8130

Epoch 30/50
390/390 [==============================] - 22s 57ms/step - loss: 0.3517 - acc: 0.8750 - val_loss: 0.5218 - val_acc: 0.8252

Epoch 31/50
390/390 [==============================] - 22s 57ms/step - loss: 0.3484 - acc: 0.8775 - val_loss: 0.5416 - val_acc: 0.8212

Epoch 32/50
390/390 [==============================] - 22s 57ms/step - loss: 0.3438 - acc: 0.8787 - val_loss: 0.5518 - val_acc: 0.8233

Epoch 33/50
390/390 [==============================] - 22s 57ms/step - loss: 0.3423 - acc: 0.8779 - val_loss: 0.5736 - val_acc: 0.8117

Epoch 34/50
390/390 [==============================] - 22s 57ms/step - loss: 0.3291 - acc: 0.8827 - val_loss: 0.5608 - val_acc: 0.8142

Epoch 35/50
390/390 [==============================] - 22s 57ms/step - loss: 0.3288 - acc: 0.8850 - val_loss: 0.6198 - val_acc: 0.8039

Epoch 36/50
390/390 [==============================] - 22s 57ms/step - loss: 0.3262 - acc: 0.8837 - val_loss: 0.5676 - val_acc: 0.8177

Epoch 37/50
390/390 [==============================] - 22s 57ms/step - loss: 0.3230 - acc: 0.8858 - val_loss: 0.6260 - val_acc: 0.8002

Epoch 38/50
390/390 [==============================] - 22s 57ms/step - loss: 0.3138 - acc: 0.8889 - val_loss: 0.5799 - val_acc: 0.8152

Epoch 39/50
390/390 [==============================] - 22s 57ms/step - loss: 0.3106 - acc: 0.8895 - val_loss: 0.5733 - val_acc: 0.8169

Epoch 40/50
390/390 [==============================] - 22s 57ms/step - loss: 0.3141 - acc: 0.8886 - val_loss: 0.5811 - val_acc: 0.8155

Epoch 41/50
390/390 [==============================] - 22s 57ms/step - loss: 0.2973 - acc: 0.8958 - val_loss: 0.5865 - val_acc: 0.8147

Epoch 42/50
390/390 [==============================] - 22s 57ms/step - loss: 0.3065 - acc: 0.8936 - val_loss: 0.5482 - val_acc: 0.8241

Epoch 43/50
390/390 [==============================] - 22s 57ms/step - loss: 0.2993 - acc: 0.8930 - val_loss: 0.5539 - val_acc: 0.8244

Epoch 44/50
390/390 [==============================] - 22s 57ms/step - loss: 0.2935 - acc: 0.8950 - val_loss: 0.5512 - val_acc: 0.8231

Epoch 45/50
390/390 [==============================] - 22s 57ms/step - loss: 0.2954 - acc: 0.8953 - val_loss: 0.6735 - val_acc: 0.7925

Epoch 46/50
390/390 [==============================] - 22s 57ms/step - loss: 0.2899 - acc: 0.8963 - val_loss: 0.5414 - val_acc: 0.8258

Epoch 47/50
390/390 [==============================] - 22s 57ms/step - loss: 0.2900 - acc: 0.8962 - val_loss: 0.5520 - val_acc: 0.8198

Epoch 48/50
390/390 [==============================] - 22s 57ms/step - loss: 0.2780 - acc: 0.9009 - val_loss: 0.5925 - val_acc: 0.8067

Epoch 49/50
390/390 [==============================] - 22s 57ms/step - loss: 0.2857 - acc: 0.8973 - val_loss: 0.6240 - val_acc: 0.8063

Epoch 50/50
390/390 [==============================] - 22s 57ms/step - loss: 0.2753 - acc: 0.9028 - val_loss: 0.5652 - val_acc: 0.8257

### Highest Validation accuracy is 82.57

### Our team members are Hiranmai Tummalapalli and Raviteja Penugonda
