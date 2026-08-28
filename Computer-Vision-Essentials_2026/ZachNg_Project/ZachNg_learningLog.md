## **Learning Log & Reflections from CVE Project**
## **Author:** Zachary Ng
## **Date Created on:** 5 May 2026

<br>

**The Concept:**
</br>

>
> **What technique did you research that was not covered in lecture?**
> 

> [***Reflection 01***]
>
> One technique that I have researched and used is Architectural Improvements using Batch Normalization, Dropout, MaxPooling2D and GlobalAveragePooling2D.
>
> It is interesting that I get to have the chance to learn this techniques in Computer Vision Essentials and it actually improved my model significantly than the baseline.
>
> There are two technique that I have researched, tried to do and it actually failed are Data Augmentation and Transfer Learning MobileNetV2.
>
> While they are interesting to learn, but they are very complex when it comes to building the structure, I failed on some parts and it wasn't an easy decision to drop these 2 techniques.
>  
> 

<br>

**The Implementation:**
</br>

>
> **Show the code where I applied this new technique.**
> 

> [***Reflection 02***]
>
> ## **Architectural Improvements using Dropout, BatchNormalization, MaxPooling2D and GlobalAveragePooling2D**
> ```python
> num_class = 53 #-> 53 categories the model will classify
>
> cnn_model = Sequential([ #-> Straight pipeline, data flows top to bottom
>       Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', activation='relu', input_shape=(96, 96, 3)),
>        #-> 32 filters, 3x3 sliding window, moves 1 pixel at a time,
>        #-> output stays same size as input, negatives turned to zero, expects 96x96 RGB images
>       BatchNormalization(), #-> Keeps values stable between layers
>       MaxPooling2D(2, 2), #-> Shrinks image by half (96x96 -> 48x48)
>       Dropout(0.25), #-> Randomly turns off 25% of neurons during training
>
>       Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
>       #-> Same as Block 1 but 64 filters, detects more patterns
>       BatchNormalization(), #-> Keeps values stable between layers
>       MaxPooling2D(2, 2), #-> Shrinks image by half again (48x48 -> 24x24)
>       Dropout(0.25), #-> Randomly turns off 25% of neurons during training
>       
>       Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'),
>       #-> Same concept, 128 filters, detects even more complex patterns
>       BatchNormalization(), #-> Keeps values stable between layers
>       Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'),
>       #-> Second 128-filter layer stacked before pooling
>       BatchNormalization(), #-> Keeps values stable between layers
>       MaxPooling2D(2,2),  #-> Shrinks image by half again (24x24 -> 12x12)
>       Dropout(0.4), #-> Randomly turns off 40% of neurons during training
>
>       GlobalAveragePooling2D(),  #-> Collapses each feature map into a single number, converts 3D to 1D
>       Dense(256, activation='relu'), #-> Fully connected layer, combines all learned features
>       Dropout(0.5), #-> Randomly turns off 50% of neurons, more aggressive for Dense layers
>       Dense(num_class, activation='softmax')
>       #-> 53 output scores converted to probabilities that add up to 1.0
>])
>
>cnn_model.summary() #-> Prints a table of all layers, output shapes and parameter counts
>
> cnn_model.compile(loss='sparse_categorical_crossentropy', #-> Loss function for multi-class with integer labels
>                   optimizer='adam', #-> Adjusts weights automatically during training
>                    metrics=['accuracy'] #-> Tracks accuracy during training
>)
>
> print("CNN Model compiled & ready for training")
>
> history = cnn_model.fit(X_train_tensor, y_train, epochs=16, validation_data=(X_val_tensor, y_val), verbose=1)
>
> print("Training Completed!")
>```
>
> ## **Transfer Learning Fine Tuning MobileNetV2 (Failed)**
>
>```python
> #This method has failed and it not is successful for Transfer Learning MobileNetV2
> #Completely Freeze the Base Model
>
> num_class = 53
>
> #Feature Extractor
> base_model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
> #Freeze the entire backbone for Phase 1
> base_model.trainable = False
>
> inputs = Input(shape=(96, 96, 3))
>
> #Standardise Pixels to [-1, 1]
> x = preprocess_input(inputs)
>
> x = base_model(x, training=False)
>
> x = GlobalAveragePooling2D()(x)
>
> x = Dense(256, activation='relu')(x)
>
> x = Dropout(0.5)(x)
>
> outputs = Dense(units=num_class, activation='softmax', name='predictor')(x)
>
> mobilenetV2_s1 = Model(inputs, outputs, name='Transfer_Learning_Model')
>
> mobilenetV2_s1.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
>
> print("Model (Frozen Layers) compiled and ready for training!")
>
> mobilenetV2_s1.fit(X_train_tensor, y_train, epochs=10, validation_data=(X_val_tensor, y_val), verbose=1)
>
> print(f"Total layers: {len(base_model.layers)}")
>
> def unfreeze_top_n(base_model, n):
>    base_model.trainable = True
>    for layer in base_model.layers[:-n]:
>        layer.trainable = False
>    trainable = sum(1 for l in base_model.layers if l.trainable)
>    print(f"Trainable layers: {trainable} / {len(base_model.layers)}")
>
> mobilenetV2_s2 = unfreeze_top_n(base_model, 30)
>
> mobilenetV2_s2.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
>
> print("Model (Fine Tuning Layers) compiled and ready for training!")
>
> mobilenetV2_s1.fit(X_train_tensor, y_train, epochs=10, validation_data=(X_val_tensor, y_val), verbose=1)
>```

<br>

**The Author's Reflection**

Date Created: 22 May 2026
Finalised on: 27 May 2026

> I am Zachary Ng, a 2nd year ITE student at 3 years Higher NITEC in AI Applications.
>
> My objective of writing this learning log and reflection is to show my understanding of advanced techniques in Computer Vision, my personal growth and maturity as a student.
>
> In this project, I attempted to explore Transfer Learning, Advanced Data Augmentation and Architectural Improvement.
>
> During my time working on this project, my prompting skills of asking the Gen AI tools was pretty much like a sloth that I put in the project brief, the whole code, output of the code and 4 pictures of the output. I felt it was okay at first but this was a snowball effect of relying too much on the tool. As a result of my output of the code ended up pretty badly with super bad results, rabbit-holes or loop holes of not sharing the concepts with deep dive of the code on what it really means. It was not professional.
>
>A sudden change of view on how I see and use Gen AI tools came from my 2 biggest wake up calls from my lectuer, Mr Brandon Gay through some of the meetings and reviews of my work after class on how I prompt the Gen AI tools to assist me. He showed me on how they are just here to assist in some parts and they will sometimes generate the worst and dire mistakes of the advanced techniques without the concept. He found out that my root cause was on the way how I prompt the Gen AI tools and advised me on taking small steps in reading what the code actually means.
>
> My Transfer Learning and Data Augmentation code were having some problems and the validation loss keeps going up as I encountered. With that, I decided to stop on Transfer Learning and Data Augmentation due to submission date. However, I managed to successfully implement Architectural Improvement and it improved by 25% bettter than the baseline.
>
> Through this experience, I concluded using Gen AI tools without understanding the purpose of the code, it might push me to a wrong concept and more lurking issues will surface and more work has to be done.
>
> This is the harshest truth and the biggest foreshadow that my lecturer gave about working in the tech industry.
>
> As a coder working with any software or program, I need to learn to bounce back from mistakes, setbacks, the hardest knocks of codes that didn't work well. It is a good growth to learn from a mentor who show me about how to balance the use of Gen AI tools and the role as a coder.