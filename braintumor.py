import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,BatchNormalization,Dropout
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
encoder=OneHotEncoder()
encoder.fit([[0],[1]])
data=[]
paths=[] 
result=[]

#0-no brain tumor 
#1-diagnosed with brain tumor

#for yes 

for r,d,f in os.walk("D:/archive (1)/yes"):
    for file in f:
        if '.jpg' in file:
         paths.append(os.path.join(r,file)) 
        
for path in paths:   
    img=Image.open(path)
    img=img.resize((128,128))
    img=np.array(img)
    if(img.shape==(128,128,3)):
        data.append(np.array(img)) 
        result.append(encoder.transform([[1]]).toarray())
        
#for no 
for r,d,f in os.walk("D:/archive (1)/no"):
    for file in f:
        if '.jpg' in file:
         paths.append(os.path.join(r,file))
          
for path in paths:                                
    img=Image.open(path)
    img=img.resize((128,128))
    img=np.array(img) 
    if(img.shape==(128,128,3)):
        data.append(np.array(img)) 
        result.append(encoder.transform([[0]]).toarray()) 

data=np.array(data)  
    
result=np.array(result)
result=result.reshape(209,2)  

#training the model for classification
x_train,x_test,y_train,y_test=train_test_split(data,result,test_size=0.2,shuffle=True,random_state=0)
model=Sequential() 
model.add(Conv2D(32,kernel_size=(2,2),input_shape=(128,128,3),padding='Same')) 
model.add(Conv2D(32,kernel_size=(2,2),input_shape=(2,2),activation='relu',padding='same'))

model.add(BatchNormalization()) 
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))    

model.add(Conv2D(64,kernel_size=(2,2),activation='relu',padding='same'))
model.add(Conv2D(64,kernel_size=(2,2),activation='relu',padding='same'))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten()) 

model.add(Dense(512,activation='relu')) 
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='Adamax',metrics=['accuracy'])
model.summary()

history=model.fit(x_train,y_train,epochs=30,batch_size=40,verbose=1,validation_data=(x_test,y_test))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss') 
plt.xlabel('Epoch')
plt.legend(['Test', 'Validation'], loc='upper right')
plt.show()

#checking model by using sample data

def names(num):
    if(num==1):
        return "Diagnosed with Brain Tumor"
    else:
        return "Not Diagnosed with Brain Tumor "

img=Image.open("D:/archive (1)/yes/Y17.jpg")
x=np.array(img.resize((128,128)))
x=x.reshape(128,128)
res=model.predict_on_batch(x)
classify=np.where(res==np.amax(res))[1][0]
print(res[1][classify]*100+"%"+names(classify))
imshow(img)

img=Image.open("D:/archive (1)/no/17 no.jpg")
x=np.array(img.resize((128,128)))
x=x.reshape(1,8192,8192,3)
res=model.predict_on_batch(x)
classify=np.where(res==np.amax(res))[1][0]
print(res[0][classify]*100+"%"+names(classify)) 
imshow(img) 