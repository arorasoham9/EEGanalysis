import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

indData = [pd.read_csv('user_'+ user+'.csv') for user in ['a', 'b', 'c', 'd']]



for i in range(len(indData)):
    indData[i]['User'] = pd.Series(i, index = indData[i].index)

allData = pd.concat(indData, axis=0).sample(frac= 1.0, random_state=123).reset_index(drop = True)








def onehot_encode(indiData, column):
    indiData = indiData.copy()
    dummies = pd.get_dummies(indiData[column], prefix=column)
    indiData = pd.concat([indiData, dummies], axis = 1)
    indiData = indiData.drop(column, axis = 1)
    return indiData
def pre_process(indiData, target = 'Class'):
    indiData = indiData.copy()
    targets =['Class', 'User']
    targets.remove(target)
    indiData = onehot_encode(indiData, column= target[0])


    yAxis = indiData[target].copy()
    xAxis = indiData.drop(target,axis = 1)


    xTrain, xTest, yTrain, yTest=  train_test_split(xAxis,yAxis, train_size= 0.7, random_state=123)
    scaler = StandardScaler()
    scaler.fit(xTrain)
    xTrain = pd.DataFrame(scaler.transform(xTrain), columns= xAxis.columns)
    xTest = pd.DataFrame(scaler.transform(xTest),columns= xAxis.columns)
    return xTrain, xTest, yTrain, yTest

def build_model(num_layers = 3):
    inputs = tf.keras.Input(shape =(xTrain.shape[1],))
    x = tf.keras.layers.Dense(128, activation = 'relu')(inputs)
    x = tf.keras.layers.Dense(128, activation = 'relu')(x)
    output = tf.keras.layers.Dense(num_layers, activation = 'softmax')(x)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    model.compile(
        optimizer ='adam',
        loss ='sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )
    return model







xTrain, xTest, yTrain, yTest = pre_process(allData,target = 'Class')



class_model = build_model(num_layers = 3)

class_history = class_model.fit(
    xTrain,
    yTrain,
    validation_split = 0.2,
    batch_size = 32,
    epochs = 50,
    callbacks =[
        tf.keras.callbacks.EarlyStopping(
        monitor ='val_loss',
        patience = 3,
        restore_best_weights =True
    )])