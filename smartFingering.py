import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Dropout

import pretty_midi

file_path = "C:/Users/USER/Desktop/Python" #change
trainingData = np.loadtxt(file_path) #data is hand crafted into a list format
trainingdf = pd.DataFrame(trainingData)

#training data processing
numLagFeatures = 3 
fingeringColumns = ['R1', 'R2', 'R3', 'R4', 'R5', 'L1', 'L2', 'L3', 'L4', 'L5']

for col in fingeringColumns: 
    for lag in range(1, numLagFeatures + 1):
        trainingdf[f'{col}_lag{lag}'] = trainingdf[col].shift(lag).fillna(0) #for fingering columns

for lag in range(1, numLagFeatures + 1):
    trainingdf[f'MIDI_lag{lag}'] = trainingdf['MIDI'].shift(lag).fillna(0) 
    trainingdf[f'linger_lag{lag}'] = trainingdf['linger'].shift(lag).fillna(0)

trainingdf = trainingdf.dropna() #make sure other data is unaffected
trainingdf['time gap'] = trainingdf['time'].diff().fillna(10)

def reshape(data, steps, targetCols, training): #steps is the windowing size over the examination of temporal dependencies. Notes 
    sequences = []                              #are likely to influence each other beyond 5 steps but for now let this be a start.
    targets = []

    for i in range(len(data)-steps+1):
        seq = data.iloc[i:i+steps, :].values #sliding window over all data

        if training:
            target = data.iloc[i+steps-1][targetCols].values
            targets.append(target)

        sequences.append(seq)

    if training:
        return np.array(sequences), np.array(targets)
    return np.array(sequences)

#reshaping time series data for LSTM whilst keeping temporal order
x, y = reshape(trainingdf, 5, fingeringColumns, True) 

model = Sequential([
    LSTM(32, activation='relu', input_shape=(x.shape[1], x.shape[2]), return_sequences=True),
    Dropout(0.2), 
    LSTM(32, activation='relu'),
    Dropout(0.2), 
    Dense(len(fingeringColumns), activation='linear')])

model.compile(optimizer='adam', loss='mse')
history = model.fit(x, y, epochs=50, batch_size=1, validation_split=0.2)
model.summary()

newData = pretty_midi.PrettyMIDI("filename.mid") #name.mid
newdf = pd.DataFrame(columns=['Time', 'time_diff', 'MIDI', 'linger'])
def incoDataProcessing(): #data processing for the new data applied to the model
    possibleLinger = []
    for instrument in newData.instruments: #only instrument should be piano anyway
        for note in instrument.notes:
            note_start_time = note.start
            index = 0

            #adding as a starting note
            if note_start_time in newdf['Time'].values:
                index = len(newdf)-1 #technically if it isn't new, index has to be the last row
                newdf.at[index, 'MIDI'].append(note.pitch)
                
                possibleLinger.append(note)
            else:
                addNote = pd.DataFrame([{'Time': note.pitch, 'time_diff': 0, 'MIDI': [], 'linger': []}])
                index = len(newData)
                pd.concat([newData, addNote], ignore_index=True)
                newdf.at[index, 'MIDI'].append(note.pitch)

                possibleLinger.append(note)
            
            #anything lingering at this moment in time should be added 
            copy = possibleLinger.copy()
            for lingerNote in copy:
                if lingerNote.end > note.start and lingerNote.start != note.start: 
                    newdf.at[index, 'linger'].append(lingerNote.pitch)
                elif lingerNote.end <= note.start: #lingering status revoked 
                    possibleLinger.remove(lingerNote)

            #adding time diff for the previous row if it exists and if valid (not the same time)
            if index > 0 and newdf.at[index-1, 'Time'] != note_start_time:
                newdf.at[index-1, 'time_diff'] = note_start_time - newdf.at[index-1, 'Time'] 

    #newdf should be in the exact same format as df from the training data
    #time, timediff, MIDI, linger, R1, R2, R3, R4, R5, L1, L2, L3, L4, L5

    #converting to sequences for LSTM
    return reshape(newdf, 5, [], False) 
    
predictions = model.predict(incoDataProcessing()) #predictions should output a 2d array where each row is
print(predictions.shape())                        #the predicted midi note values for each finger at every time step

#make easier to read
newdf['MIDI'] = newdf['MIDI'].apply(lambda x: ', '.join(map(str, x)))
newdf['linger'] = newdf['linger'].apply(lambda x: ', '.join(map(str, x)))

#add the predictions array to the original dataframe
