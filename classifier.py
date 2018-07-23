
# Remember to update the script for the new data when you change this URL
URL = "./bitcoin_2018_5min.csv"

# Uncomment this call when using matplotlib to generate images
# rather than displaying interactive UI.
#import matplotlib
#matplotlib.use('Agg')

from pandas import read_table
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Reshape
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras.layers import LSTM, LeakyReLU
from keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

try:
    # [OPTIONAL] Seaborn makes plots nicer
    import seaborn
except ImportError:
    pass

# =====================================================================

def download_data():
    frame = read_table(
        URL,
        sep=',',            # comma separated values
        skipinitialspace=True,
        index_col=None,
        header=0, # use the first line as headers
    )

    frame[:] = frame[:].convert_objects(convert_numeric=True)
    frame = frame.dropna()
    frame.info()
    X = frame.iloc[0:-2, 1:8]
    Y = frame.iloc[0:-2, 8:]
    sc = MinMaxScaler()                           #scaling using normalisation
    X = sc.fit_transform(X)
    Y = sc.fit_transform(Y)
    return X, Y


# =====================================================================


# =====================================================================


if __name__ == '__main__':
    # Download the data set from URL
    print("Downloading data from {}".format(URL))
    X, Y = download_data()
    print(X, Y)
    tscv = TimeSeriesSplit(n_splits=3)
    print(tscv)

    for train_index, test_index in tscv.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[:len(train_index)], X[len(train_index): (len(train_index)+len(test_index))]
        Y_train, Y_test = Y[:len(train_index)], Y[len(train_index): (len(train_index)+len(test_index))]

    print(len(X_train), len(X_test))

    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    model=Sequential()                                                      #initialize the RNN

    model.add(LSTM(input_shape=(1,7),output_dim=50,return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(100,return_sequences=False))
    model.add(Dropout(0.2))   #adding input layerand the LSTM layer

    model.add(Dense(units=1))                                               #adding output layers

    model.compile(optimizer='rmsprop',loss='mean_squared_error', metrics = ['accuracy'])               #compiling the RNN

    history = model.fit(X_train,Y_train,batch_size=128,epochs=10, validation_data = (X_test,Y_test))
