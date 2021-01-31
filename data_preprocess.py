import numpy as np

from keras.utils import to_categorical


def preprocess(dataset):
    data  = np.load(dataset)
    x = data['x']
    y = data['y']
    print(x.shape)
    print(y.shape)
    sc = 152
    st = 44

    train_length = int(0.7*x.shape[0]/ (sc+st)) - 1
    test_length = int(0.3 * x.shape[0] / (sc+st)) - 1
    segment_length = train_length + test_length

    trainX = []
    trainY = []
    testX = []
    testY = []

    # ==========================每人73分=====================================
    i = 0
    while i < x.shape[0]:
        if i + segment_length >= x.shape[0]:
            break
        trainX.append(x[i:i + train_length])
        trainY.append(y[i: i + train_length])
        testX.append(x[i + train_length: i + segment_length])
        testY.append(y[i + train_length: i + segment_length])
        i += segment_length





    trainX = np.concatenate(trainX)
    trainY = np.concatenate(trainY)
    testX = np.concatenate(testX)
    testY = np.concatenate(testY)
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)
    print(testY.shape)
    filename = 'channel0.npz'
    np.savez(filename,
             trainX = trainX,
             trainY = trainY,
             testX = testX,
             testY = testY
             )


def dataload(dataset):
    data = np.load(dataset)
    trainX = data['trainX']
    trainY = data['trainY']
    testX = data['testX']
    testY = data['testY']

    temp_x = np.zeros((trainX.shape[0],3072))
    temp_x[:,36:3036]=trainX
    trainX = temp_x

    temp_x = np.zeros((testX.shape[0],3072))
    temp_x[:,36:3036]=testX
    testX = temp_x

    result = [trainX, trainY, testX, testY]
    return result



if __name__ == '__main__':
    preprocess('data.npz')




