
#Main variables
Static_features = [chr(i) for i in range(ord('a'), ord('z')+1)] #array of elements from a to z [a,b..,z]
weights = [0.2]*len(Static_features)
bias = 0.5

#features and labels
def countletters(guess):
    features = [0]*26
    for i in guess:
        if i in Static_features:
            features[Static_features.index(i)] += 1
    return features
def calculateWeightedSum(guess):
    features = countletters(guess)
    sum = 0.0
    for i in range(len(features)):
        sum += features[i] * weights[i]
    sum += bias
    return sum, features
def theSigmoid(x):
    return 1.0 / (1.0 + 2.71828 ** -x)
def predictY(guess):
    z = calculateWeightedSum(guess)[0]
    prob = theSigmoid(z)
    return prob >= 0.6

#To train, design a one hot encoding. Below is an example, trains the AI to look for letters John
trainingdataX = ["John", "dave", "si james ay nasa house or bahay","Arron and John are friends","ohnj","abcdefghiklmpqrstuvwxyz"]
trainingdataY = [1, 0, 0, 1, 1,0]
def train(epochs, trainingdata, trainingdataY):
    global weights
    global bias
    LEARNING_RATE = 0.2
    print(f"Training begun")
    for i in range(0,epochs):
        for letters in trainingdata:
            countedL = calculateWeightedSum(letters)
            guessY = theSigmoid(countedL[0])
            correctY = trainingdataY[trainingdata.index(letters)]
            Ydifference = guessY - correctY
            for n in range(len(weights)):
                weights[n] -= LEARNING_RATE * Ydifference * countedL[1][n]
            bias -= LEARNING_RATE * Ydifference
        if i % 1000 == 0:
            print(f"Epoch: {i}/{epochs}")
    print(f"Epoch {epochs}/{epochs}")


def inference():
    customData = ["john","john is at home","cat","jo_hn","j"]
    for i in customData:
        print(f"{predictY(i)}   find letters 'john' == {i}")
    #use predictY("") to input any string
def customInference():
    pass

def start():
    train(1000,trainingdataX,trainingdataY)
    

    inference()


start()