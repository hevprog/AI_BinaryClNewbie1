import pdb

dataset = ["apple", "banana", "grape", "orange", "kiwi", "peach", "plum","pigpenApp","IphoneApple","application","pineappe","app"]

trainingData = ["apple", "banana", "grape juice", "orange cat","car","house tour","Farmer","Henry", "SaPisoNakaLibingSiRizzal", #0-8 index
                "pineapple","applepen","allen","plumber","app","pplea","alppe","pinepple","philippines", "green apple"]#9-18 index
trainingDataTrueY = [1,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,1,0,1]


Static_features = [chr(i) for i in range(ord('a'), ord('z')+1)] #array of elements from a to z [a,b,c...,z]
weights = [0.2]*len(Static_features)
bias = 0.6


def countletters(guess): #managing da features
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
def predictY(guess): #amo adi na part na it model na guess hin label
    z = calculateWeightedSum(guess)[0]
    prob = theSigmoid(z)
    return prob >= 0.6

#-----evaluating na the model
TP=0
FP=0
def getTruePositives(mode):
    global TP 
    if mode: TP+=1
    else: return TP
def getFalsePositive(mode):
    global FP
    if mode: FP+=1
    else: return FP
def manageConfusionMatrix(Y,guessY):
    if Y==1 and guessY>=0.6: getTruePositives(1)
    if Y==0 and guessY>=0.6: getFalsePositive(1)
    pass
def precision(): #waray gamit pero pag measure man la adi on how correct is the guess
    return round((TP / (TP + FP)),3) if TP !=0 and FP !=0 else 0
#-------------------


def train(epochs):
    global weights
    global bias
    LEARNING_RATE = 0.2
    print(f"Training begun")
    for i in range(0,epochs):
        for letters in trainingData:
            countedL = calculateWeightedSum(letters)
            guessY = theSigmoid(countedL[0])
            correctY = trainingDataTrueY[trainingData.index(letters)]
            Ydifference = guessY - correctY
            manageConfusionMatrix(correctY,guessY)
            for n in range(len(weights)):
                weights[n] -= LEARNING_RATE * Ydifference * countedL[1][n]
            bias -= LEARNING_RATE * Ydifference
        if i % 1000 == 0:
            print(f"Epoch: {i}/{epochs} Precision {precision()}")
    print(f"Epoch {epochs}/{epochs}")




def inference(infer):
    print(f"\nInferencing....{infer}\n")
    if infer:
        customData = ["Random stuffs", "Waray apple didi", "Adi lugod na string waray Aple",":)", "astig HAHA","--a-trpp-rle","adi apple",]

        for i in customData:
            print(f"the letters in \"apple\" is {"++found++" if predictY(i) else "---------"}  in   {i} ")








train(10000)
for i in dataset:
    print(f"guessing that letters in \"apple\" in the data {i} is {"++found++" if predictY(i) else "--"}")


#inferencing and playing with da model
inference(False)