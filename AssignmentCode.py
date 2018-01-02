import pandas as pd 
import numpy as np
import time

dataset = pd.read_csv("data.csv")

#========================================== Data Helper Functions ==========================================
def preProcessData(dataset, encoding, testTrainRatio):
    temp = dataset.copy()
    temp = temp.drop(["can_id", "can_nam"], axis=1)
    if(encoding):
        features = temp.drop(["winner", "can_off", "can_inc_cha_ope_sea"], axis=1)
        features = (features - features.min()) / (features.max() - features.min())
        features["can_off"] = temp['can_off']
        features["can_inc_cha_ope_sea"] = temp['can_inc_cha_ope_sea']
    else:
        temp = pd.get_dummies(temp, columns=["can_off", "can_inc_cha_ope_sea"])
        features = temp.drop(["winner"], axis=1)
        features = (features - features.min()) / (features.max() - features.min())

    features = features.values
    labels = temp["winner"].astype(int).values

    tr = int(len(temp)* testTrainRatio)
    return features[:tr], features[tr:], labels[:tr], labels[tr:]

def reshape(x):
    return x.reshape(x.shape[0], 1)

def accuracy(y_pred, y_test):
    predictions = reshape(np.array(y_pred))
    y_test = reshape(np.array(y_test))
    return ((predictions == y_test).sum() / float(y_test.size))

#===========================================================================================================

class KNN:
    def __init__(self):
        self.train_features = None
        self.train_labels = None
        self.k = 15
        return;

    def train(self, features, labels):
        self.train_features = features
        self.train_labels = labels
        return;

    def predict(self, features):
        y_pred = []
        k = self.k
        X_train = self.train_features
        y_train = self.train_labels
        for Xi in features:
            temp = []
            for j, Xj in enumerate(X_train):
                distance = np.sqrt(np.sum((Xi-Xj)**2))
                temp.append([distance, y_train[j]])
            temp.sort(key=lambda x: x[0])
            class1 = 0
            class2 = 0
            for i in range(k):
                if(temp[i][1] == 0):
                    class1 += 1
                else:
                    class2 += 1
            if class1 > class2:
                y_pred.append(0)
            else:
                y_pred.append(1)
        return y_pred;

class Perceptron:
    def __init__(self):
        self.l_rate = 0.01
        return;

    def train(self, features, labels):
        feature_len, feature_width = features.shape
        labels = labels*2 - 1
        self.w = np.random.uniform(-0.1, 0.1, feature_width+1)
        timeout = time.time() + 60
        while (time.time()<timeout):
            for i in range(feature_len):
                temp = self.w[0]
                for j in range(feature_width):
                    temp += self.w[j+1] * features[i][j]
                temp = self.sign(temp)
                if temp != labels[i]:
                    self.w[1:] += self.l_rate * labels[i] * features[i]
                    self.w[0] += self.l_rate * labels[i]
        
    def sign(self, x):
        if x > 0:
            return 1
        else: return -1

    def predict(self, features):
        if len(features.shape) > 1:
            feature_len, feature_width = features.shape
        else:
            feature_len = 1
            feature_width = self.w.shape[1] - 1
        y_pred = []
        for i in range(feature_len):
            temp = self.w[0]
            for j in range(feature_width):
                temp += self.w[j+1] * features[i][j]
            y_pred.append(self.sign(temp))
        y_pred = (np.array(y_pred) + 1)/2
        return y_pred;

class MLP:
    def __init__(self):
        self.input_nodes = 0
        self.l_rate = 0.01
        return;

    def sigmoid (self, x):
        return 1/(1 + np.exp(-x))

    def train(self, features, labels):
        feature_len, feature_width = features.shape
        feature_width += 1
        self.input_nodes = feature_width
        self.weights_0 = np.array([[np.random.uniform(-0.1, 0.1) for x in range(feature_width)] for y in range(feature_width)])
        self.weights_1 = np.array([[np.random.uniform(-0.1, 0.1) for x in range(1)] for y in range(feature_width)])
        
        features = np.c_[features, np.ones(feature_len)]
        labels= labels.reshape(feature_len,1)
        
        timeout = time.time() + 60
        while (time.time()<timeout):
            #calculating hidden layer inputs and outputs
            inp_hidden = np.dot(features, self.weights_0)
            out_hidden = self.sigmoid(inp_hidden)
            
            #calculating output layer inputs and outputs
            inp_output = np.dot(out_hidden, self.weights_1)
            out_output = self.sigmoid(inp_output)
            
            #calculating delta for output layer and hidden layer
            delta_out = -(labels - out_output) * (out_output*(1-out_output))
            delta_hidden = (np.dot(delta_out, self.weights_1.T) * (out_hidden*(1 - out_hidden)))
            
            #updating weights
            self.weights_1 = self.weights_1 - self.l_rate*(np.dot(out_hidden.T, delta_out))
            self.weights_0 = self.weights_0 - self.l_rate*(np.dot(features.T, delta_hidden))
        return;

    def sign(self, x):
        return 1*(x>0.5)

    def predict(self, features):
        feature_len, feature_width = features.shape
        features = np.c_[features, np.ones(feature_len)]
        
        inp_hidden = np.dot(features, self.weights_0)
        out_hidden = self.sigmoid(inp_hidden)
        
        inp_output = np.dot(out_hidden, self.weights_1)
        out_output = self.sigmoid(inp_output)
        y_pred = out_output
        y_pred = self.sign(y_pred)
        return y_pred;


class ID3:
    def __init__(self):
        self.decisionTree = dict()
        return;

    def infoGain(self, features, featureIndex, labels):
        classes = np.unique(features[:, featureIndex])
        if(len(classes)>5):
            features[:, featureIndex] = self.bucket(features, featureIndex)
        classes = np.unique(features[:, featureIndex])
            
        temp = np.append(features, (labels.reshape(len(labels), 1)), axis=1)

        entropy_sum = 0.0
        for x in classes:
            labelSubset = temp[temp[:, featureIndex] == x][:, temp.shape[1] -1]
            t1 = float(len(labelSubset)) / len(labels)
            entropy_sum += t1 * self.entropy(labelSubset)
        return self.entropy(labels) - entropy_sum

    def entropy(self, labels):
        classes = np.unique(labels)
        result = 0.0
        for x in classes:
            labelSubset = labels[labels[:] == x]
            temp = float(len(labelSubset)) / len(labels)
            result += temp * np.log2(temp)
        return -result

    def pluralityValue(self, labels):
        totalLength = len(labels)
        zeroLength = len(labels[labels[:] == 0])
        if(zeroLength > totalLength/2):
            return 0
        else:
            return 1

    def bucket(self, features, featureIndex):
        bucket = []
        for i in range(features.shape[0]):
            bucket.append(self.getRange(features[i, featureIndex]))
        return bucket

    def bucketContinousFeatures(self, features):
        feature_len, feature_width = features.shape
        for i in range(feature_width):
            if(len(np.unique(features[:, i])) > 5):
                features[:, i] = self.bucket(features, i)
        return features

    def train(self, features, labels):
        feature_names = [i for i in range(features.shape[1])]
        for i in range(features.shape[1]):
            classes = np.unique(features[:, i])
            if(len(classes) > 5):
                features[:, i] = self.bucket(features, i)
        self.decisionTree["parent"] = None
        self.train_aux(features, labels, self.decisionTree, feature_names)
        return;
 
    def train_aux(self, features, labels, tree, feature_names):
        if(len(labels) < 1):
            tree["value"] = tree["value"]
            tree["children"] = None
            return
        elif(len(np.unique(labels)) == 1):
            tree["children"] = None
            tree["value"] = labels[0]
            return
        elif(len(feature_names) == 0):
            tree["children"] = None
            tree["value"] = self.pluralityValue(labels)
            return
        else:
            maxGainFeature = None
            maxGainValue = float("-inf")
            for i in feature_names:
                gain = self.infoGain(features, i, labels)
                if(gain > maxGainValue):
                    maxGainValue = gain
                    maxGainFeature = i

            if(maxGainValue > 0):
                feature_names.remove(maxGainFeature)
                tree["value"] = self.pluralityValue(labels)
                tree["splitingFeature"] = maxGainFeature
                tree["children"] = dict()
                children = tree["children"]

                classes = np.unique(features[:, maxGainFeature])
                temp = np.append(features, (labels.reshape(len(labels), 1)), axis=1)
                for x in classes:
                    featureSplit = temp[temp[:, maxGainFeature] == x][:, :-1]
                    labelSplit = temp[temp[:, maxGainFeature] == x][:, -1]
                    child_node = dict()
                    child_node["parent"] = tree
                    self.train_aux(featureSplit, labelSplit, child_node, feature_names)
                    children[str(x)] = child_node
                return
            else:
                tree["children"] = None
                tree["value"] = self.pluralityValue(labels)
                return

    def getRange(self, value):
        i = 0.2
        j = 0
        while(i<=1.0):
            if(value<i):
                return j
            i += 0.2
            j += 1
        return 4

    def predict(self, features):
        for i in range(features.shape[1]):
            classes = np.unique(features[:, i])
            if(len(classes) > 5):
                features[:, i] = self.bucket(features, i)
        tree = self.decisionTree
        y_pred = []
        for x in features:
            temp = tree
            while temp["children"] != None:
                splittingFeature = temp["splitingFeature"]
                temp = temp["children"][str(x[splittingFeature])]
            y_pred.append(temp["value"])
        return y_pred


#===========================================================================================================

def main():
    # preprocessData(data, encoding, testTrainRatio)
    #encoding = True for ID3
    X_train, X_test, y_train, y_test = preProcessData(dataset, False, 0.8)

    classifier = MLP()
    classifier.train(X_train, y_train);
    y_pred = classifier.predict(X_test);

    print accuracy(y_pred, y_test)

main();