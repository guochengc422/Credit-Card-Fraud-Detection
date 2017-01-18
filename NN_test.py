"""network2.py
~~~~~~~~~~~~~~
An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.
"""

#### Libraries
# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np
from random import choice

#### Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.
        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        """
        return (a-y)


#### Main Network class
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # self.default_weight_initializer()
        self.large_weight_initializer()
        self.cost=cost

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.
        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.
        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda,
            evaluation_data,
            monitor_evaluation_cost=True,
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.
        """
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            # print "Epoch %s training complete" % j
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                # print "Cost on training data: {}".format(cost)
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                # print "Accuracy on training data: {} / {}".format(
                #     accuracy, n)
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                # print "Cost on evaluation data: {}".format(cost)
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                # print "Accuracy on evaluation data: {} / {}".format(
                #     self.accuracy(evaluation_data), n_data)
            # print
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
        #                 for w, nw in zip(self.weights, nabla_w)]

        tmp_weights = []
        for w,nw in zip(self.weights, nabla_w):
            for ww,nww in zip(w,nw):
                for i in range(len(ww)):
                    if ww[i] == 0:
                        nww[i] = 0
            tmp_weights.append((1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw)

        self.weights = tmp_weights
        # print self.weights


        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.
        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]



        # if raw_input("press any key to continue:"):
        #     pass

        # print x
        # print y
        # print results
        # for (x, y) in results:
        #     print int(x==y)
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.
    """
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


def Clustering(acts):
    centerdict = {}
    tmp = acts[0]
    D = 1
    epsilon = 0.6
    centerdict[D] = [tmp]
    acts = acts[1:]
    for a in acts:
        dist = 999
        for item in centerdict.items():
            key = item[0]
            points = item[1]
            center = sum(points)/1.0/len(points)
            if abs(center - a) < dist:
                dist = abs(center - a)
                tmpkey = key
        if dist <= epsilon:
            centerdict[tmpkey].append(a)
        else:
            D += 1
            centerdict[D] = [a]

    centerlist = []

    for item in centerdict.items():
            points = item[1]
            center = float(sum(points)/1.0/len(points))

            centerlist.append(center)

    return centerlist




num = 0

accuracylist = []


train_data = "data/features_numeric_tr"+str(num)+".txt"

test_data = "data/features_numeric_te"+str(num)+".txt"

eva_data = "data/features_numeric_test"+str(num)+"_binary.txt"


train_data = open(train_data)

test_data = open(test_data)

eva_data = open(eva_data)


train = []
test = []
eva = []

for line in train_data.readlines():
    # print line
    # linelist = line.split()
    # print linelist
    x_tmp = line.strip().split()[1:]
    x = []
    for i in x_tmp:
        x.append(int(i))

    x = np.array(x).reshape(len(x),1)

    y = line.strip().split()[0]

    #good
    if y == '0':
        y = np.array([1,0]).reshape(2,1)
    #bad
    elif y == '1':
        y = np.array([0,1]).reshape(2,1)

    # print x
    # print y
    data_list = (x,y)
    train.append(data_list)

for line in test_data.readlines():
    # print line
    # linelist = line.split()
    # print linelist
    x_tmp = line.strip().split()[1:]
    x = []
    for i in x_tmp:
        x.append(int(i))

    x = np.array(x).reshape(len(x),1)

    y = int(line.strip().split()[0])

    # if y == '0':
    #     y = np.array([1,0]).reshape(2,1)
    # elif y == '1':
    #     y = np.array([0,1]).reshape(2,1)
    data_list = (x,y)
    test.append(data_list)

for line in eva_data.readlines():
    # print line
    # linelist = line.split()
    # print linelist
    x_tmp = line.strip().split()[1:]
    x = []
    for i in x_tmp:
        x.append(int(i))

    x = np.array(x).reshape(len(x),1)

    y = int(line.strip().split()[0])

    # if y == '0':
    #     y = np.array([1,0]).reshape(2,1)
    # elif y == '1':
    #     y = np.array([0,1]).reshape(2,1)
    data_list = (x,y)
    eva.append(data_list)

ilen = len(train[0][0])
olen = len(train[0][1])
# print olen


# Num_Hidden = 3


# NN = Network([ilen,Num_Hidden,olen])
# print 'FINISHED!!!!!'


# epoch = 150
# batch_size = 100
# learning_rate = 0.1
# lmbda = 0.0


# e_cost,e_accuracy,t_cost,t_accuracy = NN.SGD(train,epoch,batch_size,learning_rate,lmbda,test)


# # accuracy = float(t_accuracy[-1])/len(train)
# accuracy = float(e_accuracy[-1])/len(test)

# print "ACCURACY!!!",accuracy


# max_accuracy = accuracy

# threshold = max_accuracy - 0.02

# while accuracy > threshold and accuracy >= 0.5:


#     weight1 = NN.weights[0]
#     weight2 = NN.weights[1]

#     count1 = 0
#     count2 = 0
#     eta1 = 0.1
#     eta2 = 0.01

#     for i in range(len(weight1)):
#         for j in range(len(weight1[i])):
#             w_list = []
#             for p in range(len(weight2)):
#                 w_list.append(abs(weight1[i][j]*weight2[p][i]))
#             if max(w_list) <= 4*eta2 and weight1[i][j] != 0:
#                 weight1[i][j] = 0
#                 # print "Pruned",i,j
#                 count1 += 1

#     # print weight1
#     print count1


#     for m in range(len(weight2)):
#         for n in range(len(weight2[m])):
#             if abs(weight2[m][n]) < 4*eta2 and weight2[m][n] != 0:
#                 weight2[m][n] = 0
#                 # print "Pruned",m,n
#                 count2 += 1

#     # print weight2
#     print count2

#     if count1 ==0 and count2 ==0:
#         omega_list = []
#         for i in range(len(weight1)):
#             for j in range(len(weight1[i])):
#                 w_list = []
#                 for p in range(len(weight2)):
#                     w_list.append(abs(weight1[i][j]*weight2[p][i]))
#                 omega_list.append(max(w_list))

#         for i in range(len(omega_list)):
#             if omega_list[i] == 0:
#                 omega_list[i] = 1

#         # print min(omega_list)
#         position = omega_list.index(min(omega_list))
#         row = position/len(weight1[0])
#         line = position%len(weight1[0])
        
#         # w_list = []
#         # for p in range(len(weight2)):
#         #     w_list.append(abs(weight1[row][line]*weight2[p][row]))
#         # if min(omega_list) == max(w_list):
#         #     print "YES!!!"

#         print weight1[row][line]

#         weight1[row][line] = 0

#         print "Pruned",row,line

#     e_cost,e_accuracy,t_cost,t_accuracy = NN.SGD(train,epoch,batch_size,learning_rate,lmbda,test)

#     # NN.save('result.txt')

#     # accuracy = float(t_accuracy[-1])/len(train)
#     accuracy = float(e_accuracy[-1])/len(test)

#     print "ACCURACY!!!",accuracy

#     if accuracy > max_accuracy:
#             max_accuracy = accuracy

#     threshold = max_accuracy - 0.02

# print "ONE ROUND FINISHED!!!!!!!"
# print '-------------------------'
# final_accuracy = float(NN.accuracy(eva))/len(eva)
# print "FINAL ACCURACY",final_accuracy
# accuracylist.append(final_accuracy)


# print accuracylist
# print "FINAL ACCURACY!!!!!",sum(accuracylist)/len(accuracylist)

# NN.save('result.txt')



NN = load('result.txt')


# actslist1 = []
# actslist2 = []

# for (x,y) in train:
#     # print "x",x


#     acts = []
    
#     for b, w in zip(NN.biases, NN.weights):
#             x = sigmoid(np.dot(w, x)+b)
#             acts.append(x)

#     acts1 = acts[0]
#     acts2 = acts[1]

#     # print "acts1",acts1
#     # print "acts2",acts2

#     actslist1.append(acts1)
#     actslist2.append(acts2)

# # print len(actslist1)
# # print len(actslist2)

# acts1 = []
# acts2 = []


# for i in range(len(actslist1[0])):

#     tmplist = []

#     for j in actslist1:
#         tmplist.append(j[i])

#     acts1.append(tmplist)

# for i in range(len(actslist2[0])):

#     tmplist = []

#     for j in actslist2:
#         tmplist.append(j[i])

#     acts2.append(tmplist)

# # print len(acts1)
# # print len(acts2)

# # print acts2[0]
# # print acts1[1]

# clusterlist1 = []

# for act in acts1:
#     clusterlist1.append(Clustering(act))
#     # print act

# # print len(clusterlist1)
# # print clusterlist1

# weight = NN.weights[1]
# bias = NN.biases[1]

# # print weight


# # print clusterlist1

# randomlist = [0,-1]

# hidden_list = []

# for j in range(1000):
#     hidden_values = []
#     for i in clusterlist1:
#         hidden_values.append(i[choice(randomlist)])


#     hidden_values = np.array(hidden_values).reshape(len(hidden_values),1)

#     hidden_list.append(hidden_values)

# final_result = []

# repeat_list = []

# for hidden_values in hidden_list:
#     outcome = []

#     for i in range(2):
#         outcome.append(sigmoid(np.dot(weight[i], hidden_values)+bias[i]))


#     result = []

#     for i in outcome:
#         result.append(i/sum(outcome))

#     if max(result) > 0.1:
#         if result not in repeat_list:
#             final_result.append([hidden_values,result])
#             repeat_list.append(result)
#             print [hidden_values,result]


# print final_result


for i in range(len(NN.weights[0][0])):
        if NN.weights[0][0][i] <= -0.1 and NN.weights[0][1][i] != 1000 and NN.weights[0][2][i] <= -0.1:
            print i+1







