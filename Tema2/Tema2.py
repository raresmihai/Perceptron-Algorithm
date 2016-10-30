import cPickle, gzip, numpy
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()



def activation(input):
    if input > 0:
        return 1
    else:
        return 0

def learn(train_set,cifra):
    learning_rate = 0.1
    nr_iter = 1
    all_classified = False
    bias = 0
    w = numpy.random.uniform(0,1,784)
    while not all_classified and nr_iter >= 0:
        all_classified = True
        length = len(train_set[0])
        for i in range(0,50000):
            x = train_set[0][i]
            t = train_set[1][i]
            if t == cifra:
                t = 1
            else:
                t = 0
            z = numpy.add(numpy.dot(w,x),bias)
            output = activation(z)
            act_x = numpy.dot(x,(t-output))
            learn_x = numpy.dot(act_x,learning_rate)
            w = numpy.add(w,learn_x)
            bias = bias + (t-output) * learning_rate
            if output != t:
                all_classified = False
        nr_iter -= 1
    return (w,bias)




def set_perceptroni():
    perceptroni = []
    for cifra in range(0,10):
        perceptron_cifra = learn(train_set,cifra)
        perceptroni += [perceptron_cifra]
    return perceptroni



def test(test_set):
    corecte = [0,0,0,0,0,0,0,0,0,0]
    total = [0,0,0,0,0,0,0,0,0,0]
    total_all = 0
    perceptroni = set_perceptroni()
    for i in range(0,len(test_set[0])):
        x = test_set[0][i]
        t = test_set[1][i]
        perceptron = perceptroni[t]
        w = perceptron[0]
        bias = perceptron[1]
        z = numpy.add(numpy.dot(w,x),bias)
        output = activation(z)
        if output == 1:
            corecte[t] += 1
            total_all += 1
        total[t] += 1
    f = open("statistics.txt","w")
    for i in range(0,len(corecte)):
        f.write(str(i) + ': ' + str(corecte[i]/float(total[i])*100) + '%\n')
    f.write('\nTotal corecte: ' + str(total_all/float(len(test_set[0]))*100)+ '%\n')

test(test_set)
