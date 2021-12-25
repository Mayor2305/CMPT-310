# abonus.py

# template for Bonus Assignment, Artificial Intelligence Survey, CMPT 310 D200
# Spring 2021, Simon Fraser University

# author: Jens Classen (jclassen@sfu.ca)

from learning import *

def generate_restaurant_dataset(size=100):
    """
    Generate a data set for the restaurant scenario, using a numerical
    representation that can be used for neural networks. Examples will
    be newly created at random from the "real" restaurant decision
    tree.
    :param size: number of examples to be included
    """

    numeric_examples = None

    ### YOUR CODE HERE ###

    lists = SyntheticRestaurant(size)
    numeric_examples = []

    for i in range(size):
        lis = []
        for j in range(len(lists.examples[i])):
            if lists.examples[i][j] == "Yes" or lists.examples[i][j] == "Some" or lists.examples[i][j] == "$$" \
                    or lists.examples[i][j] == "10-30":
                lis.append(1)
            elif lists.examples[i][j] == "No" or lists.examples[i][j] == "None" or lists.examples[i][j] == "$" \
                    or lists.examples[i][j] == "0-10":
                lis.append(0)
            elif lists.examples[i][j] == "Full" or lists.examples[i][j] == "$$$" or lists.examples[i][j] == "30-60":
                lis.append(2)
            elif lists.examples[i][j] == ">60":
                lis.append(3)
            elif lists.examples[i][j] == "Burger":
                lis.append(1)
                lis.append(0)
                lis.append(0)
                lis.append(0)
            elif lists.examples[i][j] == "French":
                lis.append(0)
                lis.append(1)
                lis.append(0)
                lis.append(0)
            elif lists.examples[i][j] == "Italian":
                lis.append(0)
                lis.append(0)
                lis.append(1)
                lis.append(0)
            elif lists.examples[i][j] == "Thai":
                lis.append(0)
                lis.append(0)
                lis.append(0)
                lis.append(1)
        numeric_examples.append(lis)

    return DataSet(name='restaurant_numeric',
                   target='Wait',
                   examples=numeric_examples,
                   attr_names='Alternate Bar Fri/Sat Hungry Patrons Price Raining Reservation Burger French Italian Thai WaitEstimate Wait')

def nn_cross_validation(dataset, hidden_units, epochs=100, k=10):
    """
    Perform k-fold cross-validation. In each round, train a
    feed-forward neural network with one hidden layer. Returns the
    error ratio averaged over all rounds.
    :param dataset:      the data set to be used
    :param hidden_units: the number of hidden units (one layer) of the neural nets to be created
    :param epochs:       the maximal number of epochs to be performed in a single round of training
    :param k:            k-parameter for cross-validation
                         (do k many rounds, use a different 1/k of data for testing in each round)
    """

    error = 0

    ### YOUR CODE HERE ###
    fold_errt = 0
    fold_errv = 0
    n = len(dataset.examples)
    examples = dataset.examples
    random.shuffle(dataset.examples)
    for fold in range(k):
        train_data, val_data = train_test_split(dataset, fold * (n // k), (fold + 1) * (n // k))
        dataset.examples = train_data
        h = NeuralNetLearner(dataset, [hidden_units])
        fold_errt += err_ratio(h, dataset, train_data)
        fold_errv += err_ratio(h, dataset, val_data)
        dataset.examples = examples

    error = (fold_errt / k + fold_errv / k) / 2

    return error


N          = 100   # number of examples to be used in experiments
k          =   5   # k parameter
epochs     = 100   # maximal number of epochs to be used in each training round
size_limit =  15   # maximal number of hidden units to be considered

# generate a new, random data set
# use the same data set for all following experiments
dataset = generate_restaurant_dataset(N)

# try out possible numbers of hidden units
for hidden_units in range(1,size_limit+1):
    # do cross-validation
    error = nn_cross_validation(dataset=dataset,
                                hidden_units=hidden_units,
                                epochs=epochs,
                                k=k)
    # report size and error ratio
    print("Size " + str(hidden_units) + ":", error)
