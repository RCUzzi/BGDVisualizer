import os, sys, getopt, argparse
import numpy as np
from matplotlib import pyplot as plt


def main(argv):
    # initialize parser
    parser = argparse.ArgumentParser()

    #add optional arguments
    parser.add_argument('-d', '--datafile', dest = 'datafile', help = "Path to data file to be used", type = str, required = True)
    parser.add_argument('-a', '--alpha', dest = 'alpha', 
    help = 'Select a learning rate for your dataset (this should be approx .01, .03, .1, .3, 1, or 3). This defaults to .01', default = .01, type = float)
    parser.add_argument('-i', '--iterations', dest = 'iterations', help = 'Number of times to run gradient descent. This defaults to 100', default = 100, type = int)
    parser.add_argument('-w', '--wait', help = "Keep graph open after animation until manually closed", action = 'store_true')
    parser.add_argument('-s', '--show', help = 'Show graph animation during calculations (will not display graph if more than 1 feature exists)', action = 'store_true')
    parser.add_argument('-c', '--cost', help = 'Output history of cost values to a file', action = 'store_true')

    # read arguments from command line
    args = parser.parse_args()
    
    # initialize variables for ease of use
    datafile = args.datafile
    alpha = args.alpha
    iterations = args.iterations

    # check that the file exists
    if(os.path.isfile(datafile)):
        process_data(datafile, alpha, iterations, args)
    else:
        print('The file does not exist.')


def process_data(datafile, alpha, iterations, args):
    # load data from file
    data = np.loadtxt(os.path.join(datafile), delimiter = ',')

    # slice csv into a set of parameters and y-values
    raw_features = data[:, :-1]
    values = data[:, -1]

    # normalize features for faster learning
    features_norm = normalize_matrix(raw_features)

    # add column of ones to X for accomodate for theta0 values
    features = np.concatenate([np.ones((values.size, 1)), features_norm], axis = 1)
    theta = np.zeros(np.size(features, axis = 1))
    # run (batch) gradient descent
    batch_gradient_descent(features, values, theta, alpha, iterations, show=args.show, show_cost=args.cost, wait=args.wait)
        

def cost_function(features, values, theta):
    m = values.shape[0]
    cost = (1/(2*m))*np.sum((np.dot(features, theta)-values)**2)
    return cost
    

def batch_gradient_descent(features, values, theta, alpha, iterations, show=False, show_cost=False, wait=False):
    m = values.shape[0]
    cost_history = []
    if(show == True):
        figure = plt.figure()

    for i in range(iterations):
        theta = theta - (alpha / m)*(np.dot(features, theta)-values).dot(features)
        cost_history.append(cost_function(features, values, theta))
        if((show == True or wait == True) and features[0].size == 2):
            plt.plot(features[:, 1:], values, 'ro', ms=10, mec='k')
            plt.ylabel('Values')
            plt.xlabel('Features')
            plt.plot(features[:, 1:], np.dot(features, theta), '-')
            plt.pause(0.00001)
            if(i != iterations - 1):
                plt.cla()

    if(wait == True):
        plt.show()
    else:
        plt.show(block=False) 
         
    if(show_cost == True):
        print('Cost History: ')
        print(cost_history)

    return cost_history

def normalize_matrix(features):
    # initialize features copy and mu/sigma vectors
    features = features.copy()
    mu = np.zeros(features.shape[1])
    sigma = np.zeros(features.shape[1])

    # normalize the features matrix
    mu = np.mean(features, axis = 0)
    sigma = np.std(features, axis = 0)
    features = (features - mu) / sigma
    
    return features

def show_graphs():
    pass

def show_cost_graph(features, values, theta):
    pass
    
def show_gradient_graph():
    pass

if __name__ == "__main__":
    main(sys.argv[1:])