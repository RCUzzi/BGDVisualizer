import os, sys, getopt, argparse
import numpy as np
from matplotlib import pyplot as plt


def main(argv):
    # initialize parser
    parser = argparse.ArgumentParser()

    #add optional arguments
    parser.add_argument('-d', '--datafile', dest='datafile', help = "Path to data file to be used", type=str, required=True)
    parser.add_argument('-a', '--alpha', dest='alpha', 
    help = 'Select a learning rate for your dataset (this should be approx .01, .03, .1, .3, 1, or 3). This defaults to .01', default = .01, type=float)
    parser.add_argument('-i', '--iterations', dest='iterations', help = 'Number of times to run gradient descent. This defaults to 100', default = 100, type=int)

    # read arguments from command line
    args = parser.parse_args()
    
    # initialize variables for ease of use
    datafile = args.datafile
    alpha = args.alpha
    iterations = args.iterations

    # check that the file exists
    if(os.path.isfile(datafile)):
        process_data(datafile, alpha, iterations)
    else:
        print('The file does not exist.')


def process_data(datafile, alpha, iterations):
    # load data from file
    data = np.loadtxt(os.path.join(datafile), delimiter=',')

    # slice csv into a set of parameters and y-values
    raw_features = data[:, :-1]
    values = data[:, -1]

    # normalize features for faster learning
    features_norm = normalize_matrix(raw_features)

    # add column of ones to X for accomodate for theta0 values
    features = np.concatenate([np.ones((values.size, 1)), features_norm], axis=1)
    theta = np.zeros(np.size(features, axis=1))

    # run (batch) gradient descent
    batch_gradient_descent(features, values, theta, alpha, iterations)
        

def cost_function(features, values, theta):
    m = values.shape[0]
    J = (1/(2*m))*np.sum((np.dot(features, theta)-values)**2)
    return J
    

def batch_gradient_descent(features, values, theta, alpha, iterations):
    m = values.shape[0]
    cost_history = []

    for i in range(iterations):
        theta = theta - (alpha / m)*(np.dot(features, theta)-values).dot(features)
        cost_history.append(cost_function(features, values, theta))
    
    print('THETA: ')
    print(theta)
    print('Cost History: ')
    print(cost_history)

def normalize_matrix(features):
    # initialize features copy and mu/sigma vectors
    features = features.copy()
    mu = np.zeros(features.shape[1])
    sigma = np.zeros(features.shape[1])

    # normalize the features matrix
    mu = np.mean(features, axis=0)
    sigma = np.std(features, axis=0)
    features = (features - mu) / sigma
    
    return features

def show_graphs():
    pass

def show_cost_graph():
    pass

def show_gradient_graph():
    pass

if __name__ == "__main__":
    main(sys.argv[1:])