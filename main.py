"""
Biometric for Mobile Authentication
Project - Task 3
Authors: Team Red - Matthew Kramer, Marcos Serrano, and Andrew Armstrong
Date: 11/25/2018

"""

import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import glob
import matplotlib.pyplot as plt

# PATH = "C:\\Users\\marco\\Documents\\School\\Biometrics\\Final"

global minlength
global trainingsamples
global validationsamples


def get_data_from_file(filename):
    """
    :param filename: The user's csv filename to be opened
    :return data: The data from the file split into time, x, y, and z dictionaries.

    Function parses each csv for that csv's specific person gait data
    """
    print("Reading data from file ", filename)
    data = dict()
    tdata = []
    xdata = []
    ydata = []
    zdata = []
    count = 0
    with open(filename) as f:
        for line in f:
            tdata.append(float(line.split(',')[0]))
            xdata.append(float(line.split(',')[1]))
            ydata.append(float(line.split(',')[2]))
            zdata.append(float(line.split(',')[3]))
            count += 1

    data['time'] = np.array(tdata)
    data['x'] = np.array(xdata)
    data['y'] = np.array(ydata)
    data['z'] = np.array(zdata)
    print("Initial Samples: ", count)
    return data


def get_cycles(data):
    """
    :param data: The user's gait data ndarray
    :return: cycles_list, cycles_dict , usersCycleData : All of which are different representations
                of the user's gait data.

    Function parses the data to output the user's cycles
    A Cycle is defined when the y value goes from negative to positive
    """
    global minlength
    minlength = 100
    weighted_y = moving_average(data['y'])
    times = data['time']
    data_set = 1
    cycles_list = []
    cycles_dict = {}
    index_start = 0
    first_cycle = True
    for idx, each in enumerate(weighted_y - 1):
        # If we detect more than 100 cycles, stop, we have enough for our testing, this speeds up each test run
        if data_set > 100:
            break
        # Detects a cycle when the y value changes from negative to positive
        if idx + 1 < len(weighted_y) and each < 0 < weighted_y[idx + 1] and index_start != idx:
            # Skip the first cycle found, since we removed 15% of data at beginning and end,
            # we could start in the middle of a cycle
            if first_cycle:
                first_cycle = False
                index_start = idx + 1
                continue
            # This forces the cycle to be a certain length
            if idx <= index_start + 6:
                continue
            cycles_list.append([times[index_start], times[idx]])
            cycles_dict[data_set] = [times[index_start], times[idx]]
            index_start = idx + 1
            data_set += 1
    print("Extracted {} cycles ".format(len(cycles_list)))

    usersCycleData = {}

    # Testing for handling different number of test data sets
    # This handles extracting the cycle values themselves, not just the times
    for indx, each in enumerate(cycles_list):
        trainingData = {"x": [], "y": [], "z": []}
        for idx, time in enumerate(data['time']):
            if each[0] <= time <= each[1]:
                add_row(data, trainingData, idx)
            if len(trainingData) > 99:
                break
            if time > each[1]:
                break
        x = trainingData['x'] - np.mean(trainingData['x'])
        y = trainingData['y'] - np.mean(trainingData['y'])
        z = trainingData['z'] - np.mean(trainingData['z'])
        magintude = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        if magintude is not None:
            usersCycleData[indx] = magintude
        maglength = len(magintude)
        if minlength > maglength and indx < 5:
            minlength = maglength

    # Returning both a dict and list, so you can choose which you prefer to interact with.
    # return cycles_list, cycles_dict, usersCycleData
    # Returns a dictionary containing the user's cycle data
    return usersCycleData


@DeprecationWarning
def get_row(data, index):
    return data['x'][index], data['y'][index], data['z'][index]


def add_row(_data, addData, index):
    """

    :param _data: The original data dictionary
    :param addData: The dictionary to add the data to
    :param index: The Index that add to the addData dictionary
    :return: NONE
    """
    addData['x'].append(_data['x'][index])
    addData['y'].append(_data['y'][index])
    addData['z'].append(_data['z'][index])
    return


def moving_average(column):
    """
     :param column: The colunm to be used for weighted moving average
     :return: weighted_data

     Function computes the moving weight average
     """
    # Remove start and end of signal to remove noise
    length = len(column)
    percent = int(length * 0.15)
    column = column[percent:-percent]

    y = column - np.mean(column)  # Subtract the mean
    weights = [0.75, 0.15, 0.1]
    weighted_data = np.zeros(len(y) - len(weights))

    # Perform weighted moving average
    for j in range(len(y) - len(weights)):
        for k in range(len(weights)):
            weighted_data[j] += (weights[k] * y[j + k])

    return weighted_data


def get_cycledata(usersData):
    trainingCycles = {}
    validationCycles = {}
    """
     :param usersData: The data split into user's cycles
     :return: trainingCycles: The cycles to be used for training (2 per user)
                validationCycles: The cycles to be used for training (2 per user)

     Function that splits the user's cycles into training and validation dictionaries
     """
    # These variables define how large the training and validation data sets are
    # These are one of our independant variables and we tested changing these numbers.
    # NOTE: The trainingsample
    global trainingsamples
    trainingsamples = 10
    global validationsamples
    validationsamples = 10
    if (trainingsamples + validationsamples > 20):
        raise ValueError('Trainingsamples + Validationsamples can\'t exceed 20 due to limited datasets. '
                         'trainingsamples={} validationsamples={} and combined they equal '
                         '{}!'.format(trainingsamples, validationsamples, trainingsamples + validationsamples))
    count = 0
    trainC = {}
    for userCycle in usersData:
        trainC[int(userCycle)] = []
        for cycle in range(trainingsamples):
            # Saves the data to be converted to a Dataframe, grab only the first 7 points, to remove NaN from dataframe
            if cycle <= trainingsamples - 1:
                trainingCycles[count] = (usersData[userCycle][cycle])[:7]
                trainC[userCycle].append((usersData[userCycle][cycle])[:7])
            count += 1
            if cycle > trainingsamples:
                break
    count = 0

    for userCycle in usersData:
        validationCycles[int(userCycle)] = []
        for cycle in range(validationsamples):
            # Saves the validation cycle data for use later
            if cycle <= validationsamples - 1:
                validationCycles[int(userCycle)].append((usersData[userCycle][cycle + trainingsamples])[:7])
            count += 1
            if cycle > validationsamples:
                break

    return trainingCycles, validationCycles, trainC


def plot_graph(graphable_data, graphtitle):
    """
     :param graphable_data: The data to be plotted
            graphtitle: The title to give the plot

     :return: result:

     A helper function to quickly plot data with a title
     """
    plt.figure()
    plt.plot(graphable_data)
    plt.title(graphtitle)
    plt.show()

    return


def get_pca_features(df):
    """
     :param trainingCycles: The 2 cycles for each user to be used for training the PCA feature extractor

     :return: pcaFeatures: The PCA feature are returned

     This funciton performs PCA on all the training data to extract the interesting features
     """
    print("Starting PCA process on all training data")

    # Standardize data for PCA
    x = StandardScaler().fit_transform(df)
    # print(x)

    # Testing doing PCA without specifying the number of components.
    pca = PCA()
    principalComponents = pca.fit_transform(x)
    col = []
    for column in range(len(pca.components_)):
        col.append('component {}'.format(column))
    pcaFeatures = pd.DataFrame(data=principalComponents, columns=col)
    # print("X components")
    # print(pcaFeatures)
    print("Finished PCA on all training data")

    return pcaFeatures


def dprime(gen_scores, imp_scores):
    x = np.sqrt(2) * abs(np.mean(gen_scores) - np.mean(imp_scores))
    y = np.sqrt(np.power(np.std(gen_scores),2) + np.power(np.std(imp_scores),2))
    return x / y


def plot_scoreDist(gen_scores, imp_scores):
    plt.figure()
    plt.hist(gen_scores, color='green', lw=2, histtype='step', hatch='//', label='Genuine Scores')
    plt.hist(imp_scores, color='red', lw=2, histtype='step', hatch='\\', label='Impostor Scores')
    plt.legend(loc='best')
    dp = dprime(gen_scores, imp_scores)
    plt.title('Score Distribution (d-prime= %.10f)' % dp)
    plt.show()
    return


def plot_det(far, frr, addtotitle):
    """

    :param far: False Accept Rate
    :param frr: False Reject Rate
    :param addtotitle: Title to be added to the graph
    :return:
    """
    # Compute eer
    far_minus_frr = 1
    eer = 0
    for i, j in zip(far, frr):
        if abs(i - j) < far_minus_frr:
            eer = i
            far_minus_frr = abs(i - j)

    plt.figure()
    plt.plot(far, frr, lw=2)
    plt.plot([0, 1], [0, 1], lw=1, color='black')
    plt.xlabel('false accept rate')
    plt.ylabel('false reject rate')
    plt.title(addtotitle + ' DET Curve (eer = %.10f)' % eer)
    plt.show()
    return


def plot_roc(far, tpr, addtotitle):
    """

    :param far: False Accept Rate
    :param tpr: True Positive Rate
    :param addtotitle: Title to be added to the graph's title
    :return:
    """
    plt.figure()
    plt.plot(far, tpr, lw=2)
    plt.xlabel('false accept rate')
    plt.ylabel('true accept rate')
    plt.title(addtotitle + ' ROC Curve')
    plt.show()
    return


def generate_curves(g_match, i_match, title):
    """
    Based homework #4 solution for generating the DET and ROC curves
    :param g_match: The Geninue matches in the system
    :param i_match: The Impostor matches in the system
    :param title: The title to be added to the ROC and DET curves
    :return:
    """
    far = []
    frr = []
    tpr = []
    thresholds = np.linspace(0, 1, 100)
    for t in thresholds:
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for g_s in g_match:
            if g_s <= t:
                tp += 1
            else:
                fn += 1
        for i_s in i_match:
            if i_s <= t:
                fp += 1
            else:
                tn += 1
        far.append(fp / (fp + tn))
        frr.append(fn / (fn + tp))
        tpr.append(tp / (tp + fn))
    plot_scoreDist(gen_match,imp_match)
    plot_roc(far, tpr, title)
    plot_det(far, frr, title)



if __name__ == '__main__':
    print("Program started!")
    # Read files and save user data to generate the system
    filenames = glob.glob("data/*.csv")
    # users = {}
    usersData = {}
    usercount = 0
    # Iterate through each file, extracting the user's gait data as cycles.
    for file in filenames:
        usercount += 1
        userid = int(file.lstrip('data\\').rstrip('.csv'))
        print("Starting processing user {}".format(userid))
        data = get_data_from_file(file)
        usersCycleData = get_cycles(data)

        # We need to save the user's data out after reading it for use later.
        usersData[userid] = usersCycleData
        # users[userid] = users_cycles_dict
        print("Finished processing user {}".format(userid))

    print("Finished reading all user data. {} total user files were read.".format(usercount))

    print("Start splitting cycles into training and validation sets.")
    trainingCycles, validationCycles, sortedTrainingCycles = get_cycledata(usersData)
    print("Finished spliting cycles {} training and {} validation cycles split".format(len(trainingCycles),
                                                                                       len(validationCycles)))
    print("Starting creating dataframe")
    dataframe = pd.DataFrame.from_dict(data=trainingCycles, orient='columns')
    print("Finished creating dataframe")

    # print(dataframe)
    print("Computing the mean cycle")
    meancycle = dataframe.mean(axis=1)

    print("Passing dataframe to PCA")
    # print(dataframe.mean())
    pcaFeatures = get_pca_features(dataframe)

    # Store the training sample set into user templates.
    print("Starting storing user templates based off PCA output")
    user_templates = {}
    count = 0
    for user in sortedTrainingCycles:
        user_templates[int(user)] = []
        for cycle in range(trainingsamples):
            template = sortedTrainingCycles[int(user)][cycle]

            template_coeffs = np.dot(pcaFeatures.transpose(), template)
            user_templates[int(user)].append(template_coeffs)
            count += 1
    print("{} users saved with {} cycles data".format(len(user_templates), count))
    print("Finished storing user templates based off PCA output")

    # Computes the match rate for the training data - MATCHER
    print("Starting matcher for training data")
    k = 7
    genuine_matches = 0
    imposter_match = 0
    dist = []
    gen_match = []
    imp_match = []
    results = {}
    for user in sortedTrainingCycles:
        results[int(user)] = {}
        for cycle in range(trainingsamples):
            results[int(user)][cycle] = []
            query = sortedTrainingCycles[int(user)][cycle]
            query = np.subtract(query, meancycle)
            query_coeffs = np.dot(pcaFeatures.transpose(), query)
            lowest_dist = 1000
            best_match = 0
            distances = []
            # The below for loops checks the query against all user templates and generates a Euclidean Distance score
            for cyc in range(trainingsamples):
                distances.append([distance.euclidean(user_templates[int(i + 1)][cyc], query_coeffs) for i in
                                  range(len(user_templates))])
            for idx, m in enumerate(distances):
                low = min(m)
                if low < lowest_dist:
                    lowest_dist = low
                    best_match = distances[idx].index(lowest_dist) + 1

            if int(user) == best_match:
                genuine_matches += 1
                # gen_match.append(lowest_dist)
                dist.append(lowest_dist)
            if int(user) != best_match:
                # imp_match.append(lowest_dist)
                imposter_match += 1
            results[int(user)][cycle].append(distances)

    # Compute the mean distance of the training set
    mean_distance = np.mean(dist)
    print("Mean distance: {}".format(mean_distance))

    # Compute the threshold to be considered a match
    # The multiplier here can be changed as an independent variable
    threshold = np.std(dist) * 1
    print("Proposed threshold: {} (standard deviation)".format(threshold))

    dist = dist - mean_distance
    # gen_match=np.subtract(gen_match,mean_distance)
    # imp_match=np.subtract(imp_match,mean_distance)
    match_std = 0
    for each in dist:
        if -threshold <= each < threshold:
            match_std += 1
    print("Number of matches within 1 std of mean distance: {}".format(match_std))

    print("Number of true matches: {}".format(genuine_matches))
    print("Number of non matches: {}".format(imposter_match))

    print("Finished matcher for training data")

    print("Starting decision for accept or fail of training data")
    # Decision
    for idx, user in enumerate(results):
        for cycle in range(trainingsamples):
            for index, matchscores in enumerate(results[int(user)][cycle][0]):
                for indx, m in enumerate(matchscores):
                    m = m - mean_distance
                    if -threshold <= m <= threshold:
                        if int(user) == indx + 1:
                            gen_match.append(m)
                        else:
                            imp_match.append(m)
    print("Finished decision for accept or fail of training data")

    print("Start drawing curves for training data")
    # Plot Curves
    generate_curves(gen_match, imp_match, "Training Data")
    print("Finished drawing curves for training data")

    # Computes the match rate for the Validation data - MATCHER
    print("Starting matcher for validation data")
    k = 7
    genuine_matches = 0
    imposter_match = 0
    dist = []
    gen_match = []
    imp_match = []
    results = {}
    for user in validationCycles:
        results[int(user)] = {}
        for cycle in range(validationsamples):
            results[int(user)][cycle] = []
            query = validationCycles[int(user)][cycle]
            query = np.subtract(query, meancycle)
            query_coeffs = np.dot(pcaFeatures.transpose(), query)
            lowest_dist = 1000
            best_match = 0
            distances = []
            # The below for loops checks the query against all user templates and generates a Euclidean Distance score
            for cyc in range(trainingsamples):
                distances.append([distance.euclidean(user_templates[int(i + 1)][cyc], query_coeffs) for i in
                                  range(len(user_templates))])
            for idx, m in enumerate(distances):
                low = min(m)
                if low < lowest_dist:
                    lowest_dist = low
                    best_match = distances[idx].index(lowest_dist) + 1
            # print("True label = {}; gait matches with {}, when k = {} with distance {}".format(user,best_match,k, lowest_dist))

            if int(user) == best_match:
                genuine_matches += 1
            if int(user) != best_match:
                imposter_match += 1
            results[int(user)][cycle].append(distances)

    print("Number of true matches: {}".format(genuine_matches))
    print("Number of non matches: {}".format(imposter_match))

    print("Finished matcher for validation data")
    # Decision
    print("Starting decision for accept or fail of validation data")
    for idx, user in enumerate(results):
        for cycle in range(validationsamples):
            for index, matchscores in enumerate(results[int(user)][cycle][0]):
                for indx, m in enumerate(matchscores):
                    m = m - mean_distance
                    if -threshold <= m <= threshold:
                        if int(user) == indx + 1:
                            gen_match.append(m)
                        else:
                            imp_match.append(m)
    print("Finished decision for accept or fail of validation data")

    print("Start drawing curves for validation data")
    generate_curves(gen_match, imp_match, "Validation Data")
    print("Finished drawing curves for validation data")

    print("Program finished!")
