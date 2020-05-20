import numpy as np
import operator
import math

dataset_data = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[1, 2, 3, 4, 5, 6, 7],
                             converters={5: lambda s: 0 if s == b"-1" else float(s),
                                         7: lambda s: 0 if s == b"-1" else float(s)})
validation_data = np.genfromtxt('validation1.csv', delimiter=';', usecols=[1, 2, 3, 4, 5, 6, 7],
                                converters={5: lambda s: 0 if s == b"-1" else float(s),
                                            7: lambda s: 0 if s == b"-1" else float(s)})
input_data = np.genfromtxt('days.csv', delimiter=';', usecols=[1, 2, 3, 4, 5, 6, 7],
                           converters={5: lambda s: 0 if s == b"-1" else float(s),
                                       7: lambda s: 0 if s == b"-1" else float(s)})

dataset_dates = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0])
dataset_labels = []
for label in dataset_dates:
    if label < 20000301:
        dataset_labels.append('winter')
    elif 20000301 <= label < 20000601:
        dataset_labels.append('spring')
    elif 20000601 <= label < 20000901:
        dataset_labels.append('summer')
    elif 20000901 <= label < 20001201:
        dataset_labels.append('fall')
    else:   # from 01-12 to end of year
        dataset_labels.append('winter')

validation_dates = np.genfromtxt('validation1.csv', delimiter=';', usecols=[0])
validation_labels = []
for label in validation_dates:
    if label < 20010301:
        validation_labels.append('winter')
    elif 20010301 <= label < 20010601:
        validation_labels.append('spring')
    elif 20010601 <= label < 20010901:
        validation_labels.append('summer')
    elif 20010901 <= label < 20011201:
        validation_labels.append('fall')
    else:   # from 01-12 to end of year
        validation_labels.append('winter')


def euclidean_distance(x, y, length):
    distance = 0
    for i in range(length):
        distance += pow((x[i] - y[i]), 2)
    return math.sqrt(distance)


def get_neighbours(dataset, data_instance, k_value):
    distances = []
    length = len(data_instance) - 1
    for iterator in range(len(dataset)):
        d = euclidean_distance(data_instance, dataset[iterator], length)
        distances.append((dataset_labels[iterator], d))
    distances.sort(key=operator.itemgetter(1))
    neighbours = []
    for iterator in range(k_value):
        neighbours.append(distances[iterator][0])
    return neighbours


def get_most_frequent(data):
    return max(set(data), key=data.count)


def get_k(dataset, validationset):
    k_value = 0
    temp = 0
    for k_value in range(120):
        counter = 0
        for iterator in range(len(validationset)):
            if get_most_frequent(get_neighbours(dataset, validationset[iterator], k_value + 1)
                                 ) == validation_labels[iterator]:
                counter += 1
        if counter > temp:
            k_value = k_value+1
            temp = counter
    print("Estimated accuracy is ", temp, "%")
    return k_value


def get_season(dataset, data_instance, k_value):
    return get_most_frequent(get_neighbours(dataset, data_instance, k_value))


k = get_k(dataset_data, validation_data)
print("K = ", k)
for i in range(len(input_data)):
    print("Day ", i, " = ", get_season(dataset_data, input_data[i], k), "     with input ", input_data[i])