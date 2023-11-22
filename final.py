import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot as plt

# IMPORT FILES
dataset = pd.read_csv("responses.csv")

# drop only child column - redundant with number of siblings
dataset.drop(dataset.columns[147], axis = 1, inplace = True)

# --------------------------------------------------------------------------------------
# -----------PART 1: INITIAL DEMOGRAPHIC BREAKDOWN -------------------------------------
# --------------------------------------------------------------------------------------
# (Connor, Sara Ann)

# Age Breakdown
age_breakdown = dataset['Age'].value_counts()
# print("Age Summary Statistics: " + age_breakdown.to_string())

# Height Breakdown
height_buckets = pd.cut(dataset['Height'], bins=range(
    145, 210, 5), include_lowest=True)
height_bucket_counts = height_buckets.value_counts(sort=False)
# print("Height Bucket Counts:" + height_bucket_counts.to_string())

# Gender Breakdown
gender_breakdown = dataset['Gender'].value_counts()
# print("Gender Summary Statistics:" + gender_breakdown.to_string())

# Weight
weight_buckets = pd.cut(dataset['Weight'], bins=range(
    40, 170, 5), include_lowest=True)
weight_bucket_counts = weight_buckets.value_counts(sort=False)
# print("Weight Bucket Counts:" + weight_bucket_counts.to_string())

# Number of siblings
sibs_breakdown = dataset['Number of siblings'].value_counts()
# print("Sibling # Value Counts:" + sibs_breakdown.to_string())

# Left - right handed
hand_breakdown = dataset['Left - right handed'].value_counts()
# print("Left or Right Handed Value Counts:" + hand_breakdown.to_string())

# Education
ed_breakdown = dataset['Education'].value_counts()
# print("Education Value Counts:" + ed_breakdown.to_string())

# Only child
#onlychild_breakdown = dataset['Only child'].value_counts()
# print("Only Child Value Counts:" + onlychild_breakdown.to_string())

# Village - town
rural_breakdown = dataset['Village - town'].value_counts()
# print("Village or Town Value Counts:" + rural_breakdown.to_string())

# House - block of flats'
house_breakdown = dataset['House - block of flats'].value_counts()
# print("House or Block of Flats Value Counts:" + house_breakdown.to_string())

# Mapping of categorical columns to 1-5 number scales
smoking_mapping = {'never smoked': 1, 'tried smoking': 2,
                   'former smoker': 4, 'current smoker': 5}

alcohol_mapping = {'never': 1, 'social drinker': 3, 'drink a lot': 5}

punctuality_mapping = {'i am always on time': 1,
                       'i am often early': 3, 'i am often running late': 5}

lying_mapping = {'never': 1, 'only to avoid hurting someone': 3,
                 'sometimes': 4, 'everytime it suits me': 5}

internet_mapping = {'less than an hour a day': 1,
                    'few hours a day': 3, 'most of the day': 5}

gender_mapping = {'female': 0, 'male': 1}

handed_mapping = {'left handed': 0, 'right handed': 1}

education_mapping = {'primary school': 1, 'secondary school': 2,
                     'college/bachelor degree': 4, 'masters degree': 5}

onlychild_mapping = {'no': 0, 'yes': 1}

city_mapping = {'village': 0, 'city': 1}

house_mapping = {'block of flats': 0, 'house/bungalow': 1}

all_mapping = [smoking_mapping, alcohol_mapping, punctuality_mapping,
               lying_mapping, internet_mapping, gender_mapping, handed_mapping,
               education_mapping, onlychild_mapping, city_mapping, house_mapping]

# --------------------------------------------------------------------------------------
# -----------PART 2: K-MEANS CLUSTERING -------------------------------------
# --------------------------------------------------------------------------------------

# ACTUAL K-MEANS METHOD (Sara Ann, Connor, Emily)
# data = dataset, k = # of centroids, iterations is maximum (won't reach if converge)
def lloyds_kmeans(data, k, iterations, ignoreindexes):
    #evil way to do this but nevertheless:
    collectcentroiddiff = []
    # randomly initialize centroids
    centroids = data.sample(n=k).values
    
    # initialize clusters
    clusters = [[] for _ in range(k)]  # NEW CODE

    for i in range(iterations):       
        print("Starting iteration " + str(i))
        print("Total distance before cluster reassignment: ",
              sumdistances(centroids, clusters, ignoreindexes))
        
        # calculate distance from each point to all centroids
        distances = [[0]*len(data) for x in range(k)]
        for centroidindex in range(k):
            # print("Centroid index: " + str(centroidindex))
            # NEW CODE, centroidindex was i before
            currentcentroid = centroids[centroidindex]
            for rowindex in range(len(data)):
                # print("Row index: " + str(rowindex))
                currentpoint = data.iloc[rowindex]
                # print("CURRENT CENTROID ", currentcentroid)
                # print("TYPE OF CURRENT CENTROID ", type(currentcentroid))
                distances[centroidindex][rowindex] = distance(currentcentroid, currentpoint, ignoreindexes)

        # assign point to correct cluster
        clusters = [[] for _ in range(k)]
        for rowindex in range(len(data)):
            tempmin = 0
            for centroidindex in range(k):
                if distances[centroidindex][rowindex] < distances[tempmin][rowindex]:
                    tempmin = centroidindex
            clusters[tempmin].append(data.iloc[rowindex])

        print("Total distance before new centroids: ",
              sumdistances(centroids, clusters, ignoreindexes))
        
        """ count = 0
        for c in range(len(clusters)):
            if len(clusters[c]) != 0 and len(centroids[c]) == 0:
                print("Major error") """


        clusterDistance(centroids, clusters, ignoreindexes)

        # new centroids based on mean of each cluster -- CONNOR
        centroidsnew = [[] for _ in range(k)]
        for c in range(len(clusters)):
            gravity = meandata(clusters[c])
            centroidsnew[c] = gravity
        #centroidsnew = [meandata(cluster) for cluster in clusters]
        #print(centroidsnew)

        clusterDistance(centroidsnew, clusters, ignoreindexes)

        # REPORT TOTAL DISTANCE OF ALL CLUSTERS
        print("Total distance: ", sumdistances(centroidsnew, clusters, ignoreindexes))

        diff = centroidDifference(centroidsnew, centroids, ignoreindexes)
        collectcentroiddiff.append(diff)
        print("Centroid difference: " + str(diff))
        
        # CONVERGENCE CHECKS
        currentindex = len(collectcentroiddiff)-1
        if np.array_equal(centroids, centroidsnew):  
            print("Exact converge after " + str(i) + " iterations")
            break
        elif collectcentroiddiff[currentindex] > collectcentroiddiff[currentindex-1]:
            print("Approx converge after " + str(i) + " iterations")
            break
        else:
            centroids = centroidsnew

    # calculate distance from each point to final centroids
    distances = [[0]*len(data) for x in range(k)]
    for centroidindex in range(k):
        # currentcentroid = centroids[i] #Where is i coming from, this is outside of for loop
        currentcentroid = centroids[centroidindex]
        for rowindex in range(len(data)):
            currentpoint = data.iloc[rowindex]
            distances[centroidindex][rowindex] = distance(
                currentcentroid, currentpoint, ignoreindexes)
            
    clusters = [[] for _ in range(k)]
    # assign point to final cluster
    for rowindex in range(len(data)):
        tempmin = 0
        for centroidindex in range(k):
            if distances[centroidindex][rowindex] < distances[tempmin][rowindex]:
                tempmin = centroidindex
        clusters[tempmin].append(data.iloc[rowindex])
    
    print("Final distance: ", sumdistances(centroidsnew, clusters, ignoreindexes))

    return centroids, clusters

# VARIOUS DISTANCE METHODS FOR DEBUGGING:
def centroidDifference(centroids1, centroids2, ignoreindexes):
    sum = 0
    for cent in range(len(centroids1)):
        sum += distance(centroids1[cent], centroids2[cent], ignoreindexes)
    return sum

def sumdistances(centroids, clusters, ignoreindexes):
    tempsum = 0
    for centerindex in range(len(centroids)):
        for pointindex in range(len(clusters[centerindex])):
            tempsum += distance(centroids[centerindex],
                                clusters[centerindex][pointindex], ignoreindexes)
    return tempsum

def clusterDistance(centroids, clusters, ignoreindexes):
    for centerindex in range(len(centroids)):
        tempsum = 0
        for pointindex in range(len(clusters[centerindex])):
            tempsum += distance(centroids[centerindex],
                                clusters[centerindex][pointindex], ignoreindexes)
        #print("Cluster: " + str(centerindex) + ", Distance: " + str(tempsum))


# MEAN OF A COLLECTION OF ROWS (Connor, Sara Ann, Emily)
def meandata(data):
    count = 0
    means_and_categorical_modes = []
    rows = len(data)
    cols = len(data[0]) if data else 0  # NEW CODE
    for col in range(cols):
        sum = 0
        frequencies = {}
        for row in range(rows):
            # The case for when the data is an integer (139 of the 150 columns)
            if not isinstance(data[row][col], str):
                if not math.isnan(data[row][col]):
                    sum += data[row][col]
            # The case for when the data is categorical (11 of the 150 columns)
            elif isinstance(data[row][col], str):
                if data[row][col] in frequencies:
                    frequencies[data[row][col]] += 1
                else:
                    frequencies[data[row][col]] = 1

        # The case for when the data is an integer (139 of the 150 columns)
        if sum != 0:
            # print("Sum: ", sum, "and Rows: ", rows)
            means_and_categorical_modes.append(
                round(float(sum) / float(rows)))
        
        # The case for when the data is categorical (11 of the 150 columns)
        else:
            count += 1
            categoricalSum = 0
            m = {}

            for identifier in frequencies:
                for mapping in all_mapping:
                    if identifier in mapping:
                        m = mapping
                        categoricalSum = categoricalSum + mapping[identifier]*frequencies[identifier]
            categoricalAvg = float(categoricalSum)/float(rows)

            temporary = 10000
            smallestId = ""
            for id in m:
                number = m[id]
                diff = abs(float(number) - categoricalAvg)
                if diff < temporary:
                    temporary = number
                    smallestId = id
            means_and_categorical_modes.append(smallestId)

    return means_and_categorical_modes

# CALCULATE DISTANCE BETWEEN TWO ROWS (Emily, Sara Ann)
def distance(user1, user2, ignoreindexes):
    distance = 0

    for x in range(len(user1)):
        if x not in ignoreindexes:
            temp = 0
            user1val = user1[x]
            user2val = user2[x]
            
            if not isinstance(user1val, str) and not isinstance(user2val, str):
                if not math.isnan(user1val) and not math.isnan(user2val):
                    tempval = user1val - user2val
                    tempval = abs(tempval)
                    temp += tempval
            
            elif isinstance(user1val, str) and isinstance(user2val, str):
                for map in all_mapping:
                    if user1val in map and user2val in map:
                        temp += abs(map[user1val] - map[user2val])
                        break
            temp = temp**2

            distance += temp
    return distance


# --------------------------------------------------------------------------------------
# ----------- PART 3: TESTING & ANALYSIS -------------------------------------
# --------------------------------------------------------------------------------------

# SUMMARY STATISTICS (Connor)
def summaryStats(cluster):
    #Needed to hardcode columns for different demographics. i.e. 140 is age

    # Age Breakdown
    age_column = pd.Series([row[140] for row in cluster])
    age_breakdown = age_column.value_counts()
    #print("Age Summary Statistics:\n" + age_breakdown.to_string())

    # Height Breakdown
    height_column = pd.Series([row[141] for row in cluster])
    height_buckets = pd.cut(height_column, bins=range(
        145, 210, 5), include_lowest=True)
    height_bucket_counts = height_buckets.value_counts()
    #print("Height Bucket Counts:\n" + height_bucket_counts.to_string())

    # Gender Breakdown
    gender_column = pd.Series([row[144] for row in cluster])
    gender_breakdown = gender_column.value_counts()
    print("Gender Summary Statistics:\n" + gender_breakdown.to_string())

    # Weight
    weight_column = pd.Series([row[142] for row in cluster])
    weight_buckets = pd.cut(weight_column, bins=range(
        40, 170, 5), include_lowest=True)
    weight_bucket_counts = weight_buckets.value_counts()
    #print("Weight Bucket Counts:\n" + weight_bucket_counts.to_string())

    # Number of siblings
    sibs_column = pd.Series([row[143] for row in cluster])
    sibs_breakdown = sibs_column.value_counts()
    #print("Sibling # Value Counts:\n" + sibs_breakdown.to_string())

    # Left - right handed
    hand_column = pd.Series([row[145] for row in cluster])
    hand_breakdown = hand_column.value_counts()
    #print("Left or Right Handed Value Counts:\n" + hand_breakdown.to_string())

    # Education
    ed_column = pd.Series([row[146] for row in cluster])
    ed_breakdown = ed_column.value_counts()
    #print("Education Value Counts:\n" + ed_breakdown.to_string())

    # Village - town
    rural_column = pd.Series([row[147] for row in cluster])
    rural_breakdown = rural_column.value_counts()
    #print("Village or Town Value Counts:\n" + rural_breakdown.to_string())

    # House - block of flats'
    house_column = pd.Series([row[148] for row in cluster])
    house_breakdown = house_column.value_counts()
    #print("House or Block of Flats Value Counts:\n" + house_breakdown.to_string())

def allSummaryStats(clusters):
    for cluster in clusters:
        #print(len(cluster))
        summaryStats(cluster)

# TESTING ZONE! (Sara Ann)
testdata = dataset.sample(n=100)
# demographic indexes = [140, 141, 142, 143, 144, 145, 146, 147, 148]
print("WITH DEMOGRAPHICS? ")
results1 = lloyds_kmeans(testdata, 4, 50, [])
print("\nWITHOUT DEMOGRAPHICS? ")
results2 = lloyds_kmeans(testdata, 4, 50, [140, 141, 142, 143, 144, 145, 146, 147, 148])
print("\nSUMMARY STATS: WITH DEMOGRAPHICS")
allSummaryStats(results1[1])
print("\nSUMMARY STATS: WITHOUT DEMOGRAPHICS")
allSummaryStats(results2[1])

testdata2 = dataset.sample(n=100)
# demographic indexes = [144] (just removing gender)
print("WITH GENDER? ")
results3 = lloyds_kmeans(testdata, 4, 50, [])
print("\nWITHOUT GENDER? ")
results4 = lloyds_kmeans(testdata, 4, 50, [144])
print("\nSUMMARY STATS: WITH GENDER")
allSummaryStats(results3[1])
print("\nSUMMARY STATS: WITHOUT GENDER")
allSummaryStats(results4[1])
