# This file select input nodes from nodes
# It also calculates the distance between nodes

import pandas as pd
import math
import random
import Parameters as para

random.seed(258)

dataframe = pd.read_csv('nodes.csv')

Set1 = [4191, 3868, 3704, 3864, 3712, 3895, 3940, 3915, 3784, 3983, 3952, 4944, 4989, 4917, 4975, 4906, 4959,
                5040, 5053, 5019, 5036, 5070, 4185, 4107, 4035, 4054, 4012, 4322, 4470, 4443, 4572, 4121, 4143, 4818,
                4823, 5078, 5784, 5790, 4501, 4532, 4657, 4269, 4624, 4722, 4758, 5889, 5721, 5892, 5772, 5740, 4641,
                5841, 5859, 5850, 5984, 5920, 5914, 5907, 6835, 6021, 6038, 6014, 5020, 5807, 5945, 5976, 4272, 4189,
                4538, 4989, 4323]

LargeSet = dataframe.loc[dataframe['ID'].isin(Set1)]
LargeSet.to_csv('LargeSet.csv', index=False)

Set2 = [4191, 4943, 4991, 4975, 4181, 4161, 5046, 5068, 4105, 3947, 3952, 4036, 4119,
                4254, 4057, 4001, 3979, 4447, 4313]

SmallSet = dataframe.loc[dataframe['ID'].isin(Set2)]
SmallSet.to_csv('SmallSet.csv', index=False)


# Generate the demand for each node
def Gen_demand(Totaldemand, num_node):
    demand_location = [0]
    for i in range(num_node - 1):
        demandTemp = int(Totaldemand/(num_node - 1)) + random.randint(-3, 4)
        demand_location.append(demandTemp)

    return demand_location


smallAreaHighDemand = Gen_demand(para.HighDemand, len(Set2))
largeAreaHighDemand = Gen_demand(para.HighDemand, len(Set1))
largeArea2kDemand = Gen_demand(para.Demand2K, len(Set1))
largeArea1kDemand = Gen_demand(para.Demand1K, len(Set1))


# Calculate the distance matrix
# n is the list of nodes(id)
# demand_node is the list of demand of each node
def Gen_distance_time_Matrix(n, dataFrame, demand_node):
    arr_distance = [[0 for col in range(len(n))] for row in range(len(n))]
    arr_time = [[0 for col in range(len(n))] for row in range(len(n))]
    for i in range(len(n)):
        for j in range(len(n)):
            if i != j:
                x1 = dataFrame.loc[dataframe['ID'] == n[i], 'X'].iloc[0]
                y1 = dataFrame.loc[dataframe['ID'] == n[i], 'Y'].iloc[0]
                x2 = dataFrame.loc[dataframe['ID'] == n[j], 'X'].iloc[0]
                y2 = dataFrame.loc[dataframe['ID'] == n[j], 'Y'].iloc[0]
                # 1.5 is the direct distance amplify index
                # 0.62 is the km to mile conversion
                travel_distance = 2 * 0.621371192/1000 * math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                travel_time_between_nodes = travel_distance * 60 / para.vehicle_speed_between_nodes
                drop_distance = demand_node[i] * para.distance_between_drops
                travel_time_within_node = drop_distance * 60 / para.vehicle_speed_between_drops
                service_time = demand_node[i] * para.service_time_per_drop
                arr_distance[i][j] = round(travel_distance + drop_distance)
                arr_time[i][j] = round(travel_time_between_nodes + travel_time_within_node + service_time)

    return arr_distance, arr_time


# The following function generates the distance martix with a shrinking of the area
def Gen_distance_time_Matrix_size_change(n, dataFrame, demand_node, size_index):
    arr_distance = [[0 for col in range(len(n))] for row in range(len(n))]
    arr_time = [[0 for col in range(len(n))] for row in range(len(n))]
    for i in range(len(n)):
        for j in range(len(n)):
            if i != j:
                x1 = dataFrame.loc[dataframe['ID'] == n[i], 'X'].iloc[0]
                y1 = dataFrame.loc[dataframe['ID'] == n[i], 'Y'].iloc[0]
                x2 = dataFrame.loc[dataframe['ID'] == n[j], 'X'].iloc[0]
                y2 = dataFrame.loc[dataframe['ID'] == n[j], 'Y'].iloc[0]
                travel_distance = 2 * 0.621371192/1000 * math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) * size_index
                travel_time_between_nodes = travel_distance * 60 / para.vehicle_speed_between_nodes
                drop_distance = demand_node[i] * para.distance_between_drops * size_index
                travel_time_within_node = drop_distance * 60 / para.vehicle_speed_between_drops
                service_time = demand_node[i] * para.service_time_per_drop
                arr_distance[i][j] = round(travel_distance + drop_distance)
                arr_time[i][j] = round(travel_time_between_nodes + travel_time_within_node + service_time)

    return arr_distance, arr_time

# smallDistanceMatrix, smallTimeMatrix = Gen_distance_time_Matrix(SmallNodeSet, SmallSet, smallAreaHighDemand)
largeMatrix, largeTimeMatrix = Gen_distance_time_Matrix(Set1, LargeSet, largeAreaHighDemand)
# Matrix2K, TimeMatrix2K = Gen_distance_time_Matrix(LargeNodeSet, LargeSet, largeArea2kDemand)
# Matrix1K, TimeMatrix1K = Gen_distance_time_Matrix(LargeNodeSet, LargeSet, largeArea1kDemand)
# Area1Marix, Area1TimeMatrix = Gen_distance_time_Matrix_size_change(LargeNodeSet, LargeSet, largeAreaHighDemand, 0.707)
# Area2Marix, Area2TimeMatrix = Gen_distance_time_Matrix_size_change(LargeNodeSet, LargeSet, largeAreaHighDemand, 0.5)
# print(largeTimeMatrix)