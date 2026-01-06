"""
Multi-Depot Input Module
This file handles input data for a 3-depot delivery optimization problem.
It splits customers among depots based on geographic clustering.
"""

import pandas as pd
import math
import random
import numpy as np
from sklearn.cluster import KMeans
import Parameters_MultiDepot as para

random.seed(258)
np.random.seed(258)

# Load all nodes
dataframe = pd.read_csv('nodes.csv')

# Original customer set (excluding original depot 4191)
CustomerSet = [3868, 3704, 3864, 3712, 3895, 3940, 3915, 3784, 3983, 3952, 4944, 4989, 4917, 4975, 4906, 4959,
               5040, 5053, 5019, 5036, 5070, 4185, 4107, 4035, 4054, 4012, 4322, 4470, 4443, 4572, 4121, 4143, 4818,
               4823, 5078, 5784, 5790, 4501, 4532, 4657, 4269, 4624, 4722, 4758, 5889, 5721, 5892, 5772, 5740, 4641,
               5841, 5859, 5850, 5984, 5920, 5914, 5907, 6835, 6021, 6038, 6014, 5020, 5807, 5945, 5976, 4272, 4189,
               4538, 4323]

# Get coordinates for clustering
customer_coords = []
for cust_id in CustomerSet:
    row = dataframe.loc[dataframe['ID'] == cust_id]
    if len(row) > 0:
        customer_coords.append([row['X'].iloc[0], row['Y'].iloc[0]])

customer_coords = np.array(customer_coords)

# Use K-means clustering to assign customers to 3 regions
kmeans = KMeans(n_clusters=3, random_state=258, n_init=10)
cluster_labels = kmeans.fit_predict(customer_coords)

# Assign customers to each depot region
Region1_customers = [CustomerSet[i] for i in range(len(CustomerSet)) if cluster_labels[i] == 0]
Region2_customers = [CustomerSet[i] for i in range(len(CustomerSet)) if cluster_labels[i] == 1]
Region3_customers = [CustomerSet[i] for i in range(len(CustomerSet)) if cluster_labels[i] == 2]

# Select depot locations (centroids of each cluster or nearest node to centroid)
def find_nearest_node_to_centroid(centroid, available_nodes, dataframe):
    """Find the node closest to a given centroid."""
    min_dist = float('inf')
    nearest_node = None
    for node_id in available_nodes:
        row = dataframe.loc[dataframe['ID'] == node_id]
        if len(row) > 0:
            x, y = row['X'].iloc[0], row['Y'].iloc[0]
            dist = math.sqrt((x - centroid[0])**2 + (y - centroid[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node_id
    return nearest_node

# Get all available nodes for depot selection
all_nodes = dataframe['ID'].tolist()

# Select depots near cluster centroids
Depot1 = 4191  # Original depot - keep for Region 1
Depot2 = find_nearest_node_to_centroid(kmeans.cluster_centers_[1], all_nodes, dataframe)
Depot3 = find_nearest_node_to_centroid(kmeans.cluster_centers_[2], all_nodes, dataframe)

# Ensure depots are not in customer lists
for depot in [Depot1, Depot2, Depot3]:
    if depot in Region1_customers:
        Region1_customers.remove(depot)
    if depot in Region2_customers:
        Region2_customers.remove(depot)
    if depot in Region3_customers:
        Region3_customers.remove(depot)

# Create node sets with depot at index 0
Region1_nodes = [Depot1] + Region1_customers
Region2_nodes = [Depot2] + Region2_customers
Region3_nodes = [Depot3] + Region3_customers

print(f"Multi-Depot Configuration:")
print(f"  Depot 1 (ID: {Depot1}): {len(Region1_customers)} customers")
print(f"  Depot 2 (ID: {Depot2}): {len(Region2_customers)} customers")
print(f"  Depot 3 (ID: {Depot3}): {len(Region3_customers)} customers")
print(f"  Total customers: {len(Region1_customers) + len(Region2_customers) + len(Region3_customers)}")

# Save region data to CSV files
def save_region_data(node_list, depot_id, filename, dataframe):
    """Save region nodes to CSV file."""
    region_df = dataframe.loc[dataframe['ID'].isin(node_list)].copy()
    # Reorder to put depot first
    depot_row = region_df[region_df['ID'] == depot_id]
    other_rows = region_df[region_df['ID'] != depot_id]
    region_df = pd.concat([depot_row, other_rows], ignore_index=True)
    region_df.to_csv(filename, index=False)
    return region_df

Region1_df = save_region_data(Region1_nodes, Depot1, 'Region1_nodes.csv', dataframe)
Region2_df = save_region_data(Region2_nodes, Depot2, 'Region2_nodes.csv', dataframe)
Region3_df = save_region_data(Region3_nodes, Depot3, 'Region3_nodes.csv', dataframe)


# Generate demand for each region
def Gen_demand(total_demand, num_nodes):
    """Generate demand for each node (depot has 0 demand)."""
    demand_location = [0]  # Depot has no demand
    for i in range(num_nodes - 1):
        demand_temp = int(total_demand / (num_nodes - 1)) + random.randint(-3, 4)
        demand_location.append(demand_temp)
    return demand_location


# Generate demands for each region
Region1_demand = Gen_demand(para.HighDemand // 3, len(Region1_nodes))
Region2_demand = Gen_demand(para.HighDemand // 3, len(Region2_nodes))
Region3_demand = Gen_demand(para.HighDemand // 3, len(Region3_nodes))


# Calculate distance and time matrices
def Gen_distance_time_Matrix(node_ids, dataFrame, demand_node):
    """Generate distance and time matrices for a region."""
    n = len(node_ids)
    arr_distance = [[0 for _ in range(n)] for _ in range(n)]
    arr_time = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                row_i = dataFrame.loc[dataFrame['ID'] == node_ids[i]]
                row_j = dataFrame.loc[dataFrame['ID'] == node_ids[j]]
                
                if len(row_i) > 0 and len(row_j) > 0:
                    x1, y1 = row_i['X'].iloc[0], row_i['Y'].iloc[0]
                    x2, y2 = row_j['X'].iloc[0], row_j['Y'].iloc[0]
                    
                    # Calculate travel distance (converted to miles)
                    travel_distance = 2 * 0.621371192 / 1000 * math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    travel_time_between_nodes = travel_distance * 60 / para.vehicle_speed_between_nodes
                    
                    # Intra-node distance based on demand
                    drop_distance = demand_node[i] * para.distance_between_drops
                    travel_time_within_node = drop_distance * 60 / para.vehicle_speed_between_drops
                    service_time = demand_node[i] * para.service_time_per_drop
                    
                    arr_distance[i][j] = round(travel_distance + drop_distance)
                    arr_time[i][j] = round(travel_time_between_nodes + travel_time_within_node + service_time)
    
    return arr_distance, arr_time


# Generate matrices for each region
Region1_distMatrix, Region1_timeMatrix = Gen_distance_time_Matrix(Region1_nodes, dataframe, Region1_demand)
Region2_distMatrix, Region2_timeMatrix = Gen_distance_time_Matrix(Region2_nodes, dataframe, Region2_demand)
Region3_distMatrix, Region3_timeMatrix = Gen_distance_time_Matrix(Region3_nodes, dataframe, Region3_demand)

# Store all region data in dictionaries for easy access
RegionData = {
    1: {
        'depot_id': Depot1,
        'nodes': Region1_nodes,
        'customers': Region1_customers,
        'demand': Region1_demand,
        'distance_matrix': Region1_distMatrix,
        'time_matrix': Region1_timeMatrix,
        'dataframe': Region1_df
    },
    2: {
        'depot_id': Depot2,
        'nodes': Region2_nodes,
        'customers': Region2_customers,
        'demand': Region2_demand,
        'distance_matrix': Region2_distMatrix,
        'time_matrix': Region2_timeMatrix,
        'dataframe': Region2_df
    },
    3: {
        'depot_id': Depot3,
        'nodes': Region3_nodes,
        'customers': Region3_customers,
        'demand': Region3_demand,
        'distance_matrix': Region3_distMatrix,
        'time_matrix': Region3_timeMatrix,
        'dataframe': Region3_df
    }
}

# Print summary
print("\nRegion Data Summary:")
for region_id, data in RegionData.items():
    print(f"  Region {region_id}: Depot={data['depot_id']}, Nodes={len(data['nodes'])}, "
          f"Distance Matrix Shape={len(data['distance_matrix'])}x{len(data['distance_matrix'][0])}")
