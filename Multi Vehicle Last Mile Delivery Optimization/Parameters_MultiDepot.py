# Parameters for Multi-Depot Last Mile Delivery Optimization

# Number of Depots
NUM_DEPOTS = 3

# Depot IDs (will be updated dynamically based on clustering)
DepotIDs = ['4191', None, None]  # First depot fixed, others auto-selected

# Objective cost coefficient
EVCoeff = 1
GVCoeff = 2

# Fixed cost per vehicle
fixCostEV = 300
fixCostGV = 200

# Range of vehicles (miles)
EVRange = 200
GVRange = 300

# Number of vehicles per depot
EVNum_per_depot = 3
GVNum_per_depot = 6
Total_vehicles_per_depot = EVNum_per_depot + GVNum_per_depot

# Number of orders (total across all depots)
HighDemand = 1800
Demand2K = 2500
Demand1K = 1500

# System parameters
distance_between_drops = 1  # miles
vehicle_speed_between_nodes = 45  # mph
vehicle_speed_between_drops = 25  # mph
maximum_working_time = 540  # minutes (9 hours)
service_time_per_drop = 1.5  # minutes
EV_cost_per_mile = 0.30  # $/mile
GV_cost_per_mile = 0.60  # $/mile
labor_cost_per_minute = 0.6  # $/minute

# Solver parameters
SOLVER_TIME_LIMIT = 5  # seconds per region
USE_GUIDED_LOCAL_SEARCH = True
