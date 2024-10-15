"""Mix vehicle routing problem.
   The objective is to minimize the summation of distance-wise cost and time-wise cost
   The input is the coordinates of the locations, the load of each location,
   Time window is not included, but the total time of travel for each vehicle is considered
   Distances are in miles and time in hours.
"""
# [START import]
from functools import partial
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from Input import largeMatrix, largeTimeMatrix

# largeMatrix is the base scenario
import os
import Parameters as para
# [END import]

# OutputDirectory = 'C:\Users\HP\PycharmProjects\pythonProject9'
OutputDirectory = 'Output1'
def Main_function(distance_matrix, time_matrix, num_vehicle, vehicle_range, EV_cost, GV_cost, num_ev):
    # Output file name
    EVRange = vehicle_range[0]
    IndividualFileName = "V_num"+str(num_vehicle)+"_EV"+str(num_ev)+"_EVRange"+str(EVRange)+"_EVCost"+str(EV_cost)+"_GVCost"+str(GV_cost)+'.txt'
    # OutputFileName = OutputDirectory +IndividualFileName

    def create_data_model():
        """Stores the data for the problem."""
        data = {}
        data['distance_matrix'] = distance_matrix
        data['time_matrix'] = time_matrix
        data['num_vehicles'] = num_vehicle
        data['depot'] = 0
        return data

    def print_solution(data, manager, routing, solution, num_ev, EV_cost, GV_cost):
        """Prints solution on console."""
        line = f'Objective: {solution.ObjectiveValue()}'
        print(line)
        if not os.path.exists(os.path.join(os.getcwd(), 'Output1')):
            os.makedirs(os.path.join(os.getcwd(), 'Output1'))
        file = open(os.path.join(OutputDirectory, IndividualFileName), "a")
        file.write(line)
        max_route_distance = 0
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            route_distance = 0
            vehicle_cost = 0
            while not routing.IsEnd(index):
                plan_output += ' {} -> '.format(manager.IndexToNode(index))
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                from_node = manager.IndexToNode(previous_index)
                to_node = manager.IndexToNode(index)
                route_distance += data['distance_matrix'][from_node][to_node]
                if vehicle_id <= num_ev - 1:
                    routingCost = round(data['distance_matrix'][from_node][to_node] * EV_cost)
                    laborCost = round(data['time_matrix'][from_node][to_node] * para.labor_cost_per_minute)
                    total_cost_segment = routingCost + laborCost
                else:
                    routingCost = round(data['distance_matrix'][from_node][to_node] * GV_cost)
                    laborCost = round(data['time_matrix'][from_node][to_node] * para.labor_cost_per_minute)
                    total_cost_segment = routingCost + laborCost
                vehicle_cost += total_cost_segment
                # route_distance += routing.GetArcCostForVehicle(
                #     previous_index, index, vehicle_id)
            plan_output += '{}\n'.format(manager.IndexToNode(index))
            plan_output += 'Distance of the route: {}m\n'.format(route_distance)
            plan_output += 'Total Cost of the route: {}m\n'.format(vehicle_cost)
            print(plan_output)

            file.write(plan_output)

            max_route_distance = max(route_distance, max_route_distance)
        file.close()
        print('Maximum of the route distances: {}m'.format(max_route_distance))

    def main():
        """Entry point of the program."""
        # Instantiate the data problem.
        data = create_data_model()

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        # Create and register a transit callback.
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        # Create and register a travel time callback
        def travel_time_callback(from_index, to_index):
            """Returns the travel time between the two nodes."""
            # Convert from routing variable Index to time matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            travelTime = data['time_matrix'][from_node][to_node]
            return travelTime

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        travel_time_callback_index = routing.RegisterTransitCallback(travel_time_callback)

        # Add Distance constraint.
        dimension_name = 'distance'
        routing.AddDimensionWithVehicleCapacity(
            transit_callback_index,
            0,  # no slack
            vehicle_range,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name)

        # Add working hours constraint
        dimension_name = 'working_time'
        routing.AddDimension(
            travel_time_callback_index,
            0,
            420, # 9 hours minus 1
            True,
            dimension_name)

        # Cost index
        def vehicle_cost_callback(from_index, to_index, vehicle_type):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            if vehicle_type == "EV":
                EV_cost_per_mile = EV_cost
                routingCost = round(data['distance_matrix'][from_node][to_node] * EV_cost_per_mile)
                laborCost = round(data['time_matrix'][from_node][to_node] * para.labor_cost_per_minute)
            elif vehicle_type == 'GV':
                cost_coefficient = GV_cost
                routingCost = round(data['distance_matrix'][from_node][to_node] * cost_coefficient)
                laborCost = round(data['time_matrix'][from_node][to_node] * para.labor_cost_per_minute)

            totalCost = routingCost + laborCost

            return totalCost

        EV_cost_callback = partial(vehicle_cost_callback, vehicle_type='EV')
        GV_cost_callback = partial(vehicle_cost_callback, vehicle_type='GV')

        EV_cost_index = routing.RegisterTransitCallback(EV_cost_callback)
        GV_cost_index = routing.RegisterTransitCallback(GV_cost_callback)

        for i in range(num_ev):
            routing.SetArcCostEvaluatorOfVehicle(EV_cost_index, i)

        for i in range(num_ev, num_vehicle):
            routing.SetArcCostEvaluatorOfVehicle(GV_cost_index, i)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.FromSeconds(2)
        search_parameters.log_search = True
        # [END parameters]

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        # Print solution on console.
        if solution:
            print_solution(data, manager, routing, solution, num_ev, EV_cost, GV_cost)
        else:
            print('No solution found !')

    main()

# Main_function(smallMatrix, num_vehicle, vehicle_range, 3, num_ev)


for EVrange in [200]:
    for EVnum in [0]:
        for num_vehicle in range(25, 30):
            # Construct vehicle range list
            # num_vehicle = 38
            num_ev = EVnum
            vehicle_range = []
            working_hours = []
            for i in range(num_ev):
                vehicle_range.append(EVrange)
            for i in range(num_ev, num_vehicle):
                vehicle_range.append(300)

            # function call for large matrix
            Main_function(largeMatrix, largeTimeMatrix, num_vehicle, vehicle_range, para.EV_cost_per_mile,
            para.GV_cost_per_mile, num_ev)

