"""
Multi-Depot Mixed Vehicle Routing Problem Optimizer

This module solves the Vehicle Routing Problem for multiple depots with
heterogeneous fleet (Electric Vehicles and Gasoline Vehicles).

The objective is to minimize total operational cost:
- Distance-based cost (differentiated by vehicle type)
- Time-based labor cost
"""

from functools import partial
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import os
import json
import Parameters_MultiDepot as para
from Input_MultiDepot import RegionData

# Output directory
OUTPUT_DIR = 'Output_MultiDepot'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def solve_region(region_id, region_data, num_vehicles, vehicle_range, EV_cost, GV_cost, num_ev):
    """
    Solve VRP for a single depot/region.
    
    Args:
        region_id: Region identifier (1, 2, or 3)
        region_data: Dictionary containing region-specific data
        num_vehicles: Total number of vehicles for this region
        vehicle_range: List of vehicle ranges
        EV_cost: Cost per mile for EVs
        GV_cost: Cost per mile for GVs
        num_ev: Number of electric vehicles
    
    Returns:
        Dictionary with optimization results
    """
    distance_matrix = region_data['distance_matrix']
    time_matrix = region_data['time_matrix']
    depot_id = region_data['depot_id']
    
    def create_data_model():
        """Stores the data for the problem."""
        data = {
            'distance_matrix': distance_matrix,
            'time_matrix': time_matrix,
            'num_vehicles': num_vehicles,
            'depot': 0
        }
        return data
    
    def solve():
        """Solve the VRP for this region."""
        data = create_data_model()
        
        # Create routing index manager
        manager = pywrapcp.RoutingIndexManager(
            len(data['distance_matrix']),
            data['num_vehicles'],
            data['depot']
        )
        
        # Create routing model
        routing = pywrapcp.RoutingModel(manager)
        
        # Distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]
        
        # Travel time callback
        def travel_time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['time_matrix'][from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        travel_time_callback_index = routing.RegisterTransitCallback(travel_time_callback)
        
        # Add distance constraint
        routing.AddDimensionWithVehicleCapacity(
            transit_callback_index,
            0,  # no slack
            vehicle_range,  # vehicle maximum travel distance
            True,  # start cumul to zero
            'distance'
        )
        
        # Add working hours constraint
        routing.AddDimension(
            travel_time_callback_index,
            0,
            para.maximum_working_time,
            True,
            'working_time'
        )
        
        # Cost callbacks for EV and GV
        def vehicle_cost_callback(from_index, to_index, vehicle_type):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            
            if vehicle_type == "EV":
                routing_cost = round(data['distance_matrix'][from_node][to_node] * EV_cost)
            else:
                routing_cost = round(data['distance_matrix'][from_node][to_node] * GV_cost)
            
            labor_cost = round(data['time_matrix'][from_node][to_node] * para.labor_cost_per_minute)
            return routing_cost + labor_cost
        
        EV_cost_callback = partial(vehicle_cost_callback, vehicle_type='EV')
        GV_cost_callback = partial(vehicle_cost_callback, vehicle_type='GV')
        
        EV_cost_index = routing.RegisterTransitCallback(EV_cost_callback)
        GV_cost_index = routing.RegisterTransitCallback(GV_cost_callback)
        
        # Set cost evaluators per vehicle
        for i in range(num_ev):
            routing.SetArcCostEvaluatorOfVehicle(EV_cost_index, i)
        for i in range(num_ev, num_vehicles):
            routing.SetArcCostEvaluatorOfVehicle(GV_cost_index, i)
        
        # Search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        
        if para.USE_GUIDED_LOCAL_SEARCH:
            search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        
        search_parameters.time_limit.FromSeconds(para.SOLVER_TIME_LIMIT)
        
        # Solve
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            return extract_solution(data, manager, routing, solution, num_ev, EV_cost, GV_cost)
        else:
            return None
    
    def extract_solution(data, manager, routing, solution, num_ev, EV_cost, GV_cost):
        """Extract solution details."""
        results = {
            'region_id': region_id,
            'depot_id': depot_id,
            'objective': solution.ObjectiveValue(),
            'routes': [],
            'total_distance': 0,
            'total_cost': 0
        }
        
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            route_nodes = []
            route_distance = 0
            route_cost = 0
            vehicle_type = 'EV' if vehicle_id < num_ev else 'GV'
            cost_per_mile = EV_cost if vehicle_id < num_ev else GV_cost
            
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route_nodes.append(node)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                
                from_node = manager.IndexToNode(previous_index)
                to_node = manager.IndexToNode(index)
                
                segment_distance = data['distance_matrix'][from_node][to_node]
                segment_cost = round(segment_distance * cost_per_mile) + \
                              round(data['time_matrix'][from_node][to_node] * para.labor_cost_per_minute)
                
                route_distance += segment_distance
                route_cost += segment_cost
            
            route_nodes.append(manager.IndexToNode(index))  # Add final depot
            
            results['routes'].append({
                'vehicle_id': vehicle_id,
                'vehicle_type': vehicle_type,
                'route': route_nodes,
                'distance': route_distance,
                'cost': route_cost,
                'num_stops': len(route_nodes) - 2 if len(route_nodes) > 2 else 0
            })
            
            results['total_distance'] += route_distance
            results['total_cost'] += route_cost
        
        return results
    
    return solve()


def run_multi_depot_optimization(num_vehicles_per_depot, num_ev_per_depot, ev_range, gv_range, EV_cost, GV_cost):
    """
    Run optimization for all depots.
    
    Args:
        num_vehicles_per_depot: Number of vehicles per depot
        num_ev_per_depot: Number of EVs per depot
        ev_range: Range of EVs in miles
        gv_range: Range of GVs in miles
        EV_cost: EV cost per mile
        GV_cost: GV cost per mile
    
    Returns:
        Combined results from all depots
    """
    all_results = {
        'config': {
            'num_depots': para.NUM_DEPOTS,
            'num_vehicles_per_depot': num_vehicles_per_depot,
            'num_ev_per_depot': num_ev_per_depot,
            'ev_range': ev_range,
            'gv_range': gv_range,
            'EV_cost': EV_cost,
            'GV_cost': GV_cost
        },
        'regions': [],
        'summary': {
            'total_objective': 0,
            'total_distance': 0,
            'total_cost': 0,
            'total_active_vehicles': 0
        }
    }
    
    # Build vehicle range list
    vehicle_range = [ev_range] * num_ev_per_depot + [gv_range] * (num_vehicles_per_depot - num_ev_per_depot)
    
    print("=" * 70)
    print("MULTI-DEPOT VEHICLE ROUTING OPTIMIZATION")
    print("=" * 70)
    
    for region_id in [1, 2, 3]:
        region_data = RegionData[region_id]
        print(f"\nSolving Region {region_id} (Depot ID: {region_data['depot_id']}, "
              f"Customers: {len(region_data['customers'])})...")
        
        result = solve_region(
            region_id=region_id,
            region_data=region_data,
            num_vehicles=num_vehicles_per_depot,
            vehicle_range=vehicle_range,
            EV_cost=EV_cost,
            GV_cost=GV_cost,
            num_ev=num_ev_per_depot
        )
        
        if result:
            all_results['regions'].append(result)
            all_results['summary']['total_objective'] += result['objective']
            all_results['summary']['total_distance'] += result['total_distance']
            all_results['summary']['total_cost'] += result['total_cost']
            
            active = sum(1 for r in result['routes'] if r['distance'] > 0)
            all_results['summary']['total_active_vehicles'] += active
            
            print(f"  Objective: ${result['objective']}")
            print(f"  Total Distance: {result['total_distance']} miles")
            print(f"  Active Vehicles: {active}/{num_vehicles_per_depot}")
        else:
            print(f"  No solution found for Region {region_id}!")
    
    return all_results


def save_results(results, filename):
    """Save results to JSON and text files."""
    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, filename + '.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save readable text
    txt_path = os.path.join(OUTPUT_DIR, filename + '.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MULTI-DEPOT VEHICLE ROUTING OPTIMIZATION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        # Configuration
        f.write("CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        for key, value in results['config'].items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Summary
        f.write("OVERALL SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total Objective Value: ${results['summary']['total_objective']}\n")
        f.write(f"  Total Distance: {results['summary']['total_distance']} miles\n")
        f.write(f"  Total Active Vehicles: {results['summary']['total_active_vehicles']}\n")
        f.write("\n")
        
        # Per-region details
        for region in results['regions']:
            f.write(f"\nREGION {region['region_id']} (Depot ID: {region['depot_id']}):\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Objective: ${region['objective']}\n")
            f.write(f"  Total Distance: {region['total_distance']} miles\n")
            f.write("\n  Routes:\n")
            
            for route in region['routes']:
                if route['distance'] > 0:
                    f.write(f"    Vehicle {route['vehicle_id']} ({route['vehicle_type']}): ")
                    f.write(f"{' -> '.join(map(str, route['route']))}\n")
                    f.write(f"      Distance: {route['distance']} mi, Cost: ${route['cost']}, "
                           f"Stops: {route['num_stops']}\n")
    
    print(f"\nResults saved to:\n  {json_path}\n  {txt_path}")


def main():
    """Main execution function."""
    # Test different configurations
    configurations = [
        # (vehicles_per_depot, ev_per_depot, ev_range, gv_range, ev_cost, gv_cost)
        (10, 0, 200, 300, 0.25, 0.50),   # All GV
        (10, 3, 200, 300, 0.25, 0.50),   # Mixed fleet
        (10, 5, 200, 300, 0.25, 0.50),   # More EVs
        (10, 3, 200, 300, 0.30, 0.60),   # Higher costs
        (12, 4, 200, 300, 0.25, 0.50),   # Larger fleet
    ]
    
    for config in configurations:
        num_v, num_ev, ev_r, gv_r, ev_c, gv_c = config
        
        print(f"\n{'='*70}")
        print(f"Configuration: {num_v} vehicles/depot, {num_ev} EVs, "
              f"EV Cost=${ev_c}/mi, GV Cost=${gv_c}/mi")
        print(f"{'='*70}")
        
        results = run_multi_depot_optimization(
            num_vehicles_per_depot=num_v,
            num_ev_per_depot=num_ev,
            ev_range=ev_r,
            gv_range=gv_r,
            EV_cost=ev_c,
            GV_cost=gv_c
        )
        
        # Generate filename
        filename = f"MultiDepot_V{num_v}_EV{num_ev}_EVRange{ev_r}_EVCost{ev_c}_GVCost{gv_c}"
        save_results(results, filename)
        
        # Print summary
        print(f"\n{'='*70}")
        print("FINAL SUMMARY")
        print(f"{'='*70}")
        print(f"Total Objective: ${results['summary']['total_objective']}")
        print(f"Total Distance: {results['summary']['total_distance']} miles")
        print(f"Total Active Vehicles: {results['summary']['total_active_vehicles']} / {num_v * 3}")


if __name__ == '__main__':
    main()
