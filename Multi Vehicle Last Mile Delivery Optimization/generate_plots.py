"""
Academic Research Plots for Multi-Vehicle Last-Mile Delivery Optimization
This script generates publication-quality visualizations for research papers.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Create plots directory
PLOTS_DIR = 'plots'
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_node_coordinates(filepath='LargeSet.csv'):
    """Load node coordinates from CSV file."""
    df = pd.read_csv(filepath)
    return df

def parse_output_file(filepath):
    """Parse optimization output file to extract route information."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract objective value
    obj_match = re.search(r'Objective: (\d+)', content)
    objective = int(obj_match.group(1)) if obj_match else None
    
    # Extract route information
    routes = []
    route_pattern = r'Route for vehicle (\d+):\n(.*?)\nDistance of the route: (\d+)m\nTotal Cost of the route: (\d+)m'
    matches = re.findall(route_pattern, content)
    
    for match in matches:
        vehicle_id = int(match[0])
        route_str = match[1].strip()
        distance = int(match[2])
        cost = int(match[3])
        
        # Parse route nodes
        nodes = [int(n.strip()) for n in route_str.replace(' -> ', ',').split(',') if n.strip().isdigit()]
        
        routes.append({
            'vehicle_id': vehicle_id,
            'route': nodes,
            'distance': distance,
            'cost': cost,
            'num_stops': len(nodes) - 2 if len(nodes) > 2 else 0  # Exclude depot visits
        })
    
    return {'objective': objective, 'routes': routes}

def load_all_outputs(output_dir='Output1'):
    """Load all output files and extract experiment configurations."""
    results = []
    for filename in os.listdir(output_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(output_dir, filename)
            
            # Parse filename for configuration
            config_match = re.match(
                r'V_num(\d+)_EV(\d+)_EVRange(\d+)_EVCost([\d.]+)_GVCost([\d.]+)\.txt',
                filename
            )
            if config_match:
                config = {
                    'num_vehicles': int(config_match.group(1)),
                    'num_ev': int(config_match.group(2)),
                    'ev_range': int(config_match.group(3)),
                    'ev_cost': float(config_match.group(4)),
                    'gv_cost': float(config_match.group(5)),
                    'filename': filename
                }
                
                data = parse_output_file(filepath)
                config.update(data)
                results.append(config)
    
    return results

# ============================================================================
# Plot 1: Delivery Network Topology
# ============================================================================

def plot_network_topology(nodes_df, depot_id=0):
    """Plot the spatial distribution of delivery nodes."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Convert coordinates to km for readability
    x = (nodes_df['X'] - nodes_df['X'].mean()) / 1000
    y = (nodes_df['Y'] - nodes_df['Y'].mean()) / 1000
    
    # Plot all nodes
    ax.scatter(x[1:], y[1:], c='steelblue', s=80, alpha=0.7, 
               edgecolors='darkblue', linewidth=0.5, label='Customer Nodes')
    
    # Highlight depot
    ax.scatter(x.iloc[0], y.iloc[0], c='red', s=200, marker='s', 
               edgecolors='darkred', linewidth=2, label='Depot', zorder=5)
    
    # Add node labels
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.annotate(str(i), (xi, yi), fontsize=7, ha='center', va='bottom',
                   xytext=(0, 3), textcoords='offset points')
    
    ax.set_xlabel('Relative X Position (km)')
    ax.set_ylabel('Relative Y Position (km)')
    ax.set_title('Delivery Network Topology')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig1_network_topology.png'))
    plt.savefig(os.path.join(PLOTS_DIR, 'fig1_network_topology.pdf'))
    plt.close()
    print("Generated: fig1_network_topology")

# ============================================================================
# Plot 2: Fleet Size vs. Total Cost Analysis
# ============================================================================

def plot_fleet_size_vs_cost(results):
    """Plot the relationship between fleet size and total operational cost."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Group by cost configuration
    cost_configs = {}
    for r in results:
        key = f"EV: ${r['ev_cost']}/mi, GV: ${r['gv_cost']}/mi"
        if key not in cost_configs:
            cost_configs[key] = {'vehicles': [], 'objectives': []}
        cost_configs[key]['vehicles'].append(r['num_vehicles'])
        cost_configs[key]['objectives'].append(r['objective'])
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    markers = ['o', 's', '^', 'D']
    
    for i, (config, data) in enumerate(cost_configs.items()):
        # Sort by number of vehicles
        sorted_data = sorted(zip(data['vehicles'], data['objectives']))
        vehicles, objectives = zip(*sorted_data)
        
        ax.plot(vehicles, objectives, marker=markers[i % len(markers)], 
                color=colors[i % len(colors)], linewidth=2, markersize=8,
                label=config)
    
    ax.set_xlabel('Fleet Size (Number of Vehicles)')
    ax.set_ylabel('Total Operational Cost ($)')
    ax.set_title('Impact of Fleet Size on Total Operational Cost')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig2_fleet_size_vs_cost.png'))
    plt.savefig(os.path.join(PLOTS_DIR, 'fig2_fleet_size_vs_cost.pdf'))
    plt.close()
    print("Generated: fig2_fleet_size_vs_cost")

# ============================================================================
# Plot 3: Route Distance Distribution
# ============================================================================

def plot_route_distance_distribution(results):
    """Plot distribution of route distances across all vehicles."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Collect all route distances (excluding empty routes)
    all_distances = []
    all_costs = []
    for r in results:
        for route in r['routes']:
            if route['distance'] > 0:
                all_distances.append(route['distance'])
                all_costs.append(route['cost'])
    
    # Distance distribution
    ax1 = axes[0]
    ax1.hist(all_distances, bins=20, color='steelblue', edgecolor='darkblue', alpha=0.7)
    ax1.axvline(np.mean(all_distances), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(all_distances):.1f} mi')
    ax1.axvline(np.median(all_distances), color='orange', linestyle=':', linewidth=2,
                label=f'Median: {np.median(all_distances):.1f} mi')
    ax1.set_xlabel('Route Distance (miles)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('(a) Distribution of Route Distances')
    ax1.legend()
    
    # Cost distribution
    ax2 = axes[1]
    ax2.hist(all_costs, bins=20, color='#2E86AB', edgecolor='#1a5276', alpha=0.7)
    ax2.axvline(np.mean(all_costs), color='red', linestyle='--', linewidth=2,
                label=f'Mean: ${np.mean(all_costs):.1f}')
    ax2.axvline(np.median(all_costs), color='orange', linestyle=':', linewidth=2,
                label=f'Median: ${np.median(all_costs):.1f}')
    ax2.set_xlabel('Route Cost ($)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('(b) Distribution of Route Costs')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig3_route_distributions.png'))
    plt.savefig(os.path.join(PLOTS_DIR, 'fig3_route_distributions.pdf'))
    plt.close()
    print("Generated: fig3_route_distributions")

# ============================================================================
# Plot 4: Vehicle Utilization Analysis
# ============================================================================

def plot_vehicle_utilization(results):
    """Plot vehicle utilization across different fleet configurations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Select a representative result (e.g., 28 vehicles)
    target_result = None
    for r in results:
        if r['num_vehicles'] == 28 and r['ev_cost'] == 0.25:
            target_result = r
            break
    
    if target_result is None:
        target_result = results[0]
    
    vehicles = []
    distances = []
    costs = []
    stops = []
    
    for route in target_result['routes']:
        vehicles.append(route['vehicle_id'])
        distances.append(route['distance'])
        costs.append(route['cost'])
        stops.append(route['num_stops'])
    
    x = np.arange(len(vehicles))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, distances, width, label='Distance (mi)', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, costs, width, label='Cost ($)', color='#F18F01', alpha=0.8)
    
    ax.set_xlabel('Vehicle ID')
    ax.set_ylabel('Value')
    ax.set_title(f'Vehicle Utilization Analysis (Fleet Size: {target_result["num_vehicles"]})')
    ax.set_xticks(x[::2])
    ax.set_xticklabels([str(v) for v in vehicles[::2]])
    ax.legend()
    
    # Add utilization percentage on secondary axis
    ax2 = ax.twinx()
    max_range = 300  # Maximum vehicle range
    utilization = [d / max_range * 100 for d in distances]
    ax2.plot(x, utilization, 'r--', marker='o', markersize=4, label='Range Utilization %')
    ax2.set_ylabel('Range Utilization (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig4_vehicle_utilization.png'))
    plt.savefig(os.path.join(PLOTS_DIR, 'fig4_vehicle_utilization.pdf'))
    plt.close()
    print("Generated: fig4_vehicle_utilization")

# ============================================================================
# Plot 5: Cost Comparison - EV vs GV
# ============================================================================

def plot_ev_gv_cost_comparison():
    """Plot theoretical cost comparison between EV and GV fleets."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Parameters from the study
    distances = np.linspace(0, 300, 100)
    
    # Cost per mile comparison
    ev_costs = [0.25, 0.30, 0.35]
    gv_costs = [0.50, 0.60, 0.70]
    
    ax1 = axes[0]
    colors_ev = ['#27ae60', '#2ecc71', '#82e0aa']
    colors_gv = ['#c0392b', '#e74c3c', '#f1948a']
    
    for i, (ev_c, gv_c) in enumerate(zip(ev_costs, gv_costs)):
        ax1.plot(distances, distances * ev_c, color=colors_ev[i], linewidth=2,
                label=f'EV (${ev_c}/mi)')
        ax1.plot(distances, distances * gv_c, color=colors_gv[i], linewidth=2,
                linestyle='--', label=f'GV (${gv_c}/mi)')
    
    ax1.set_xlabel('Distance Traveled (miles)')
    ax1.set_ylabel('Distance-Based Cost ($)')
    ax1.set_title('(a) Distance-Based Cost Comparison')
    ax1.legend(loc='upper left', ncol=2)
    ax1.set_xlim(0, 300)
    
    # Total cost breakdown
    ax2 = axes[1]
    
    # Sample scenario: 150 miles, 200 minutes labor
    distance = 150
    labor_time = 200  # minutes
    labor_cost_rate = 0.6  # $/minute
    
    categories = ['EV Fleet\n($0.25/mi)', 'EV Fleet\n($0.30/mi)', 
                  'GV Fleet\n($0.50/mi)', 'GV Fleet\n($0.60/mi)']
    distance_costs = [distance * 0.25, distance * 0.30, distance * 0.50, distance * 0.60]
    labor_costs = [labor_time * labor_cost_rate] * 4
    
    x = np.arange(len(categories))
    width = 0.5
    
    bars1 = ax2.bar(x, distance_costs, width, label='Distance Cost', color='steelblue')
    bars2 = ax2.bar(x, labor_costs, width, bottom=distance_costs, label='Labor Cost', color='#F18F01')
    
    # Add total labels
    for i, (dc, lc) in enumerate(zip(distance_costs, labor_costs)):
        ax2.annotate(f'${dc + lc:.0f}', xy=(i, dc + lc + 5), ha='center', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Total Cost ($)')
    ax2.set_title('(b) Cost Breakdown per Vehicle Route\n(150 mi, 200 min)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig5_ev_gv_cost_comparison.png'))
    plt.savefig(os.path.join(PLOTS_DIR, 'fig5_ev_gv_cost_comparison.pdf'))
    plt.close()
    print("Generated: fig5_ev_gv_cost_comparison")

# ============================================================================
# Plot 6: Sensitivity Analysis Heatmap
# ============================================================================

def plot_sensitivity_heatmap():
    """Plot sensitivity analysis heatmap for cost parameters."""
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Create synthetic sensitivity data
    ev_cost_range = np.arange(0.20, 0.45, 0.05)
    gv_cost_range = np.arange(0.40, 0.80, 0.05)
    
    # Generate cost matrix (total cost as function of EV and GV costs)
    # Assuming base cost components
    base_ev_distance = 500  # Total EV fleet distance
    base_gv_distance = 800  # Total GV fleet distance
    base_labor_cost = 2000
    
    cost_matrix = np.zeros((len(gv_cost_range), len(ev_cost_range)))
    for i, gv_c in enumerate(gv_cost_range):
        for j, ev_c in enumerate(ev_cost_range):
            cost_matrix[i, j] = base_ev_distance * ev_c + base_gv_distance * gv_c + base_labor_cost
    
    im = ax.imshow(cost_matrix, cmap='RdYlGn_r', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(ev_cost_range)))
    ax.set_yticks(np.arange(len(gv_cost_range)))
    ax.set_xticklabels([f'${x:.2f}' for x in ev_cost_range])
    ax.set_yticklabels([f'${y:.2f}' for y in gv_cost_range])
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Total Operational Cost ($)', rotation=-90, va='bottom')
    
    # Add value annotations
    for i in range(len(gv_cost_range)):
        for j in range(len(ev_cost_range)):
            text = ax.text(j, i, f'{cost_matrix[i, j]:.0f}',
                          ha='center', va='center', color='black', fontsize=9)
    
    ax.set_xlabel('EV Cost per Mile')
    ax.set_ylabel('GV Cost per Mile')
    ax.set_title('Sensitivity Analysis: Impact of Vehicle Operating Costs')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig6_sensitivity_heatmap.png'))
    plt.savefig(os.path.join(PLOTS_DIR, 'fig6_sensitivity_heatmap.pdf'))
    plt.close()
    print("Generated: fig6_sensitivity_heatmap")

# ============================================================================
# Plot 7: Optimized Routes Visualization
# ============================================================================

def plot_optimized_routes(nodes_df, results):
    """Visualize optimized delivery routes on the network."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Select a result to visualize
    target_result = None
    for r in results:
        if r['num_vehicles'] == 25:
            target_result = r
            break
    
    if target_result is None:
        target_result = results[0]
    
    # Convert coordinates
    x = (nodes_df['X'] - nodes_df['X'].mean()) / 1000
    y = (nodes_df['Y'] - nodes_df['Y'].mean()) / 1000
    coords = list(zip(x, y))
    
    # Generate distinct colors for routes
    n_routes = len([r for r in target_result['routes'] if r['distance'] > 0])
    cmap = plt.cm.get_cmap('tab20', n_routes)
    
    # Plot routes
    route_idx = 0
    for route_data in target_result['routes']:
        if route_data['distance'] > 0 and len(route_data['route']) > 2:
            route = route_data['route']
            color = cmap(route_idx)
            
            # Plot route edges
            for i in range(len(route) - 1):
                if route[i] < len(coords) and route[i+1] < len(coords):
                    x1, y1 = coords[route[i]]
                    x2, y2 = coords[route[i+1]]
                    ax.plot([x1, x2], [y1, y2], color=color, linewidth=1.5, alpha=0.7)
                    
                    # Add arrow
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    dx, dy = x2 - x1, y2 - y1
                    ax.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1),
                               xytext=(mid_x - dx*0.1, mid_y - dy*0.1),
                               arrowprops=dict(arrowstyle='->', color=color, lw=1))
            
            route_idx += 1
    
    # Plot nodes
    ax.scatter(x[1:], y[1:], c='white', s=100, edgecolors='darkblue', 
               linewidth=1.5, zorder=5)
    
    # Plot depot
    ax.scatter(x.iloc[0], y.iloc[0], c='red', s=250, marker='s',
               edgecolors='darkred', linewidth=2, zorder=6, label='Depot')
    
    # Add node labels
    for i, (xi, yi) in enumerate(coords):
        ax.annotate(str(i), (xi, yi), fontsize=7, ha='center', va='center',
                   fontweight='bold', zorder=7)
    
    ax.set_xlabel('Relative X Position (km)')
    ax.set_ylabel('Relative Y Position (km)')
    ax.set_title(f'Optimized Delivery Routes (Fleet Size: {target_result["num_vehicles"]}, '
                f'Objective: ${target_result["objective"]})')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig7_optimized_routes.png'))
    plt.savefig(os.path.join(PLOTS_DIR, 'fig7_optimized_routes.pdf'))
    plt.close()
    print("Generated: fig7_optimized_routes")

# ============================================================================
# Plot 8: Convergence Analysis (Computational Performance)
# ============================================================================

def plot_computational_analysis(results):
    """Plot computational performance metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Fleet size vs number of active vehicles
    ax1 = axes[0]
    
    for r in sorted(results, key=lambda x: x['num_vehicles']):
        active_vehicles = sum(1 for route in r['routes'] if route['distance'] > 0)
        total_vehicles = r['num_vehicles']
        
        ax1.scatter(total_vehicles, active_vehicles, s=100, c='steelblue', 
                   edgecolors='darkblue', alpha=0.7)
    
    # Add diagonal reference line
    ax1.plot([20, 35], [20, 35], 'r--', linewidth=1, label='100% Utilization')
    
    ax1.set_xlabel('Total Fleet Size')
    ax1.set_ylabel('Active Vehicles')
    ax1.set_title('(a) Fleet Utilization Efficiency')
    ax1.legend()
    ax1.set_xlim(24, 30)
    ax1.set_ylim(15, 30)
    
    # Average stops per vehicle
    ax2 = axes[1]
    
    fleet_sizes = []
    avg_stops = []
    std_stops = []
    
    for r in sorted(results, key=lambda x: x['num_vehicles']):
        if r['ev_cost'] == 0.25:  # Filter to one cost config
            stops = [route['num_stops'] for route in r['routes'] if route['distance'] > 0]
            if stops:
                fleet_sizes.append(r['num_vehicles'])
                avg_stops.append(np.mean(stops))
                std_stops.append(np.std(stops))
    
    ax2.errorbar(fleet_sizes, avg_stops, yerr=std_stops, fmt='o-', 
                color='steelblue', capsize=5, capthick=2, linewidth=2, markersize=8)
    
    ax2.set_xlabel('Fleet Size')
    ax2.set_ylabel('Average Stops per Vehicle')
    ax2.set_title('(b) Workload Distribution')
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig8_computational_analysis.png'))
    plt.savefig(os.path.join(PLOTS_DIR, 'fig8_computational_analysis.pdf'))
    plt.close()
    print("Generated: fig8_computational_analysis")

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Generate all plots for the research paper."""
    print("=" * 60)
    print("Generating Academic Research Plots")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    nodes_df = load_node_coordinates('LargeSet.csv')
    results = load_all_outputs('Output1')
    print(f"Loaded {len(nodes_df)} nodes and {len(results)} experiment results")
    
    # Generate all plots
    print("\nGenerating plots...\n")
    
    plot_network_topology(nodes_df)
    plot_fleet_size_vs_cost(results)
    plot_route_distance_distribution(results)
    plot_vehicle_utilization(results)
    plot_ev_gv_cost_comparison()
    plot_sensitivity_heatmap()
    plot_optimized_routes(nodes_df, results)
    plot_computational_analysis(results)
    
    print("\n" + "=" * 60)
    print(f"All plots saved to '{PLOTS_DIR}/' directory")
    print("Formats: PNG (300 DPI) and PDF (vector)")
    print("=" * 60)

if __name__ == '__main__':
    main()
