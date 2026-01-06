"""
Academic Research Plots for Multi-Depot Last-Mile Delivery Optimization
This script generates publication-quality visualizations for research papers.
Supports 3-depot configuration with heterogeneous fleet.
"""

import os
import re
import json
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

# Color schemes for depots
DEPOT_COLORS = ['#e74c3c', '#27ae60', '#3498db']  # Red, Green, Blue
DEPOT_NAMES = ['Depot 1', 'Depot 2', 'Depot 3']

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_all_nodes():
    """Load all node coordinates."""
    return pd.read_csv('nodes.csv')

def load_region_data():
    """Load region-specific node data."""
    regions = {}
    for i in [1, 2, 3]:
        filepath = f'Region{i}_nodes.csv'
        if os.path.exists(filepath):
            regions[i] = pd.read_csv(filepath)
    return regions

def load_multi_depot_results(output_dir='Output_MultiDepot'):
    """Load all multi-depot optimization results."""
    results = []
    if not os.path.exists(output_dir):
        return results
    
    for filename in os.listdir(output_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                results.append(data)
    
    return results

# ============================================================================
# Plot 1: Multi-Depot Network Topology
# ============================================================================

def plot_multi_depot_topology(all_nodes, region_data):
    """Plot the spatial distribution of delivery nodes across 3 depots."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Normalize coordinates
    x_mean = all_nodes['X'].mean()
    y_mean = all_nodes['Y'].mean()
    
    # Plot each region with different colors
    for region_id, df in region_data.items():
        x = (df['X'] - x_mean) / 1000
        y = (df['Y'] - y_mean) / 1000
        color = DEPOT_COLORS[region_id - 1]
        
        # Plot customer nodes (skip first row which is depot)
        ax.scatter(x[1:], y[1:], c=color, s=60, alpha=0.7,
                  edgecolors='black', linewidth=0.5,
                  label=f'Region {region_id} Customers')
        
        # Plot depot with distinct marker
        ax.scatter(x.iloc[0], y.iloc[0], c=color, s=300, marker='s',
                  edgecolors='black', linewidth=2, zorder=5)
        
        # Add depot label
        ax.annotate(f'D{region_id}', (x.iloc[0], y.iloc[0]),
                   fontsize=10, fontweight='bold', ha='center', va='center',
                   color='white')
    
    # Add node labels
    for region_id, df in region_data.items():
        x = (df['X'] - x_mean) / 1000
        y = (df['Y'] - y_mean) / 1000
        for i, (xi, yi) in enumerate(zip(x[1:], y[1:])):
            ax.annotate(str(i+1), (xi, yi), fontsize=6, ha='center', va='bottom',
                       xytext=(0, 2), textcoords='offset points', alpha=0.7)
    
    ax.set_xlabel('Relative X Position (km)')
    ax.set_ylabel('Relative Y Position (km)')
    ax.set_title('Multi-Depot Delivery Network Topology (3 Depots)')
    
    # Custom legend
    depot_markers = [plt.scatter([], [], c=DEPOT_COLORS[i], s=200, marker='s',
                                 edgecolors='black', linewidth=2)
                    for i in range(3)]
    customer_markers = [plt.scatter([], [], c=DEPOT_COLORS[i], s=60,
                                    edgecolors='black', linewidth=0.5)
                       for i in range(3)]
    
    legend_elements = [
        *[Line2D([0], [0], marker='s', color='w', markerfacecolor=DEPOT_COLORS[i],
                markersize=12, markeredgecolor='black', label=f'Depot {i+1}')
         for i in range(3)],
        *[Line2D([0], [0], marker='o', color='w', markerfacecolor=DEPOT_COLORS[i],
                markersize=8, markeredgecolor='black', label=f'Region {i+1} Customers')
         for i in range(3)]
    ]
    ax.legend(handles=legend_elements, loc='upper right', ncol=2)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig1_multi_depot_topology.png'))
    plt.savefig(os.path.join(PLOTS_DIR, 'fig1_multi_depot_topology.pdf'))
    plt.close()
    print("Generated: fig1_multi_depot_topology")

# ============================================================================
# Plot 2: Regional Customer Distribution
# ============================================================================

def plot_region_distribution(region_data):
    """Plot customer distribution across depots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Customer count per region
    ax1 = axes[0]
    regions = list(region_data.keys())
    customer_counts = [len(df) - 1 for df in region_data.values()]  # -1 for depot
    
    bars = ax1.bar(regions, customer_counts, color=DEPOT_COLORS, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, count in zip(bars, customer_counts):
        ax1.annotate(f'{count}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_xlabel('Depot Region')
    ax1.set_ylabel('Number of Customers')
    ax1.set_title('(a) Customer Distribution by Depot')
    ax1.set_xticks(regions)
    ax1.set_xticklabels([f'Depot {r}' for r in regions])
    
    # Pie chart
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(customer_counts, labels=[f'Depot {r}' for r in regions],
                                        colors=DEPOT_COLORS, autopct='%1.1f%%',
                                        startangle=90, explode=[0.02]*3)
    ax2.set_title('(b) Customer Share by Depot')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig2_region_distribution.png'))
    plt.savefig(os.path.join(PLOTS_DIR, 'fig2_region_distribution.pdf'))
    plt.close()
    print("Generated: fig2_region_distribution")

# ============================================================================
# Plot 3: Multi-Depot Cost Comparison
# ============================================================================

def plot_multi_depot_cost_comparison(results):
    """Compare costs across different configurations."""
    if not results:
        print("No results available for cost comparison plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Per-depot costs
    ax1 = axes[0]
    
    # Use first result for detailed breakdown
    result = results[0]
    regions = [r['region_id'] for r in result['regions']]
    objectives = [r['objective'] for r in result['regions']]
    distances = [r['total_distance'] for r in result['regions']]
    
    x = np.arange(len(regions))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, objectives, width, label='Objective Value ($)',
                   color='steelblue', edgecolor='black')
    bars2 = ax1.bar(x + width/2, distances, width, label='Total Distance (mi)',
                   color='#F18F01', edgecolor='black')
    
    ax1.set_xlabel('Depot')
    ax1.set_ylabel('Value')
    ax1.set_title('(a) Cost and Distance by Depot')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Depot {r}' for r in regions])
    ax1.legend()
    
    # Configuration comparison
    ax2 = axes[1]
    
    configs = []
    total_costs = []
    
    for r in results:
        config = r['config']
        label = f"V={config['num_vehicles_per_depot']}, EV={config['num_ev_per_depot']}"
        configs.append(label)
        total_costs.append(r['summary']['total_objective'])
    
    bars = ax2.bar(range(len(configs)), total_costs, color='steelblue',
                  edgecolor='black', linewidth=1.5)
    
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Total Objective Value ($)')
    ax2.set_title('(b) Total Cost by Fleet Configuration')
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    
    # Add value labels
    for bar, cost in zip(bars, total_costs):
        ax2.annotate(f'${cost}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig3_multi_depot_cost_comparison.png'))
    plt.savefig(os.path.join(PLOTS_DIR, 'fig3_multi_depot_cost_comparison.pdf'))
    plt.close()
    print("Generated: fig3_multi_depot_cost_comparison")

# ============================================================================
# Plot 4: Route Distribution by Depot
# ============================================================================

def plot_route_distribution_by_depot(results):
    """Plot route distance and cost distribution by depot."""
    if not results:
        print("No results available for route distribution plot")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    result = results[0]  # Use first result
    
    for idx, region_result in enumerate(result['regions']):
        ax = axes[idx]
        region_id = region_result['region_id']
        
        # Get non-empty routes
        distances = [r['distance'] for r in region_result['routes'] if r['distance'] > 0]
        costs = [r['cost'] for r in region_result['routes'] if r['distance'] > 0]
        vehicle_types = [r['vehicle_type'] for r in region_result['routes'] if r['distance'] > 0]
        
        if distances:
            colors = ['#27ae60' if vt == 'EV' else '#e74c3c' for vt in vehicle_types]
            
            x = np.arange(len(distances))
            ax.bar(x, distances, color=colors, edgecolor='black', alpha=0.8)
            
            ax.set_xlabel('Vehicle')
            ax.set_ylabel('Route Distance (miles)')
            ax.set_title(f'Depot {region_id}\n({len(distances)} active vehicles)')
            ax.axhline(y=np.mean(distances), color='blue', linestyle='--',
                      label=f'Mean: {np.mean(distances):.1f} mi')
            ax.legend()
    
    # Add overall legend
    fig.legend([mpatches.Patch(color='#27ae60'), mpatches.Patch(color='#e74c3c')],
              ['EV', 'GV'], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig4_route_distribution_by_depot.png'))
    plt.savefig(os.path.join(PLOTS_DIR, 'fig4_route_distribution_by_depot.pdf'))
    plt.close()
    print("Generated: fig4_route_distribution_by_depot")

# ============================================================================
# Plot 5: Vehicle Utilization Heatmap
# ============================================================================

def plot_vehicle_utilization_heatmap(results):
    """Create heatmap of vehicle utilization across depots."""
    if not results:
        print("No results available for utilization heatmap")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    result = results[0]
    max_vehicles = max(len(r['routes']) for r in result['regions'])
    
    # Create utilization matrix
    utilization_matrix = np.zeros((3, max_vehicles))
    
    for region_result in result['regions']:
        region_idx = region_result['region_id'] - 1
        for route in region_result['routes']:
            vehicle_idx = route['vehicle_id']
            # Utilization = distance / max_range
            max_range = 200 if route['vehicle_type'] == 'EV' else 300
            utilization = min(route['distance'] / max_range * 100, 100)
            utilization_matrix[region_idx, vehicle_idx] = utilization
    
    im = ax.imshow(utilization_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Labels
    ax.set_yticks(range(3))
    ax.set_yticklabels([f'Depot {i+1}' for i in range(3)])
    ax.set_xticks(range(max_vehicles))
    ax.set_xticklabels([f'V{i}' for i in range(max_vehicles)])
    ax.set_xlabel('Vehicle ID')
    ax.set_ylabel('Depot')
    ax.set_title('Vehicle Range Utilization (%) by Depot')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Range Utilization (%)')
    
    # Add text annotations
    for i in range(3):
        for j in range(max_vehicles):
            val = utilization_matrix[i, j]
            if val > 0:
                color = 'white' if val > 60 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                       color=color, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig5_vehicle_utilization_heatmap.png'))
    plt.savefig(os.path.join(PLOTS_DIR, 'fig5_vehicle_utilization_heatmap.pdf'))
    plt.close()
    print("Generated: fig5_vehicle_utilization_heatmap")

# ============================================================================
# Plot 6: EV vs GV Fleet Composition Analysis
# ============================================================================

def plot_fleet_composition_analysis(results):
    """Analyze impact of fleet composition on costs."""
    if not results:
        print("No results available for fleet composition analysis")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract data
    ev_counts = []
    total_objectives = []
    total_distances = []
    
    for r in results:
        ev_counts.append(r['config']['num_ev_per_depot'] * 3)  # Total EVs across depots
        total_objectives.append(r['summary']['total_objective'])
        total_distances.append(r['summary']['total_distance'])
    
    # Sort by EV count
    sorted_data = sorted(zip(ev_counts, total_objectives, total_distances))
    ev_counts, total_objectives, total_distances = zip(*sorted_data) if sorted_data else ([], [], [])
    
    # Cost vs EV count
    ax1 = axes[0]
    if ev_counts:
        ax1.plot(ev_counts, total_objectives, 'o-', color='steelblue',
                linewidth=2, markersize=10, markeredgecolor='black')
        ax1.set_xlabel('Total Number of EVs (across all depots)')
        ax1.set_ylabel('Total Objective Value ($)')
        ax1.set_title('(a) Impact of EV Adoption on Total Cost')
    
    # Cost breakdown by vehicle type
    ax2 = axes[1]
    
    if results:
        result = results[0]
        ev_distance = 0
        gv_distance = 0
        ev_cost = 0
        gv_cost = 0
        
        for region_result in result['regions']:
            for route in region_result['routes']:
                if route['vehicle_type'] == 'EV':
                    ev_distance += route['distance']
                    ev_cost += route['cost']
                else:
                    gv_distance += route['distance']
                    gv_cost += route['cost']
        
        categories = ['Distance (mi)', 'Cost ($)']
        ev_values = [ev_distance, ev_cost]
        gv_values = [gv_distance, gv_cost]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax2.bar(x - width/2, ev_values, width, label='Electric Vehicles',
               color='#27ae60', edgecolor='black')
        ax2.bar(x + width/2, gv_values, width, label='Gasoline Vehicles',
               color='#e74c3c', edgecolor='black')
        
        ax2.set_ylabel('Value')
        ax2.set_title('(b) Distance and Cost by Vehicle Type')
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories)
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig6_fleet_composition_analysis.png'))
    plt.savefig(os.path.join(PLOTS_DIR, 'fig6_fleet_composition_analysis.pdf'))
    plt.close()
    print("Generated: fig6_fleet_composition_analysis")

# ============================================================================
# Plot 7: Optimized Routes Visualization (Multi-Depot)
# ============================================================================

def plot_multi_depot_routes(all_nodes, region_data, results):
    """Visualize optimized routes for all depots."""
    if not results:
        print("No results available for route visualization")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    x_mean = all_nodes['X'].mean()
    y_mean = all_nodes['Y'].mean()
    
    result = results[0]
    
    for idx, region_result in enumerate(result['regions']):
        ax = axes[idx]
        region_id = region_result['region_id']
        
        if region_id not in region_data:
            continue
            
        df = region_data[region_id]
        x = (df['X'] - x_mean) / 1000
        y = (df['Y'] - y_mean) / 1000
        coords = list(zip(x, y))
        
        # Get active routes
        active_routes = [r for r in region_result['routes'] if r['distance'] > 0]
        n_routes = len(active_routes)
        
        # Color palette for routes
        route_colors = plt.cm.tab10(np.linspace(0, 1, max(n_routes, 1)))
        
        # Plot routes
        for route_idx, route_data in enumerate(active_routes):
            route = route_data['route']
            color = route_colors[route_idx]
            
            for i in range(len(route) - 1):
                if route[i] < len(coords) and route[i+1] < len(coords):
                    x1, y1 = coords[route[i]]
                    x2, y2 = coords[route[i+1]]
                    ax.plot([x1, x2], [y1, y2], color=color, linewidth=1.5, alpha=0.7)
        
        # Plot nodes
        ax.scatter(x[1:], y[1:], c='white', s=80, edgecolors='black',
                  linewidth=1, zorder=5)
        
        # Plot depot
        depot_color = DEPOT_COLORS[region_id - 1]
        ax.scatter(x.iloc[0], y.iloc[0], c=depot_color, s=200, marker='s',
                  edgecolors='black', linewidth=2, zorder=6)
        
        # Labels
        for i, (xi, yi) in enumerate(coords):
            ax.annotate(str(i), (xi, yi), fontsize=7, ha='center', va='center',
                       fontweight='bold', zorder=7)
        
        ax.set_xlabel('X Position (km)')
        ax.set_ylabel('Y Position (km)')
        ax.set_title(f'Depot {region_id} Routes\n'
                    f'({n_routes} vehicles, ${region_result["objective"]} cost)')
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig7_multi_depot_routes.png'))
    plt.savefig(os.path.join(PLOTS_DIR, 'fig7_multi_depot_routes.pdf'))
    plt.close()
    print("Generated: fig7_multi_depot_routes")

# ============================================================================
# Plot 8: Summary Statistics Dashboard
# ============================================================================

def plot_summary_dashboard(results):
    """Create a summary dashboard with key metrics."""
    if not results:
        print("No results available for summary dashboard")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    result = results[0]
    
    # (a) Total metrics comparison
    ax1 = axes[0, 0]
    metrics = ['Objective ($)', 'Distance (mi)', 'Active Vehicles']
    values = [
        result['summary']['total_objective'],
        result['summary']['total_distance'],
        result['summary']['total_active_vehicles']
    ]
    
    colors = ['steelblue', '#F18F01', '#27ae60']
    bars = ax1.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_title('(a) Overall Performance Metrics')
    ax1.set_ylabel('Value')
    
    for bar, val in zip(bars, values):
        ax1.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # (b) Per-depot breakdown
    ax2 = axes[0, 1]
    depots = [f'Depot {r["region_id"]}' for r in result['regions']]
    objectives = [r['objective'] for r in result['regions']]
    
    ax2.barh(depots, objectives, color=DEPOT_COLORS, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Objective Value ($)')
    ax2.set_title('(b) Cost by Depot')
    
    for i, v in enumerate(objectives):
        ax2.text(v + 20, i, f'${v}', va='center', fontsize=10)
    
    # (c) Workload balance
    ax3 = axes[1, 0]
    
    stops_per_depot = []
    for region_result in result['regions']:
        total_stops = sum(r['num_stops'] for r in region_result['routes'])
        stops_per_depot.append(total_stops)
    
    ax3.pie(stops_per_depot, labels=[f'Depot {i+1}' for i in range(3)],
           colors=DEPOT_COLORS, autopct='%1.1f%%', startangle=90)
    ax3.set_title('(c) Workload Distribution (by stops)')
    
    # (d) Distance per vehicle
    ax4 = axes[1, 1]
    
    all_distances = []
    all_labels = []
    all_colors = []
    
    for region_result in result['regions']:
        region_id = region_result['region_id']
        for route in region_result['routes']:
            if route['distance'] > 0:
                all_distances.append(route['distance'])
                all_labels.append(f'D{region_id}-V{route["vehicle_id"]}')
                all_colors.append(DEPOT_COLORS[region_id - 1])
    
    if all_distances:
        ax4.bar(range(len(all_distances)), all_distances, color=all_colors,
               edgecolor='black', alpha=0.8)
        ax4.set_xlabel('Vehicle')
        ax4.set_ylabel('Distance (miles)')
        ax4.set_title('(d) Route Distance per Active Vehicle')
        ax4.axhline(y=np.mean(all_distances), color='red', linestyle='--',
                   label=f'Mean: {np.mean(all_distances):.1f} mi')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig8_summary_dashboard.png'))
    plt.savefig(os.path.join(PLOTS_DIR, 'fig8_summary_dashboard.pdf'))
    plt.close()
    print("Generated: fig8_summary_dashboard")

# ============================================================================
# Plot 9: Sensitivity Analysis - Multi-Depot
# ============================================================================

def plot_sensitivity_analysis():
    """Plot sensitivity analysis for multi-depot scenario."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create synthetic sensitivity data for 3-depot scenario
    ev_cost_range = np.arange(0.20, 0.45, 0.05)
    gv_cost_range = np.arange(0.40, 0.80, 0.10)
    
    # Base assumptions (3 depots)
    base_ev_distance_per_depot = 200
    base_gv_distance_per_depot = 300
    base_labor_cost_per_depot = 800
    
    cost_matrix = np.zeros((len(gv_cost_range), len(ev_cost_range)))
    
    for i, gv_c in enumerate(gv_cost_range):
        for j, ev_c in enumerate(ev_cost_range):
            total_cost = 3 * (base_ev_distance_per_depot * ev_c + 
                            base_gv_distance_per_depot * gv_c + 
                            base_labor_cost_per_depot)
            cost_matrix[i, j] = total_cost
    
    im = ax.imshow(cost_matrix, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(np.arange(len(ev_cost_range)))
    ax.set_yticks(np.arange(len(gv_cost_range)))
    ax.set_xticklabels([f'${x:.2f}' for x in ev_cost_range])
    ax.set_yticklabels([f'${y:.2f}' for y in gv_cost_range])
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Total System Cost ($)')
    
    for i in range(len(gv_cost_range)):
        for j in range(len(ev_cost_range)):
            ax.text(j, i, f'{cost_matrix[i, j]:.0f}',
                   ha='center', va='center', fontsize=9)
    
    ax.set_xlabel('EV Cost per Mile')
    ax.set_ylabel('GV Cost per Mile')
    ax.set_title('Sensitivity Analysis: Multi-Depot Total Operating Cost')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig9_sensitivity_analysis.png'))
    plt.savefig(os.path.join(PLOTS_DIR, 'fig9_sensitivity_analysis.pdf'))
    plt.close()
    print("Generated: fig9_sensitivity_analysis")

# ============================================================================
# Plot 10: Depot Comparison Radar Chart
# ============================================================================

def plot_depot_comparison_radar(results):
    """Create radar chart comparing depot performance."""
    if not results:
        print("No results available for radar chart")
        return
    
    result = results[0]
    
    # Metrics for comparison
    categories = ['Objective\nValue', 'Total\nDistance', 'Active\nVehicles', 
                 'Avg Stops\nper Vehicle', 'Efficiency\n(stops/mi)']
    
    # Calculate metrics for each depot
    depot_metrics = []
    for region_result in result['regions']:
        active_routes = [r for r in region_result['routes'] if r['distance'] > 0]
        total_stops = sum(r['num_stops'] for r in active_routes)
        total_distance = region_result['total_distance']
        
        metrics = [
            region_result['objective'] / 1000,  # Scale for visualization
            total_distance / 100,
            len(active_routes),
            total_stops / len(active_routes) if active_routes else 0,
            (total_stops / total_distance * 10) if total_distance > 0 else 0
        ]
        depot_metrics.append(metrics)
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for i, metrics in enumerate(depot_metrics):
        values = metrics + metrics[:1]
        ax.plot(angles, values, 'o-', linewidth=2, color=DEPOT_COLORS[i],
               label=f'Depot {i+1}')
        ax.fill(angles, values, alpha=0.25, color=DEPOT_COLORS[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title('Multi-Depot Performance Comparison', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig10_depot_radar_comparison.png'))
    plt.savefig(os.path.join(PLOTS_DIR, 'fig10_depot_radar_comparison.pdf'))
    plt.close()
    print("Generated: fig10_depot_radar_comparison")

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Generate all plots for the multi-depot research paper."""
    print("=" * 70)
    print("Generating Multi-Depot Academic Research Plots")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    all_nodes = load_all_nodes()
    region_data = load_region_data()
    results = load_multi_depot_results()
    
    print(f"Loaded {len(all_nodes)} total nodes")
    print(f"Loaded {len(region_data)} region datasets")
    print(f"Loaded {len(results)} optimization results")
    
    # Generate all plots
    print("\nGenerating plots...\n")
    
    if region_data:
        plot_multi_depot_topology(all_nodes, region_data)
        plot_region_distribution(region_data)
    else:
        print("Warning: Region data not available. Run main_MultiDepot.py first.")
    
    if results:
        plot_multi_depot_cost_comparison(results)
        plot_route_distribution_by_depot(results)
        plot_vehicle_utilization_heatmap(results)
        plot_fleet_composition_analysis(results)
        if region_data:
            plot_multi_depot_routes(all_nodes, region_data, results)
        plot_summary_dashboard(results)
        plot_depot_comparison_radar(results)
    else:
        print("Warning: No optimization results found. Run main_MultiDepot.py first.")
    
    plot_sensitivity_analysis()
    
    print("\n" + "=" * 70)
    print(f"All plots saved to '{PLOTS_DIR}/' directory")
    print("Formats: PNG (300 DPI) and PDF (vector)")
    print("=" * 70)

if __name__ == '__main__':
    main()
