# Multi-Depot Vehicle Routing Problem with Heterogeneous Fleet

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![OR-Tools](https://img.shields.io/badge/OR--Tools-9.0+-green.svg)](https://developers.google.com/optimization)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Abstract

This repository presents a computational framework for solving the **Multi-Depot Heterogeneous Fleet Vehicle Routing Problem (MDHFVRP)** in last-mile delivery logistics. The system optimizes delivery routes across $K = 3$ distribution centers serving $N = 69$ geographically dispersed customers using a mixed fleet of Electric Vehicles (EVs) and Gasoline Vehicles (GVs).

📄 **[Download Full Technical Report (PDF)](docs/Technical_Report.pdf)**

---

## Mathematical Formulation

### Sets and Indices

| Symbol | Description |
|--------|-------------|
| $\mathcal{D} = \{1, 2, 3\}$ | Set of depots |
| $\mathcal{C}_k$ | Set of customers assigned to depot $k \in \mathcal{D}$ |
| $\mathcal{V}_k$ | Set of vehicles at depot $k$ |
| $\mathcal{V}_k^{EV} \subseteq \mathcal{V}_k$ | Subset of electric vehicles |
| $\mathcal{V}_k^{GV} \subseteq \mathcal{V}_k$ | Subset of gasoline vehicles |

### Parameters

| Parameter | Description | Value |
|-----------|-------------|-------|
| $d_{ij}$ | Distance from node $i$ to node $j$ | miles |
| $t_{ij}$ | Travel time from node $i$ to node $j$ | minutes |
| $Q^{EV}$ | Maximum range for EVs | 200 miles |
| $Q^{GV}$ | Maximum range for GVs | 300 miles |
| $T_{max}$ | Maximum working time | 540 minutes |
| $c^{EV}$ | EV operating cost | $0.25-0.30/mile |
| $c^{GV}$ | GV operating cost | $0.50-0.60/mile |
| $c^{L}$ | Labor cost | $0.60/minute |

### Decision Variables

```math
x_{ij}^{kv} = \begin{cases} 1 & \text{if vehicle } v \text{ from depot } k \text{ traverses arc } (i,j) \\ 0 & \text{otherwise} \end{cases}
```

### Objective Function

Minimize total operational cost across all depots:

```math
\min Z = \sum_{k \in \mathcal{D}} \sum_{v \in \mathcal{V}_k} \sum_{(i,j) \in \mathcal{A}} \left( c_v \cdot d_{ij} + c^{L} \cdot t_{ij} \right) x_{ij}^{kv}
```

where the vehicle-specific cost coefficient is:

```math
c_v = \begin{cases} c^{EV} & \text{if } v \in \mathcal{V}_k^{EV} \\ c^{GV} & \text{if } v \in \mathcal{V}_k^{GV} \end{cases}
```

### Constraints

**1. Customer Visit Constraint** — Each customer visited exactly once:
```math
\sum_{v \in \mathcal{V}_k} \sum_{i \in \mathcal{C}_k \cup \{0\}} x_{ij}^{kv} = 1 \quad \forall j \in \mathcal{C}_k, \forall k \in \mathcal{D}
```

**2. Flow Conservation**:
```math
\sum_{i} x_{ij}^{kv} = \sum_{i} x_{ji}^{kv} \quad \forall j, \forall v \in \mathcal{V}_k, \forall k \in \mathcal{D}
```

**3. Vehicle Range Constraint**:
```math
\sum_{(i,j) \in \mathcal{A}} d_{ij} \cdot x_{ij}^{kv} \leq Q_v \quad \forall v \in \mathcal{V}_k, \forall k \in \mathcal{D}
```

**4. Working Time Constraint**:
```math
\sum_{(i,j) \in \mathcal{A}} t_{ij} \cdot x_{ij}^{kv} \leq T_{max} \quad \forall v \in \mathcal{V}_k, \forall k \in \mathcal{D}
```

---

## Customer Assignment via K-Means Clustering

Customers are partitioned into $K$ regions by minimizing within-cluster variance:

```math
\min \sum_{k=1}^{K} \sum_{i \in \mathcal{C}_k} \| \mathbf{p}_i - \boldsymbol{\mu}_k \|^2
```

where $\mathbf{p}_i = (x_i, y_i)$ denotes customer coordinates and $\boldsymbol{\mu}_k$ is the centroid of cluster $k$.

---

## Distance and Time Computation

**Travel Distance:**
```math
d_{ij} = \underbrace{2 \times 0.621 \times 10^{-3} \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}}_{\text{Inter-node distance}} + \underbrace{\delta_i \cdot \rho}_{\text{Intra-node distance}}
```

**Travel Time:**
```math
t_{ij} = \frac{d_{ij}^{inter}}{v_{inter}} \times 60 + \frac{d_{ij}^{intra}}{v_{intra}} \times 60 + \delta_i \cdot s
```

where $v_{inter} = 45$ mph, $v_{intra} = 25$ mph, and $s = 1.5$ min/drop.

---

## Computational Results

### Configuration Comparison

| Configuration | EVs | GVs | Total Cost ($) | Distance (mi) | Active Vehicles |
|--------------|-----|-----|----------------|---------------|-----------------|
| All GV | 0 | 30 | **6,568** | 2,908 | 18 |
| Mixed (30% EV) | 9 | 21 | **6,200** | 2,931 | 19 |
| Mixed (50% EV) | 15 | 15 | **5,971** | 2,939 | 19 |
| Higher costs | 9 | 21 | **6,394** | 2,918 | 18 |

### Cost Reduction from Fleet Electrification

```math
\Delta Z = \frac{Z_{GV} - Z_{mixed}}{Z_{GV}} \times 100\% = \frac{6568 - 5971}{6568} \times 100\% \approx \mathbf{9.1\%}
```

### Per-Depot Performance

| Depot | Customers | Cost ($) | Avg Distance (mi) | Utilization |
|-------|-----------|----------|-------------------|-------------|
| 1 | 30 | 2,219 | 167.3 | 55.8% |
| 2 | 23 | 1,964 | 156.0 | 52.0% |
| 3 | 16 | 1,788 | 138.7 | 46.2% |

**Average Vehicle Utilization:**
```math
\bar{U} = \frac{1}{|\mathcal{V}^{active}|} \sum_{v \in \mathcal{V}^{active}} \frac{d_v}{Q_v} = 51.3\%
```

---

## Solution Methodology

The problem is solved using a **decomposition approach**:

1. **Initial Solution**: PATH_CHEAPEST_ARC heuristic
2. **Improvement**: Guided Local Search (GLS) metaheuristic
3. **Time Limit**: τ = 5 seconds per depot

**GLS Penalization:**
```math
c'_{ij} = c_{ij} + \lambda \cdot p_{ij}
```

where $p_{ij}$ is the penalty counter and $\lambda$ is the penalty weight.

---

## Project Structure

```
├── Multi Vehicle Last Mile Delivery Optimization/
│   ├── main_MultiDepot.py          # Main multi-depot optimizer
│   ├── Input_MultiDepot.py         # Data processing & clustering
│   ├── Parameters_MultiDepot.py    # Configuration parameters
│   ├── generate_plots_MultiDepot.py # Academic visualization
│   ├── main.py                     # Single-depot version
│   ├── Input.py                    # Original input module
│   ├── Parameters.py               # Original parameters
│   ├── nodes.csv                   # Customer coordinates
│   ├── Output_MultiDepot/          # Multi-depot results (JSON/TXT)
│   ├── Output1/                    # Single-depot results
│   └── plots/                      # Generated figures (PNG/PDF)
├── docs/
│   └── Technical_Report.pdf        # Full technical documentation
├── LICENSE
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Last-Mile-Delivery-Optimization.git
cd Last-Mile-Delivery-Optimization

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn ortools
```

---

## Usage

### Run Multi-Depot Optimization
```bash
cd "Multi Vehicle Last Mile Delivery Optimization"
python main_MultiDepot.py
```

### Generate Academic Plots
```bash
python generate_plots_MultiDepot.py
```

---

## Generated Figures

| Figure | Description |
|--------|-------------|
| `fig1_multi_depot_topology.png` | 3-depot network topology |
| `fig2_region_distribution.png` | Customer distribution by depot |
| `fig3_multi_depot_cost_comparison.png` | Cost analysis across configurations |
| `fig4_route_distribution_by_depot.png` | Route distance distributions |
| `fig5_vehicle_utilization_heatmap.png` | Vehicle range utilization matrix |
| `fig6_fleet_composition_analysis.png` | EV vs GV performance comparison |
| `fig7_multi_depot_routes.png` | Optimized route visualization |
| `fig8_summary_dashboard.png` | Key performance metrics |
| `fig9_sensitivity_analysis.png` | Cost parameter sensitivity |
| `fig10_depot_radar_comparison.png` | Multi-dimensional depot comparison |

---

## Key Findings

1. **Fleet electrification reduces costs by 9.1%** while maintaining service levels
2. **K-means clustering** effectively partitions customers for multi-depot operations
3. **Average vehicle utilization of 51.3%** indicates potential for fleet size optimization
4. **Guided Local Search** achieves near-optimal solutions within 5 seconds per depot

---

## Dependencies

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Google OR-Tools

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mdhfvrp2026,
  title={Multi-Depot Vehicle Routing Problem with Heterogeneous Fleet},
  author={[Author Name]},
  year={2026},
  url={https://github.com/yourusername/Last-Mile-Delivery-Optimization}
}
```

---

## Contact

For questions or collaboration opportunities, please open an issue or contact the repository maintainer.