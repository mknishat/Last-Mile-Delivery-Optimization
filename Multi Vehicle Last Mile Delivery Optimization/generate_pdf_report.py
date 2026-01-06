"""
Generate PDF Technical Report for Multi-Depot VRP
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
import os

# Create docs directory
if not os.path.exists('docs'):
    os.makedirs('docs')

def create_technical_report():
    """Generate the technical report PDF."""
    
    doc = SimpleDocTemplate(
        "docs/Technical_Report.pdf",
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.darkblue
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceBefore=15,
        spaceAfter=8,
        textColor=colors.darkblue
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        leading=14
    )
    
    equation_style = ParagraphStyle(
        'Equation',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=12,
        alignment=TA_CENTER,
        backColor=colors.Color(0.95, 0.95, 0.95),
        borderPadding=10
    )
    
    # Build document content
    story = []
    
    # Title
    story.append(Paragraph(
        "Multi-Depot Vehicle Routing Problem with<br/>Heterogeneous Fleet",
        title_style
    ))
    story.append(Paragraph(
        "A Mathematical Formulation and Computational Study",
        ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=12, alignment=TA_CENTER, spaceAfter=30)
    ))
    story.append(Spacer(1, 20))
    
    # Abstract
    story.append(Paragraph("Abstract", heading_style))
    story.append(Paragraph(
        """This technical report presents a computational framework for solving the Multi-Depot 
        Heterogeneous Fleet Vehicle Routing Problem (MDHFVRP) in last-mile delivery logistics. 
        The system optimizes delivery routes across K = 3 distribution centers serving N = 69 
        geographically dispersed customers using a mixed fleet of Electric Vehicles (EVs) and 
        Gasoline Vehicles (GVs). Our results demonstrate that strategic fleet electrification 
        achieves a 9.1% cost reduction while maintaining service levels.""",
        body_style
    ))
    story.append(Spacer(1, 20))
    
    # Problem Formulation
    story.append(Paragraph("1. Problem Formulation", heading_style))
    
    story.append(Paragraph("1.1 Sets and Indices", subheading_style))
    
    sets_data = [
        ['Symbol', 'Description'],
        ['D = {1, 2, 3}', 'Set of depots'],
        ['Ck', 'Set of customers assigned to depot k'],
        ['Vk', 'Set of vehicles at depot k'],
        ['VkEV ⊆ Vk', 'Subset of electric vehicles'],
        ['VkGV ⊆ Vk', 'Subset of gasoline vehicles'],
    ]
    
    sets_table = Table(sets_data, colWidths=[2*inch, 4*inch])
    sets_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(sets_table)
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("1.2 Parameters", subheading_style))
    
    params_data = [
        ['Parameter', 'Description', 'Value'],
        ['dij', 'Distance from node i to node j', 'miles'],
        ['tij', 'Travel time from node i to node j', 'minutes'],
        ['QEV', 'Maximum range for EVs', '200 miles'],
        ['QGV', 'Maximum range for GVs', '300 miles'],
        ['Tmax', 'Maximum working time', '540 minutes'],
        ['cEV', 'EV operating cost', '$0.25-0.30/mile'],
        ['cGV', 'GV operating cost', '$0.50-0.60/mile'],
        ['cL', 'Labor cost', '$0.60/minute'],
    ]
    
    params_table = Table(params_data, colWidths=[1.5*inch, 2.5*inch, 1.5*inch])
    params_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(params_table)
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("1.3 Objective Function", subheading_style))
    story.append(Paragraph(
        """The objective is to minimize total operational cost across all depots, comprising 
        distance-based vehicle operating costs and time-based labor costs:""",
        body_style
    ))
    story.append(Paragraph(
        "min Z = Σk∈D Σv∈Vk Σ(i,j)∈A ( cv · dij + cL · tij ) · xijkv",
        equation_style
    ))
    story.append(Paragraph(
        """where xijkv is a binary decision variable equal to 1 if vehicle v from depot k 
        traverses arc (i,j), and cv is the vehicle-specific cost coefficient (cEV for electric 
        vehicles, cGV for gasoline vehicles).""",
        body_style
    ))
    
    story.append(Paragraph("1.4 Constraints", subheading_style))
    story.append(Paragraph(
        """<b>Customer Visit Constraint:</b> Each customer must be visited exactly once by 
        exactly one vehicle from its assigned depot.""",
        body_style
    ))
    story.append(Paragraph(
        """<b>Flow Conservation:</b> For each vehicle, the number of arrivals at a node must 
        equal the number of departures.""",
        body_style
    ))
    story.append(Paragraph(
        """<b>Vehicle Range Constraint:</b> The total distance traveled by each vehicle must 
        not exceed its maximum range (QEV = 200 mi for EVs, QGV = 300 mi for GVs).""",
        body_style
    ))
    story.append(Paragraph(
        """<b>Working Time Constraint:</b> The total time for each vehicle route must not 
        exceed the maximum working time (Tmax = 540 minutes).""",
        body_style
    ))
    
    story.append(PageBreak())
    
    # Customer Assignment
    story.append(Paragraph("2. Customer Assignment via K-Means Clustering", heading_style))
    story.append(Paragraph(
        """Customers are partitioned into K = 3 regions using K-means clustering, which 
        minimizes the within-cluster sum of squared distances:""",
        body_style
    ))
    story.append(Paragraph(
        "min Σk=1K Σi∈Ck || pi - μk ||²",
        equation_style
    ))
    story.append(Paragraph(
        """where pi = (xi, yi) denotes the coordinates of customer i, and μk is the centroid 
        of cluster k. This approach ensures geographically compact service regions for each depot.""",
        body_style
    ))
    
    # Distance Computation
    story.append(Paragraph("3. Distance and Time Computation", heading_style))
    story.append(Paragraph(
        """The travel distance between nodes incorporates both inter-node travel and 
        intra-node service distances:""",
        body_style
    ))
    story.append(Paragraph(
        "dij = 2 × 0.621 × 10⁻³ × √[(xi - xj)² + (yi - yj)²] + δi × ρ",
        equation_style
    ))
    story.append(Paragraph(
        """where δi is the demand at node i and ρ is the distance between drops within a 
        location. Travel time includes inter-node travel, intra-node travel, and service time 
        components.""",
        body_style
    ))
    
    # Computational Results
    story.append(Paragraph("4. Computational Results", heading_style))
    
    story.append(Paragraph("4.1 Configuration Comparison", subheading_style))
    
    results_data = [
        ['Configuration', 'EVs', 'GVs', 'Total Cost ($)', 'Distance (mi)', 'Active Vehicles'],
        ['All GV', '0', '30', '6,568', '2,908', '18'],
        ['Mixed (30% EV)', '9', '21', '6,200', '2,931', '19'],
        ['Mixed (50% EV)', '15', '15', '5,971', '2,939', '19'],
        ['Higher costs', '9', '21', '6,394', '2,918', '18'],
    ]
    
    results_table = Table(results_data, colWidths=[1.3*inch, 0.6*inch, 0.6*inch, 1.1*inch, 1.1*inch, 1.1*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(results_table)
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("4.2 Cost Reduction Analysis", subheading_style))
    story.append(Paragraph(
        """The percentage cost reduction from fleet electrification (50% EV deployment) 
        compared to an all-GV fleet:""",
        body_style
    ))
    story.append(Paragraph(
        "ΔZ = (ZGV - Zmixed) / ZGV × 100% = (6568 - 5971) / 6568 × 100% ≈ 9.1%",
        equation_style
    ))
    
    story.append(Paragraph("4.3 Per-Depot Performance", subheading_style))
    
    depot_data = [
        ['Depot', 'Customers', 'Cost ($)', 'Avg Distance (mi)', 'Utilization'],
        ['1', '30', '2,219', '167.3', '55.8%'],
        ['2', '23', '1,964', '156.0', '52.0%'],
        ['3', '16', '1,788', '138.7', '46.2%'],
    ]
    
    depot_table = Table(depot_data, colWidths=[1*inch, 1.1*inch, 1.1*inch, 1.4*inch, 1.1*inch])
    depot_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(depot_table)
    story.append(Spacer(1, 15))
    
    # Solution Methodology
    story.append(Paragraph("5. Solution Methodology", heading_style))
    story.append(Paragraph(
        """The problem is solved using a decomposition approach where each depot's subproblem 
        is optimized independently. The solution procedure consists of:""",
        body_style
    ))
    story.append(Paragraph(
        """<b>1. Initial Solution Generation:</b> The PATH_CHEAPEST_ARC heuristic constructs 
        an initial feasible solution by iteratively adding the cheapest available arc.""",
        body_style
    ))
    story.append(Paragraph(
        """<b>2. Solution Improvement:</b> Guided Local Search (GLS) metaheuristic improves 
        the solution by penalizing frequently used arcs: c'ij = cij + λ × pij""",
        body_style
    ))
    story.append(Paragraph(
        """<b>3. Termination:</b> The solver terminates after τ = 5 seconds per depot.""",
        body_style
    ))
    
    # Conclusions
    story.append(Paragraph("6. Key Findings", heading_style))
    story.append(Paragraph(
        """<b>1.</b> Fleet electrification (50% EV deployment) reduces total operational 
        costs by approximately 9.1% compared to an all-GV fleet.""",
        body_style
    ))
    story.append(Paragraph(
        """<b>2.</b> K-means clustering effectively partitions customers into geographically 
        compact service regions, enabling efficient multi-depot operations.""",
        body_style
    ))
    story.append(Paragraph(
        """<b>3.</b> Average vehicle range utilization of 51.3% indicates potential for 
        further fleet size optimization.""",
        body_style
    ))
    story.append(Paragraph(
        """<b>4.</b> The Guided Local Search metaheuristic achieves near-optimal solutions 
        within 5 seconds per depot, demonstrating computational efficiency.""",
        body_style
    ))
    
    # Build PDF
    doc.build(story)
    print("Technical Report generated: docs/Technical_Report.pdf")

if __name__ == '__main__':
    create_technical_report()
