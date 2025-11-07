"""
Generate Figure 2: Convex Optimization Pipeline Flowchart
Shows the step-by-step process of Convex-PSRL
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np


def create_figure2(save_path: str = '../figures/figure2.pdf'):
    """
    Create flowchart showing the Convex-PSRL pipeline:
    1. Collect data (trajectories)
    2. Formulate matrix X and targets Y
    3. Set up convex dual program
    4. Solve convex QP → obtain weights w*
    5. Use w* as model (policy) for RL
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_title('Convex-PSRL Optimization Pipeline', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Define box positions and sizes
    box_width = 8
    box_height = 1.5
    box_x = 2
    y_positions = [12, 9.5, 7, 4.5, 2]
    
    # Colors for different stages
    colors = ['#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#27ae60']
    
    # Step 1: Collect Data
    box1 = FancyBboxPatch((box_x, y_positions[0]), box_width, box_height,
                         boxstyle="round,pad=0.15",
                         facecolor=colors[0], edgecolor='#2874a6', linewidth=2.5)
    ax.add_patch(box1)
    ax.text(box_x + box_width/2, y_positions[0] + box_height/2 + 0.3,
           '1. Collect Data (Trajectories)', ha='center', va='center',
           fontsize=13, fontweight='bold', color='white')
    ax.text(box_x + box_width/2, y_positions[0] + box_height/2 - 0.3,
           'Interact with environment: (sₜ, aₜ, rₜ, sₜ₊₁)',
           ha='center', va='center', fontsize=10, style='italic', color='white')
    
    # Arrow 1
    arrow1 = FancyArrowPatch((box_x + box_width/2, y_positions[0]),
                            (box_x + box_width/2, y_positions[1] + box_height),
                            arrowstyle='->', mutation_scale=25, linewidth=3,
                            color='#34495e')
    ax.add_patch(arrow1)
    
    # Step 2: Formulate Matrices
    box2 = FancyBboxPatch((box_x, y_positions[1]), box_width, box_height,
                         boxstyle="round,pad=0.15",
                         facecolor=colors[1], edgecolor='#7d3c98', linewidth=2.5)
    ax.add_patch(box2)
    ax.text(box_x + box_width/2, y_positions[1] + box_height/2 + 0.3,
           '2. Formulate Matrix X and Targets Y', ha='center', va='center',
           fontsize=13, fontweight='bold', color='white')
    ax.text(box_x + box_width/2, y_positions[1] + box_height/2 - 0.3,
           'X = [s₁, a₁; s₂, a₂; ...],  Y = [s₂; s₃; ...]',
           ha='center', va='center', fontsize=10, style='italic', color='white')
    
    # Arrow 2
    arrow2 = FancyArrowPatch((box_x + box_width/2, y_positions[1]),
                            (box_x + box_width/2, y_positions[2] + box_height),
                            arrowstyle='->', mutation_scale=25, linewidth=3,
                            color='#34495e')
    ax.add_patch(arrow2)
    
    # Step 3: Set up Convex Dual Program
    box3 = FancyBboxPatch((box_x, y_positions[2]), box_width, box_height,
                         boxstyle="round,pad=0.15",
                         facecolor=colors[2], edgecolor='#c0392b', linewidth=2.5)
    ax.add_patch(box3)
    ax.text(box_x + box_width/2, y_positions[2] + box_height/2 + 0.4,
           '3. Set Up Convex Dual Program', ha='center', va='center',
           fontsize=13, fontweight='bold', color='white')
    ax.text(box_x + box_width/2, y_positions[2] + box_height/2 - 0.1,
           'Reformulate 2-layer ReLU as convex QP',
           ha='center', va='center', fontsize=10, style='italic', color='white')
    ax.text(box_x + box_width/2, y_positions[2] + box_height/2 - 0.5,
           'min ||Y - h(X; W)||² + λ||W||²  (see Eq. in paper)',
           ha='center', va='center', fontsize=9, family='monospace', color='white')
    
    # Arrow 3
    arrow3 = FancyArrowPatch((box_x + box_width/2, y_positions[2]),
                            (box_x + box_width/2, y_positions[3] + box_height),
                            arrowstyle='->', mutation_scale=25, linewidth=3,
                            color='#34495e')
    ax.add_patch(arrow3)
    
    # Step 4: Solve Convex QP
    box4 = FancyBboxPatch((box_x, y_positions[3]), box_width, box_height,
                         boxstyle="round,pad=0.15",
                         facecolor=colors[3], edgecolor='#d68910', linewidth=2.5)
    ax.add_patch(box4)
    ax.text(box_x + box_width/2, y_positions[3] + box_height/2 + 0.3,
           '4. Solve Convex QP → Obtain Weights w*', ha='center', va='center',
           fontsize=13, fontweight='bold', color='white')
    ax.text(box_x + box_width/2, y_positions[3] + box_height/2 - 0.3,
           'Use CVXPY/OSQP solver (tractable, polynomial time)',
           ha='center', va='center', fontsize=10, style='italic', color='white')
    
    # Arrow 4
    arrow4 = FancyArrowPatch((box_x + box_width/2, y_positions[3]),
                            (box_x + box_width/2, y_positions[4] + box_height),
                            arrowstyle='->', mutation_scale=25, linewidth=3,
                            color='#34495e')
    ax.add_patch(arrow4)
    
    # Step 5: Use Model for RL
    box5 = FancyBboxPatch((box_x, y_positions[4]), box_width, box_height,
                         boxstyle="round,pad=0.15",
                         facecolor=colors[4], edgecolor='#1e8449', linewidth=2.5)
    ax.add_patch(box5)
    ax.text(box_x + box_width/2, y_positions[4] + box_height/2 + 0.3,
           '5. Use w* as Dynamics Model for RL Policy', ha='center', va='center',
           fontsize=13, fontweight='bold', color='white')
    ax.text(box_x + box_width/2, y_positions[4] + box_height/2 - 0.3,
           'Plan actions via MPC, collect new data, repeat',
           ha='center', va='center', fontsize=10, style='italic', color='white')
    
    # Feedback loop
    feedback_arrow = FancyArrowPatch((box_x + box_width, y_positions[4] + box_height/2),
                                    (box_x + box_width + 1, y_positions[4] + box_height/2),
                                    arrowstyle='->', mutation_scale=20, linewidth=2.5,
                                    color='#16a085', linestyle='dashed')
    ax.add_patch(feedback_arrow)
    
    feedback_arrow2 = FancyArrowPatch((box_x + box_width + 1, y_positions[4] + box_height/2),
                                     (box_x + box_width + 1, y_positions[0] + box_height/2),
                                     arrowstyle='->', mutation_scale=20, linewidth=2.5,
                                     color='#16a085', linestyle='dashed')
    ax.add_patch(feedback_arrow2)
    
    feedback_arrow3 = FancyArrowPatch((box_x + box_width + 1, y_positions[0] + box_height/2),
                                     (box_x + box_width, y_positions[0] + box_height/2),
                                     arrowstyle='->', mutation_scale=20, linewidth=2.5,
                                     color='#16a085', linestyle='dashed')
    ax.add_patch(feedback_arrow3)
    
    ax.text(box_x + box_width + 1.3, y_positions[2] + 2,
           'Iterate:\nPosterior\nSampling\nLoop',
           ha='left', va='center', fontsize=11, fontweight='bold',
           color='#16a085', style='italic')
    
    # Key advantages box
    adv_box = FancyBboxPatch((0.5, 0.2), 11, 1.2,
                            boxstyle="round,pad=0.15",
                            facecolor='#ecf0f1', edgecolor='#95a5a6', linewidth=2)
    ax.add_patch(adv_box)
    ax.text(6, 1.1, 'Key Advantages:', ha='center', va='center',
           fontsize=12, fontweight='bold', color='#2c3e50')
    ax.text(6, 0.7, '✓ Exact MAP inference (no MCMC)  |  ✓ Polynomial-time complexity  |  '
           '✓ Provable convergence  |  ✓ Sample efficient',
           ha='center', va='center', fontsize=10, color='#2c3e50')
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Figure 2 saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    # Ensure figures directory exists
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    save_path = figures_dir / 'figure2.pdf'
    create_figure2(str(save_path))
