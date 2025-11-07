"""
Generate Figure 1: Conceptual Contrast Diagram
Comparing Standard Deep PSRL (MCMC-based) vs Convex-PSRL (convex program)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np


def create_figure1(save_path: str = '../figures/figure1.pdf'):
    """
    Create conceptual contrast diagram showing:
    - Left: Standard Deep PSRL with MCMC (intractable)
    - Right: Convex-PSRL with convex program (tractable)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ============ LEFT: Standard Deep PSRL ============
    ax_left = axes[0]
    ax_left.set_xlim(0, 10)
    ax_left.set_ylim(0, 10)
    ax_left.axis('off')
    ax_left.set_title('Standard Deep PSRL', fontsize=16, fontweight='bold', pad=20)
    
    # Deep neural network icon (multiple layers)
    layer_x = 2
    layer_y_start = 7
    layer_height = 0.5
    layer_width = 1.5
    n_layers = 5
    
    for i in range(n_layers):
        y = layer_y_start - i * 0.8
        rect = FancyBboxPatch((layer_x, y), layer_width, layer_height,
                             boxstyle="round,pad=0.05", 
                             facecolor='#3498db', edgecolor='#2874a6', linewidth=2)
        ax_left.add_patch(rect)
        
        if i < n_layers - 1:
            # Connection lines
            ax_left.plot([layer_x + layer_width/2, layer_x + layer_width/2],
                        [y, y - 0.3], 'k-', linewidth=1.5, alpha=0.5)
    
    ax_left.text(layer_x + layer_width/2, layer_y_start + 0.8, 
                'Deep Network\n(Many Layers)', ha='center', fontsize=11, fontweight='bold')
    
    # Arrow to MCMC
    arrow1 = FancyArrowPatch((layer_x + layer_width, layer_y_start - 2),
                            (5, layer_y_start - 2),
                            arrowstyle='->', mutation_scale=20, linewidth=2.5,
                            color='#e74c3c')
    ax_left.add_patch(arrow1)
    
    # MCMC chain icon
    mcmc_x = 5.5
    mcmc_y = 7
    n_samples = 6
    for i in range(n_samples):
        x = mcmc_x + (i % 3) * 0.8
        y = mcmc_y - (i // 3) * 0.8
        circle = Circle((x, y), 0.25, facecolor='#e67e22', edgecolor='#d35400', linewidth=1.5)
        ax_left.add_patch(circle)
        if i < n_samples - 1:
            next_x = mcmc_x + ((i+1) % 3) * 0.8
            next_y = mcmc_y - ((i+1) // 3) * 0.8
            ax_left.plot([x, next_x], [y, next_y], 'k--', linewidth=1, alpha=0.6)
    
    ax_left.text(mcmc_x + 1, mcmc_y + 0.8, 'MCMC Chain\n(Sampling)', 
                ha='center', fontsize=11, fontweight='bold')
    
    # Arrow to approximate posterior
    arrow2 = FancyArrowPatch((mcmc_x + 1.5, mcmc_y - 0.5),
                            (mcmc_x + 1.5, 3.5),
                            arrowstyle='->', mutation_scale=20, linewidth=2.5,
                            color='#e74c3c')
    ax_left.add_patch(arrow2)
    
    # Approximate posterior sample box
    post_box = FancyBboxPatch((5, 2), 3, 1,
                             boxstyle="round,pad=0.1",
                             facecolor='#95a5a6', edgecolor='#7f8c8d', linewidth=2)
    ax_left.add_patch(post_box)
    ax_left.text(6.5, 2.5, 'Approx. Posterior\nSample', ha='center', va='center',
                fontsize=11, fontweight='bold')
    
    # Intractable annotation
    intract_box = FancyBboxPatch((1, 0.5), 7, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor='#ffcccc', edgecolor='#c0392b', linewidth=2)
    ax_left.add_patch(intract_box)
    ax_left.text(4.5, 0.9, '⚠ INTRACTABLE (requires MCMC)', ha='center', va='center',
                fontsize=12, fontweight='bold', color='#c0392b')
    
    # ============ RIGHT: Convex-PSRL ============
    ax_right = axes[1]
    ax_right.set_xlim(0, 10)
    ax_right.set_ylim(0, 10)
    ax_right.axis('off')
    ax_right.set_title('Convex-PSRL (2-Layer ReLU)', fontsize=16, fontweight='bold', pad=20)
    
    # 2-layer ReLU network icon (simplified)
    layer2_x = 2
    layer2_y_start = 7
    
    # Input layer
    rect1 = FancyBboxPatch((layer2_x, layer2_y_start), 1.5, 0.5,
                          boxstyle="round,pad=0.05",
                          facecolor='#3498db', edgecolor='#2874a6', linewidth=2)
    ax_right.add_patch(rect1)
    ax_right.text(layer2_x + 0.75, layer2_y_start + 0.25, 'Input', ha='center', va='center',
                 fontsize=9, fontweight='bold', color='white')
    
    # Hidden layer (ReLU)
    rect2 = FancyBboxPatch((layer2_x, layer2_y_start - 1.2), 1.5, 0.5,
                          boxstyle="round,pad=0.05",
                          facecolor='#27ae60', edgecolor='#1e8449', linewidth=2)
    ax_right.add_patch(rect2)
    ax_right.text(layer2_x + 0.75, layer2_y_start - 0.95, 'ReLU', ha='center', va='center',
                 fontsize=9, fontweight='bold', color='white')
    
    # Output layer
    rect3 = FancyBboxPatch((layer2_x, layer2_y_start - 2.4), 1.5, 0.5,
                          boxstyle="round,pad=0.05",
                          facecolor='#3498db', edgecolor='#2874a6', linewidth=2)
    ax_right.add_patch(rect3)
    ax_right.text(layer2_x + 0.75, layer2_y_start - 2.15, 'Output', ha='center', va='center',
                 fontsize=9, fontweight='bold', color='white')
    
    # Connections
    ax_right.plot([layer2_x + 0.75, layer2_x + 0.75],
                 [layer2_y_start, layer2_y_start - 0.7], 'k-', linewidth=1.5)
    ax_right.plot([layer2_x + 0.75, layer2_x + 0.75],
                 [layer2_y_start - 1.2, layer2_y_start - 1.9], 'k-', linewidth=1.5)
    
    ax_right.text(layer2_x + 0.75, layer2_y_start + 0.8,
                 '2-Layer Network', ha='center', fontsize=11, fontweight='bold')
    
    # Morphing arrow with "convex reformulation"
    arrow3 = FancyArrowPatch((layer2_x + 1.5, layer2_y_start - 1),
                            (5, layer2_y_start - 1),
                            arrowstyle='->', mutation_scale=20, linewidth=2.5,
                            color='#27ae60')
    ax_right.add_patch(arrow3)
    ax_right.text(4, layer2_y_start - 0.5, 'convex\nreformulation',
                 ha='center', fontsize=9, style='italic', color='#27ae60')
    
    # Convex program box
    cvx_box = FancyBboxPatch((5, 5.5), 2.5, 1.5,
                            boxstyle="round,pad=0.1",
                            facecolor='#f39c12', edgecolor='#d68910', linewidth=2)
    ax_right.add_patch(cvx_box)
    ax_right.text(6.25, 6.5, 'Convex QP', ha='center', va='top',
                 fontsize=11, fontweight='bold')
    ax_right.text(6.25, 6.2, 'min ||y - f(x)||²', ha='center', va='center',
                 fontsize=9, style='italic')
    ax_right.text(6.25, 5.9, '+ λ||w||²', ha='center', va='center',
                 fontsize=9, style='italic')
    
    # Arrow to MAP sample
    arrow4 = FancyArrowPatch((6.25, 5.5),
                            (6.25, 3.5),
                            arrowstyle='->', mutation_scale=20, linewidth=2.5,
                            color='#27ae60')
    ax_right.add_patch(arrow4)
    
    # Exact MAP sample box
    map_box = FancyBboxPatch((5, 2), 2.5, 1,
                            boxstyle="round,pad=0.1",
                            facecolor='#2ecc71', edgecolor='#27ae60', linewidth=2)
    ax_right.add_patch(map_box)
    ax_right.text(6.25, 2.5, 'Exact MAP\nSample', ha='center', va='center',
                 fontsize=11, fontweight='bold')
    
    # Tractable annotation
    tract_box = FancyBboxPatch((1, 0.5), 7, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor='#ccffcc', edgecolor='#27ae60', linewidth=2)
    ax_right.add_patch(tract_box)
    ax_right.text(4.5, 0.9, '✓ TRACTABLE (convex program)', ha='center', va='center',
                 fontsize=12, fontweight='bold', color='#27ae60')
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Figure 1 saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    # Ensure figures directory exists
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    save_path = figures_dir / 'figure1.pdf'
    create_figure1(str(save_path))
