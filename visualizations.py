"""
Visualizations for Ring Convolution Networks
Generate figures for the paper
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches


def visualize_ring_structure(save_path='ring_structure.png'):
    """Visualize the ring weight structure."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Center weight
    center = Circle((0, 0), 0.3, color='#3498db', alpha=0.8)
    ax.add_patch(center)
    ax.text(0, 0, 'w_center', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='white')
    
    # First ring
    angles = np.linspace(0, 2*np.pi, 9)[:-1]
    r1 = 1.2
    for i, angle in enumerate(angles):
        x, y = r1 * np.cos(angle), r1 * np.sin(angle)
        color = '#e74c3c' if i % 2 == 0 else '#2ecc71'
        circle = Circle((x, y), 0.25, color=color, alpha=0.7)
        ax.add_patch(circle)
        
        text = 'w_left' if i % 2 == 0 else 'w_right'
        ax.text(x, y, text, ha='center', va='center', fontsize=9, color='white')
    
    # Connections
    for angle in angles[::2]:
        x1, y1 = 0.3 * np.cos(angle), 0.3 * np.sin(angle)
        x2, y2 = 0.95 * np.cos(angle), 0.95 * np.sin(angle)
        ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3)
    
    # Superposition formula
    ax.text(0, -2.5, 'Superposition = 0.5×w_center + 0.25×w_left + 0.25×w_right', 
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8))
    
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-3, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Ring Weight Structure', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ring structure visualization to {save_path}")


def compare_architectures_visual(save_path='architecture_comparison.png'):
    """Compare traditional NN and RCN architectures."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Traditional NN
    ax1.set_title("Traditional Neural Network", fontsize=14, fontweight='bold')
    ax1.set_xlim(-1, 3)
    ax1.set_ylim(-1, 3)
    
    # Input neurons
    for i in range(3):
        circle = Circle((0, i), 0.2, color='lightblue', ec='black')
        ax1.add_patch(circle)
        ax1.text(-0.5, i, f'x_{i+1}', fontsize=10, va='center')
    
    # Output neurons
    for i in range(2):
        circle = Circle((2, i+0.5), 0.2, color='lightgreen', ec='black')
        ax1.add_patch(circle)
        ax1.text(2.5, i+0.5, f'y_{i+1}', fontsize=10, va='center')
    
    # Weights
    for i in range(3):
        for j in range(2):
            ax1.plot([0.2, 1.8], [i, j+0.5], 'k-', alpha=0.3)
            ax1.text(1, i*0.5 + j*0.5 + 0.25, 'w', fontsize=8, ha='center')
    
    ax1.axis('off')
    
    # RCN
    ax2.set_title("Ring Convolution Network", fontsize=14, fontweight='bold')
    ax2.set_xlim(-1, 3)
    ax2.set_ylim(-1, 3)
    
    # Input neurons
    for i in range(3):
        circle = Circle((0, i), 0.2, color='lightblue', ec='black')
        ax2.add_patch(circle)
        ax2.text(-0.5, i, f'x_{i+1}', fontsize=10, va='center')
    
    # Output neurons
    for i in range(2):
        circle = Circle((2, i+0.5), 0.2, color='lightgreen', ec='black')
        ax2.add_patch(circle)
        ax2.text(2.5, i+0.5, f'y_{i+1}', fontsize=10, va='center')
    
    # Ring weights
    for i in range(3):
        for j in range(2):
            # Center
            ax2.plot([0.2, 1.8], [i, j+0.5], 'k-', alpha=0.6, linewidth=2)
            # Left mirror
            ax2.plot([0.2, 1.8], [i-0.1, j+0.5-0.05], 'b--', alpha=0.4)
            # Right mirror
            ax2.plot([0.2, 1.8], [i+0.1, j+0.5+0.05], 'r--', alpha=0.4)
            
            # Ring symbol
            ring = Circle((1, i*0.5 + j*0.5 + 0.25), 0.15, 
                         fill=False, ec='purple', linewidth=2)
            ax2.add_patch(ring)
    
    # Legend
    center_line = mpatches.Patch(color='black', label='Center weight')
    left_line = mpatches.Patch(color='blue', label='Left mirror')
    right_line = mpatches.Patch(color='red', label='Right mirror')
    ax2.legend(handles=[center_line, left_line, right_line], 
              loc='upper right', bbox_to_anchor=(1.2, 1))
    
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved architecture comparison to {save_path}")


def plot_results(save_path='results.png'):
    """Plot experimental results."""
    # Data from experiments
    datasets = ['Random\nData', 'Smooth\nSignals', 'MNIST\nImages']
    standard = [10.2, 12.2, 8.8]
    rcn = [9.6, 9.1, 10.6]
    improvements = [-5.6, -25.7, 19.8]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy comparison
    bars1 = ax1.bar(x - width/2, standard, width, label='Standard NN', 
                     color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, rcn, width, label='RCN', 
                     color='#2ecc71', alpha=0.8)
    
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_xlabel('Dataset Type', fontsize=12)
    ax1.set_title('Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add values
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Improvement plot
    colors = ['#e74c3c' if x < 0 else '#27ae60' for x in improvements]
    bars3 = ax2.bar(x, improvements, color=colors, alpha=0.8)
    
    ax2.set_ylabel('Improvement (%)', fontsize=12)
    ax2.set_xlabel('Dataset Type', fontsize=12)
    ax2.set_title('RCN Improvement over Standard NN', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add values
    for bar, imp in zip(bars3, improvements):
        height = bar.get_height()
        y_pos = height + 1 if height > 0 else height - 1
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{imp:+.1f}%', ha='center', 
                va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved results plot to {save_path}")


def generate_all_figures():
    """Generate all figures for the paper."""
    print("Generating figures for Ring Convolution Networks paper...")
    print("-" * 50)
    
    visualize_ring_structure()
    compare_architectures_visual()
    plot_results()
    
    print("-" * 50)
    print("All figures generated successfully!")


if __name__ == "__main__":
    generate_all_figures()
