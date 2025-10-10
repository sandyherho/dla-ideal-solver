"""Professional DLA visualization with parallel rendering."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
from multiprocessing import Pool, cpu_count
import io
from PIL import Image

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
mpl.rcParams['font.size'] = 10


def _render_single_frame(frame_data):
    """Render single DLA frame."""
    (frame_idx, grid, count, N, center, D, n_agg, title, cmap) = frame_data
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='black')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.08)
    
    ax.set_facecolor('#0a0a0a')
    
    # Create visualization
    display = np.zeros((N, N, 3))
    
    # Color particles by age (approximate from center)
    cx, cy = center
    for i in range(N):
        for j in range(N):
            if grid[i, j] == 2:  # Stuck particle
                dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                color_val = min(1.0, dist / (N * 0.4))
                
                if cmap == 'hot':
                    display[i, j] = [color_val, color_val * 0.5, 0.0]
                elif cmap == 'cool':
                    display[i, j] = [0.0, color_val * 0.7, color_val]
                elif cmap == 'viridis':
                    r = 0.3 + 0.7 * color_val
                    g = 0.2 + 0.6 * (1 - color_val)
                    b = 0.5 + 0.5 * (1 - abs(color_val - 0.5))
                    display[i, j] = [r, g, b]
    
    ax.imshow(display, origin='lower', interpolation='nearest')
    
    # Title and info
    ax.set_title(f'{title}\nParticles: {count}', 
                color='white', fontsize=14, fontweight='bold', pad=15)
    
    # Stats box
    stats_text = f'Aggregates: {n_agg}   |   D = {D:.3f}'
    
    ax.text(0.5, 0.02, stats_text,
           transform=ax.transAxes,
           ha='center', va='bottom',
           color='white',
           fontsize=11,
           fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.6',
                   facecolor='#1a1a1a',
                   edgecolor='#FFD700',
                   linewidth=2.0,
                   alpha=0.9))
    
    ax.axis('off')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, facecolor='black',
               bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    img = Image.open(buf)
    img_copy = img.copy()
    buf.close()
    plt.close(fig)
    
    return img_copy


class Animator:
    """Professional DLA animations with parallel rendering."""
    
    @staticmethod
    def create_gif(result, filename, output_dir="outputs",
                  title="DLA Growth", fps=30, colormap='hot'):
        """Create animated GIF of DLA growth."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        snapshots = result['snapshots']
        counts = result['glued_counts']
        N = result['params']['N']
        center = result['center']
        D = result['fractal_dimension']
        n_agg = result['n_aggregates']
        
        n_frames = len(snapshots)
        print(f"    Rendering {n_frames} frames in parallel...")
        
        frame_data_list = []
        for i in range(n_frames):
            frame_data = (
                i, snapshots[i], counts[i], N, center, 
                D, n_agg, title, colormap
            )
            frame_data_list.append(frame_data)
        
        n_processes = max(1, cpu_count() - 1)
        print(f"    Using {n_processes} CPU cores...")
        
        with Pool(processes=n_processes) as pool:
            frames = pool.map(_render_single_frame, frame_data_list)
        
        print(f"    Saving GIF ({n_frames} frames @ {fps} fps)...")
        
        frames[0].save(
            filepath,
            save_all=True,
            append_images=frames[1:],
            duration=1000/fps,
            loop=0,
            optimize=False
        )
        
        print(f"    ✓ Animation complete!")
