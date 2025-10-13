"""Professional DLA visualization with parallel rendering - Enhanced for Detail Capture."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from multiprocessing import Pool, cpu_count
import io
from PIL import Image, ImageEnhance

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
mpl.rcParams['font.size'] = 10


def _create_elegant_colormap(name='hot'):
    """Create smooth, elegant colormaps."""
    colormaps = {
        'hot': ['#0a0a0a', '#1a0f00', '#4a1500', '#ff4500', '#ff8c00', '#ffd700', '#ffffe0'],
        'cool': ['#0a0a0a', '#001020', '#002040', '#0066cc', '#00ccff', '#66ffff', '#ccffff'],
        'viridis': ['#0a0a0a', '#440154', '#31688e', '#35b779', '#fde724'],
        'plasma': ['#0a0a0a', '#0d0887', '#7e03a8', '#cc4778', '#f89540', '#f0f921'],
        'aurora': ['#0a0a0a', '#001a33', '#003366', '#00ff88', '#66ffcc', '#ccffee'],
        'fire': ['#000000', '#330000', '#990000', '#ff3300', '#ff9900', '#ffcc00', '#ffffff'],
        'ocean': ['#000000', '#001133', '#003366', '#0066cc', '#00ccff', '#66ffff'],
        'sunset': ['#0a0a0a', '#1a0033', '#660066', '#cc0066', '#ff6600', '#ffcc00', '#ffff99']
    }
    
    colors = colormaps.get(name, colormaps['hot'])
    return LinearSegmentedColormap.from_list('elegant', colors, N=256)


def _render_single_frame(frame_data):
    """Render single DLA frame with enhanced elegance."""
    try:
        (frame_idx, grid, count, total_walkers, N, center, D, n_agg, title, cmap_name) = frame_data
        
        # Figure setup with elegant styling
        fig = plt.figure(figsize=(12, 11), facecolor='#000000')
        ax = plt.axes([0.05, 0.08, 0.9, 0.85])
        ax.set_facecolor('#0a0a0a')
        
        # Create visualization
        display = np.zeros((N, N, 3))
        
        # Get elegant colormap
        cmap = _create_elegant_colormap(cmap_name)
        
        cx, cy = center
        max_dist = 0
        
        # Find max distance for normalization
        for i in range(N):
            for j in range(N):
                if grid[i, j] == 2:
                    dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                    max_dist = max(max_dist, dist)
        
        if max_dist == 0:
            max_dist = 1
        
        # Color particles with smooth gradients
        for i in range(N):
            for j in range(N):
                if grid[i, j] == 2:  # Stuck particle
                    dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                    # Smooth gradient from center
                    color_val = dist / max_dist
                    
                    # Add subtle texture variation
                    color_val += np.random.uniform(-0.03, 0.03)
                    color_val = np.clip(color_val, 0, 1)
                    
                    # Apply colormap
                    rgba = cmap(color_val)
                    display[i, j] = rgba[:3]
        
        # Use bicubic interpolation for smoothness
        interpolation = 'bicubic' if N <= 512 else 'bilinear'
        ax.imshow(display, origin='lower', interpolation=interpolation)
        
        # Calculate progress
        progress = count / total_walkers if total_walkers > 0 else 0
        
        # Title with progress
        title_text = f'{title}\nParticles: {count:,} / {total_walkers:,} ({progress*100:.1f}%)'
        ax.set_title(title_text, 
                    color='white', fontsize=14, fontweight='bold', 
                    pad=20, family='monospace')
        
        # Elegant stats box
        stats_text = (f'Aggregates: {n_agg}  |  Fractal Dim: {D:.3f}  |  '
                     f'Max Radius: {int(max_dist)}')
        
        ax.text(0.5, 0.015, stats_text,
               transform=ax.transAxes,
               ha='center', va='bottom',
               color='#ffffff',
               fontsize=10,
               fontweight='bold',
               family='monospace',
               bbox=dict(boxstyle='round,pad=0.8',
                       facecolor='#1a1a1a',
                       edgecolor='#ffd700',
                       linewidth=2.5,
                       alpha=0.95))
        
        # Progress bar at top
        bar_width = 0.8
        bar_height = 0.02
        bar_x = 0.1
        bar_y = 0.96
        
        # Background bar
        ax.add_patch(plt.Rectangle((bar_x, bar_y), bar_width, bar_height,
                                   transform=ax.transAxes,
                                   facecolor='#2a2a2a',
                                   edgecolor='#555555',
                                   linewidth=1.5,
                                   zorder=100))
        
        # Progress fill
        ax.add_patch(plt.Rectangle((bar_x, bar_y), bar_width * progress, bar_height,
                                   transform=ax.transAxes,
                                   facecolor='#00ff88',
                                   edgecolor='#00ff88',
                                   linewidth=0,
                                   alpha=0.8,
                                   zorder=101))
        
        ax.axis('off')
        
        # Save to buffer with high quality
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, facecolor='#000000',
                   bbox_inches='tight', pad_inches=0.15)
        buf.seek(0)
        img = Image.open(buf)
        
        # Enhance contrast for elegance
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)
        
        img_copy = img.copy()
        buf.close()
        plt.close(fig)
        
        return img_copy
    
    except Exception as e:
        print(f"Error rendering frame {frame_idx}: {e}")
        # Return a blank frame on error
        return Image.new('RGB', (1200, 1100), color='black')


class Animator:
    """Professional DLA animations with enhanced detail capture."""
    
    @staticmethod
    def create_gif(result, filename, output_dir="outputs",
                  title="DLA Growth", fps=20, colormap='hot'):
        """
        Create animated GIF of DLA growth with enhanced quality.
        
        Args:
            result: DLA simulation results dictionary
            filename: Output filename (e.g., 'simulation.gif')
            output_dir: Output directory path
            title: Animation title text
            fps: Frames per second (lower = slower, more detail)
            colormap: Color scheme ('hot', 'cool', 'viridis', 'plasma', 
                                   'aurora', 'fire', 'ocean', 'sunset')
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        snapshots = result['snapshots']
        counts = result['glued_counts']
        N = result['params']['N']
        center = result['center']
        D = result['fractal_dimension']
        n_agg = result['n_aggregates']
        total_walkers = result['params']['n_walkers']
        
        n_frames = len(snapshots)
        duration_sec = n_frames / fps
        
        print(f"    Rendering {n_frames} frames with enhanced quality...")
        print(f"    Playback: {fps} fps ({duration_sec:.1f}s duration)")
        print(f"    Colormap: {colormap}")
        
        # Prepare frame data
        frame_data_list = []
        for i in range(n_frames):
            frame_data = (
                i, snapshots[i], counts[i], total_walkers, N, center, 
                D, n_agg, title, colormap
            )
            frame_data_list.append(frame_data)
        
        # Parallel rendering
        n_processes = max(1, cpu_count() - 1)
        print(f"    Using {n_processes} CPU cores...")
        
        with Pool(processes=n_processes) as pool:
            frames = pool.map(_render_single_frame, frame_data_list)
        
        # Add pause frames at end (2 seconds)
        pause_frames = int(fps * 2.0)
        frames.extend([frames[-1]] * pause_frames)
        print(f"    Added {pause_frames} pause frames at end")
        
        print(f"    Saving GIF ({len(frames)} total frames @ {fps} fps)...")
        
        # Calculate duration per frame in milliseconds
        duration_ms = int(1000 / fps)
        
        # Save with optimization
        frames[0].save(
            filepath,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False
        )
        
        file_size_mb = filepath.stat().st_size / 1024 / 1024
        print(f"    ✓ Animation complete!")
        print(f"    File: {filepath} ({file_size_mb:.1f} MB)")
