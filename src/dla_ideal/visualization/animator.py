"""Enhanced DLA visualization with adaptive detail capture and elegant rendering."""

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


def create_elegant_colormap(name='hot'):
    """Create smooth, elegant colormaps."""
    if name == 'hot':
        colors = ['#0a0a0a', '#1a0f00', '#4a1500', '#ff4500', '#ff8c00', '#ffd700', '#ffffe0']
    elif name == 'cool':
        colors = ['#0a0a0a', '#001020', '#002040', '#0066cc', '#00ccff', '#66ffff', '#ccffff']
    elif name == 'viridis':
        colors = ['#0a0a0a', '#440154', '#31688e', '#35b779', '#fde724']
    elif name == 'plasma':
        colors = ['#0a0a0a', '#0d0887', '#7e03a8', '#cc4778', '#f89540', '#f0f921']
    elif name == 'aurora':
        colors = ['#0a0a0a', '#001a33', '#003366', '#00ff88', '#66ffcc', '#ccffee']
    elif name == 'fire':
        colors = ['#000000', '#330000', '#990000', '#ff3300', '#ff9900', '#ffcc00', '#ffffff']
    else:
        colors = ['#0a0a0a', '#ff4500', '#ffd700']
    
    return LinearSegmentedColormap.from_list('elegant', colors, N=256)


def _render_elegant_frame(frame_data):
    """Render single elegant DLA frame with enhanced visuals."""
    (frame_idx, grid, count, total_walkers, N, center, D, n_agg, 
     title, cmap_name, show_grid, glow_effect) = frame_data
    
    # Figure setup
    fig = plt.figure(figsize=(12, 11), facecolor='#000000')
    ax = plt.axes([0.05, 0.08, 0.9, 0.85])
    ax.set_facecolor('#0a0a0a')
    
    # Create elegant visualization
    display = np.zeros((N, N, 3))
    
    # Get colormap
    cmap = create_elegant_colormap(cmap_name)
    
    cx, cy = center
    max_dist = 0
    
    # First pass: find max distance for normalization
    for i in range(N):
        for j in range(N):
            if grid[i, j] == 2:
                dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                max_dist = max(max_dist, dist)
    
    if max_dist == 0:
        max_dist = 1
    
    # Second pass: color particles with smooth gradients
    intensity_map = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if grid[i, j] == 2:
                dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                # Smooth gradient from center
                color_val = dist / max_dist
                
                # Add slight randomness for texture
                color_val += np.random.uniform(-0.05, 0.05)
                color_val = np.clip(color_val, 0, 1)
                
                intensity_map[i, j] = color_val
                
                # Apply colormap
                rgba = cmap(color_val)
                display[i, j] = rgba[:3]
    
    # Optional glow effect
    if glow_effect:
        from scipy.ndimage import gaussian_filter
        glow = gaussian_filter(intensity_map, sigma=2.0)
        glow = glow / glow.max() if glow.max() > 0 else glow
        
        for i in range(N):
            for j in range(N):
                if grid[i, j] == 2:
                    display[i, j] = np.clip(display[i, j] * (1 + glow[i, j] * 0.3), 0, 1)
    
    ax.imshow(display, origin='lower', interpolation='bicubic' if N < 256 else 'bilinear')
    
    # Progress indicator
    progress = count / total_walkers
    
    # Title with progress
    title_text = f'{title}\nParticles: {count:,} / {total_walkers:,} ({progress*100:.1f}%)'
    ax.set_title(title_text, 
                color='white', fontsize=14, fontweight='bold', 
                pad=20, family='monospace')
    
    # Elegant stats box with gradient background
    stats_text = (f'Aggregates: {n_agg}  |  Fractal Dim: {D:.3f}  |  '
                 f'Max Radius: {int(max_dist)}')
    
    # Stats box
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
    
    # Progress bar
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
    
    # Progress bar
    ax.add_patch(plt.Rectangle((bar_x, bar_y), bar_width * progress, bar_height,
                               transform=ax.transAxes,
                               facecolor='#00ff88',
                               edgecolor='#00ff88',
                               linewidth=0,
                               alpha=0.8,
                               zorder=101))
    
    # Optional grid
    if show_grid and N <= 256:
        ax.grid(True, alpha=0.1, color='#444444', linewidth=0.5)
    
    ax.axis('off')
    
    # Save to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, facecolor='#000000',
               bbox_inches='tight', pad_inches=0.15)
    buf.seek(0)
    img = Image.open(buf)
    
    # Enhance contrast slightly for elegance
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.1)
    
    img_copy = img.copy()
    buf.close()
    plt.close(fig)
    
    return img_copy


class ElegantAnimator:
    """Enhanced DLA animations with adaptive detail and elegant rendering."""
    
    @staticmethod
    def create_gif(result, filename, output_dir="outputs",
                  title="DLA Growth", fps=10, colormap='hot',
                  show_grid=False, glow_effect=False,
                  quality='high', pause_at_end=2.0):
        """
        Create elegant animated GIF of DLA growth.
        
        Args:
            result: DLA simulation results
            filename: Output filename
            output_dir: Output directory
            title: Animation title
            fps: Frames per second (lower = slower, more detail)
                 Recommended: 5-10 for detailed viewing
            colormap: Color scheme ('hot', 'cool', 'viridis', 'plasma', 'aurora', 'fire')
            show_grid: Show subtle grid lines (for smaller lattices)
            glow_effect: Add glow around particles (requires scipy)
            quality: 'high' or 'medium' (affects interpolation)
            pause_at_end: Seconds to pause on final frame
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
        print(f"    Rendering {n_frames} elegant frames...")
        print(f"    FPS: {fps} | Duration: {n_frames/fps:.1f}s")
        print(f"    Colormap: {colormap} | Effects: {'glow ' if glow_effect else ''}{'grid' if show_grid else ''}")
        
        # Prepare frame data
        frame_data_list = []
        for i in range(n_frames):
            frame_data = (
                i, snapshots[i], counts[i], total_walkers, N, center, 
                D, n_agg, title, colormap, show_grid, glow_effect
            )
            frame_data_list.append(frame_data)
        
        # Parallel rendering
        n_processes = max(1, cpu_count() - 1)
        print(f"    Using {n_processes} CPU cores...")
        
        with Pool(processes=n_processes) as pool:
            frames = pool.map(_render_elegant_frame, frame_data_list)
        
        # Add pause frames at end
        if pause_at_end > 0:
            pause_frames = int(fps * pause_at_end)
            frames.extend([frames[-1]] * pause_frames)
            print(f"    Added {pause_frames} pause frames at end")
        
        print(f"    Saving GIF ({len(frames)} frames @ {fps} fps)...")
        
        # Calculate duration per frame in milliseconds
        duration_ms = int(1000 / fps)
        
        frames[0].save(
            filepath,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False
        )
        
        print(f"    ✓ Elegant animation complete!")
        print(f"    Size: {filepath.stat().st_size / 1024 / 1024:.1f} MB")


# Convenience function for different speed presets
def create_detailed_animation(result, filename, output_dir="outputs",
                             title="DLA Growth", speed='slow', **kwargs):
    """
    Create animation with preset speed configurations.
    
    Args:
        speed: 'very_slow' (5 fps), 'slow' (10 fps), 'normal' (20 fps), 'fast' (30 fps)
    """
    fps_presets = {
        'very_slow': 5,
        'slow': 10,
        'normal': 20,
        'fast': 30
    }
    
    fps = fps_presets.get(speed, 10)
    
    ElegantAnimator.create_gif(
        result, filename, output_dir, title, fps=fps, **kwargs
    )
