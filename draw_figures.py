"""
NMI Paper Figure Generation Script
Generates Figure 2, Figure 3, Figure 4 following Nature Machine Intelligence style.

Output:
  - figures/figure2_attack_params.pdf/.png
  - figures/figure3_alignment_bypass.pdf/.png
  - figures/figure4_defense_performance.pdf/.png
"""

import os
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# =============================================================================
# Nature MI Style Configuration
# =============================================================================
RESULTS_DIR = r"C:\Users\Administrator\Desktop\NMI\results"
FIGURES_DIR = r"C:\Users\Administrator\Desktop\NMI\figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Okabe-Ito color palette
COLOR_ATTACK = '#CC0000'   # Red - Attack/PRNG
COLOR_DEFENSE = '#029E73'  # Green - Defense/QRNG
COLOR_NEUTRAL = '#0173B2'  # Blue - Neutral/Model
COLOR_ACCENT = '#DE8F05'   # Orange - Emphasis
COLOR_GRAY = '#999999'     # Gray - Control

# Nature MI dimensions (mm -> inches, 1 inch = 25.4 mm)
SINGLE_COL_WIDTH = 89 / 25.4    # ~3.504 inches
DOUBLE_COL_WIDTH = 183 / 25.4   # ~7.205 inches

# Font sizes
LABEL_SIZE = 7   # pt
TICK_SIZE = 6    # pt
TITLE_SIZE = 8   # pt

def setup_nature_style():
    """Configure matplotlib for Nature MI style."""
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': LABEL_SIZE,
        'axes.labelsize': LABEL_SIZE,
        'axes.titlesize': TITLE_SIZE,
        'xtick.labelsize': TICK_SIZE,
        'ytick.labelsize': TICK_SIZE,
        'legend.fontsize': TICK_SIZE,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })

setup_nature_style()


# =============================================================================
# Utility functions
# =============================================================================
def load_json(filename):
    """Load a JSON file from results directory. Returns None if not found."""
    filepath = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def save_figure(fig, basename):
    """Save figure as both PDF and PNG."""
    pdf_path = os.path.join(FIGURES_DIR, f"{basename}.pdf")
    png_path = os.path.join(FIGURES_DIR, f"{basename}.png")
    fig.savefig(pdf_path, format='pdf')
    fig.savefig(png_path, format='png', dpi=300)
    plt.close(fig)
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")


# =============================================================================
# Figure 2: Attack success rate heatmap (3x3: temp x top_p)
# =============================================================================
def draw_figure2():
    """Heatmap of injection rate across 9 sampling configurations."""
    print("\n[Figure 2] Attack success rate across sampling configurations")

    # Local font sizes (+1pt for this figure)
    F2_LABEL = LABEL_SIZE + 1
    F2_TICK = TICK_SIZE + 1
    F2_TITLE = TITLE_SIZE + 1

    # Try loading real data
    data = load_json("exp1_attack_b.json")

    temperatures = [0.7, 1.0, 1.5]
    top_ps = [0.9, 0.95, 1.0]

    if data and "conditions" in data:
        print("  Using real data from exp1_attack_b.json")
        # Build matrix from JSON
        matrix = np.zeros((3, 3))
        matches_matrix = np.zeros((3, 3), dtype=int)
        runs_matrix = np.zeros((3, 3), dtype=int)

        for cond in data["conditions"]:
            t = cond["temperature"]
            p = cond["top_p"]
            ti = temperatures.index(t)
            pi = top_ps.index(p)
            matrix[ti, pi] = cond["exact_match_rate"] * 100
            matches_matrix[ti, pi] = cond["exact_matches"]
            runs_matrix[ti, pi] = cond["n_runs"]

        overall_matches = data["overall_summary"]["exact_matches"]
        overall_total = data["overall_summary"]["total_runs"]
        overall_rate = data["overall_summary"]["exact_match_rate"] * 100
    else:
        print("  WARNING: exp1_attack_b.json not found or missing fields. Using fallback data.")
        matrix = np.array([
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 98.3],
            [98.3, 100.0, 100.0],
        ])
        matches_matrix = np.array([
            [60, 60, 60],
            [60, 60, 59],
            [59, 60, 60],
        ])
        runs_matrix = np.full((3, 3), 60)
        overall_matches = 538
        overall_total = 540
        overall_rate = 99.6

    # Create figure
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, SINGLE_COL_WIDTH * 0.85))

    # Custom colormap: light blue to deep blue (Nature-style academic palette)
    cmap = LinearSegmentedColormap.from_list(
        'attack_heat',
        ['#F7FBFF', '#6BAED6', '#2171B5', '#08306B'],
        N=256
    )

    # Plot heatmap
    im = ax.imshow(matrix, cmap=cmap, vmin=97.5, vmax=100.5, aspect='auto')

    # Add text annotations
    for i in range(3):
        for j in range(3):
            val = matrix[i, j]
            m = int(matches_matrix[i, j])
            n = int(runs_matrix[i, j])
            # White text on dark background
            text_color = 'white' if val > 99.5 else 'black'
            ax.text(j, i, f"{val:.1f}%\n({m}/{n})",
                    ha='center', va='center', fontsize=F2_TICK,
                    color=text_color, fontweight='bold')

    # Labels
    ax.set_xticks(range(3))
    ax.set_xticklabels([str(p) for p in top_ps])
    ax.set_yticks(range(3))
    ax.set_yticklabels([str(t) for t in temperatures])
    ax.set_xlabel('Top-p', fontsize=F2_LABEL)
    ax.set_ylabel(r'Temperature ($\tau$)', fontsize=F2_LABEL)

    # Heatmap: keep all 4 spines visible (no arrows)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Injection Rate (%)', fontsize=F2_TICK)
    cbar.ax.tick_params(labelsize=F2_TICK)

    # Flip Y-axis so Temperature goes from low (bottom) to high (top)
    ax.set_ylim(-0.5, 2.5)

    plt.tight_layout(pad=0.4)
    save_figure(fig, "figure2_attack_params")


# =============================================================================
# Figure 3: Attack success across models with different alignment
# =============================================================================
def draw_figure3():
    """Horizontal bar chart comparing attack across differently-aligned models."""
    print("\n[Figure 3] Attack success across models with different alignment")

    # Try loading exp4 data
    data = load_json("exp4_attack_c.json")

    # Fixed order: GPT-2 → Qwen2 → DeepSeek
    MODEL_ORDER = [
        ("gpt2-baseline", "GPT-2", "No alignment"),
        ("qwen2-1.5b-instruct", "Qwen2-1.5B-Instruct", "RLHF"),
        ("deepseek-r1-distill-1.5b", "DeepSeek-R1-Distill-1.5B", "R1 Distillation"),
    ]

    if data and "models" in data and isinstance(data["models"], dict):
        print("  Using real data from exp4_attack_c.json")
        models = []
        rates = []
        alignments = []
        for key, display_name, alignment in MODEL_ORDER:
            model_data = data["models"].get(key, {})
            attack_data = model_data.get("attack_b", {})
            rate = attack_data.get("overall_exact_match_rate", 0) * 100
            models.append(display_name)
            rates.append(rate)
            alignments.append(alignment)
            print(f"    {display_name} ({alignment}): {rate:.1f}%")
    else:
        print("  WARNING: exp4_attack_c.json not found or missing fields. Using fallback/placeholder data.")
        models = [
            'GPT-2',
            'Qwen2-1.5B-Instruct',
            'DeepSeek-R1-Distill-1.5B'
        ]
        rates = [100.0, 100.0, 100.0]
        alignments = ['No alignment', 'RLHF', 'R1 Distillation']

    # Colors for each model
    colors = [COLOR_NEUTRAL, COLOR_ACCENT, COLOR_DEFENSE]

    # Create figure
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, SINGLE_COL_WIDTH * 0.55))

    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, rates, height=0.55, color=colors, edgecolor='none', alpha=0.9)

    # Y-axis labels: model name + alignment
    labels = [f"{m}\n({a})" for m, a in zip(models, alignments)]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=TICK_SIZE)

    # X-axis
    ax.set_xlim(0, 110)
    ax.set_xlabel('Injection Rate (%)', fontsize=LABEL_SIZE)
    ax.set_xticks([0, 25, 50, 75, 100])

    # Value labels at end of bars
    for i, (bar, rate) in enumerate(zip(bars, rates)):
        ax.text(rate + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1f}%", va='center', ha='left',
                fontsize=TICK_SIZE, fontweight='bold')

    # Red dashed line at 100%
    ax.axvline(x=100, color=COLOR_ATTACK, linestyle='--', linewidth=0.8, alpha=0.7)

    # Remove top/right spines (already done via rcParams, ensure left/bottom remain)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    ax.invert_yaxis()  # Top model first
    plt.tight_layout()
    save_figure(fig, "figure3_alignment_bypass")


# =============================================================================
# Figure 4: QRNG defense efficacy and performance overhead (three-panel)
# =============================================================================
def draw_figure4():
    """Three-panel figure: (a) defense efficacy, (b) latency comparison, (c) memory usage."""
    print("\n[Figure 4] QRNG defense efficacy, latency, and memory (3-panel)")

    # --- Load data ---
    data_defense = load_json("exp2_qrng_defense.json")
    data_perf = load_json("exp3_performance.json")

    # Panel (a) data
    if data_defense and "attack_b_defense" in data_defense:
        print("  Panel (a): Using real data from exp2_qrng_defense.json")
        prng_rate = data_defense["attack_b_defense"]["prng_exact_match_rate"] * 100
        qrng_rate = data_defense["attack_b_defense"]["qrng_exact_match_rate"] * 100
        prng_matches = data_defense["attack_b_defense"]["prng_matches"]
        qrng_matches = data_defense["attack_b_defense"]["qrng_matches"]
        total_runs = data_defense["attack_b_defense"]["total_runs"]
    else:
        print("  WARNING: exp2_qrng_defense.json not found. Using fallback.")
        prng_rate = 100.0
        qrng_rate = 0.0
        prng_matches = 100
        qrng_matches = 0
        total_runs = 100

    # Panel (b) & (c) data
    if data_perf and "prng" in data_perf and "qrng" in data_perf:
        print("  Panel (b)/(c): Using real data from exp3_performance.json")
        prng_median = data_perf["prng"]["per_token_latency_ms"]["p50"]
        qrng_median = data_perf["qrng"]["per_token_latency_ms"]["p50"]
        prng_p95 = data_perf["prng"]["per_token_latency_ms"]["p95"]
        qrng_p95 = data_perf["qrng"]["per_token_latency_ms"]["p95"]
        prng_mean = data_perf["prng"]["per_token_latency_ms"]["mean"]
        qrng_mean = data_perf["qrng"]["per_token_latency_ms"]["mean"]
        prng_mem = data_perf["prng"]["memory_mb"]
        qrng_mem = data_perf["qrng"]["memory_mb"]
    else:
        print("  WARNING: exp3_performance.json not found. Using fallback.")
        prng_median = 12.58
        qrng_median = 12.65
        prng_p95 = 13.34
        qrng_p95 = 21.30
        prng_mean = 12.81
        qrng_mean = 13.21
        prng_mem = 1306.1
        qrng_mem = 1313.7

    mem_diff = qrng_mem - prng_mem
    pct_median = (qrng_median - prng_median) / prng_median * 100
    pct_p95 = (qrng_p95 - prng_p95) / prng_p95 * 100
    pct_mean = (qrng_mean - prng_mean) / prng_mean * 100

    print(f"    Median: PRNG {prng_median:.2f} ms, QRNG {qrng_median:.2f} ms (+{pct_median:.1f}%)")
    print(f"    P95:    PRNG {prng_p95:.2f} ms, QRNG {qrng_p95:.2f} ms (+{pct_p95:.1f}%)")
    print(f"    Mean:   PRNG {prng_mean:.2f} ms, QRNG {qrng_mean:.2f} ms (+{pct_mean:.1f}%)")
    print(f"    Memory: PRNG {prng_mem:.1f} MB, QRNG {qrng_mem:.1f} MB (+{mem_diff:.1f} MB)")

    # Font size compensation for multi-panel shrinkage (+5pt vs base)
    F4_LABEL = LABEL_SIZE + 5
    F4_TICK = TICK_SIZE + 5
    F4_TITLE = TITLE_SIZE + 5

    # Nature Research color palette for this figure
    F4_COLOR_PRNG = '#E64B35'   # Nature red
    F4_COLOR_QRNG = '#4DBBD5'   # Nature cyan

    # Create two-row figure: Row 1 = (a)+(b) side by side, Row 2 = (c) full width
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.75), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.1], hspace=0.15, wspace=0.01)
    ax1 = fig.add_subplot(gs[0, 0])  # Row 1 left: (a) Defense Efficacy
    ax2 = fig.add_subplot(gs[0, 1])  # Row 1 right: (b) Memory Usage
    ax3 = fig.add_subplot(gs[1, :])  # Row 2 full width: (c) Latency Comparison

    # =========================================================================
    # Panel (a): Defense Efficacy
    # =========================================================================
    categories_a = ['PRNG\n(Vulnerable)', 'QRNG\n(Defended)']
    values_a = [prng_rate, qrng_rate]
    colors_a = [F4_COLOR_PRNG, F4_COLOR_QRNG]

    bars_a = ax1.bar(categories_a, values_a, width=0.5, color=colors_a, edgecolor='none', alpha=0.9)
    ax1.set_ylabel('Injection Rate (%)', fontsize=F4_LABEL)
    ax1.set_ylim(0, 120)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.tick_params(axis='both', labelsize=F4_TICK)

    # Value labels
    ax1.text(0, prng_rate + 3, f"{prng_rate:.0f}%\n({prng_matches}/{total_runs})",
             ha='center', va='bottom', fontsize=F4_TICK, fontweight='bold', color=F4_COLOR_PRNG)
    ax1.text(1, qrng_rate + 3, f"{qrng_rate:.0f}%\n({qrng_matches}/{total_runs})",
             ha='center', va='bottom', fontsize=F4_TICK, fontweight='bold', color=F4_COLOR_QRNG)

    ax1.set_title('(a)', fontsize=F4_TITLE, fontweight='bold', loc='left')

    # =========================================================================
    # Panel (b): Memory Usage - non-zero baseline
    # =========================================================================
    categories_c = ['PRNG', 'QRNG']
    mem_vals = [prng_mem, qrng_mem]
    colors_c = [F4_COLOR_PRNG, F4_COLOR_QRNG]

    # Determine y-axis range: start from a sensible baseline
    y_min = int(min(mem_vals) - 10)  # e.g., ~1296
    y_max = int(max(mem_vals) + 15)  # e.g., ~1329, extra room for annotation

    bars_c = ax2.bar(categories_c, mem_vals, width=0.5, color=colors_c, edgecolor='none', alpha=0.9)
    ax2.set_ylabel('Memory (MB)', fontsize=F4_LABEL)
    ax2.set_ylim(y_min, y_max)
    ax2.tick_params(axis='both', labelsize=F4_TICK)

    # Value labels
    for bar, val in zip(bars_c, mem_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                 f"{val:.1f}", ha='center', va='bottom',
                 fontsize=F4_TICK, fontweight='bold')

    # Annotate difference above QRNG bar
    y_annot_c = qrng_mem + (y_max - y_min) * 0.14
    ax2.text(1, y_annot_c, f"+{mem_diff:.1f} MB",
             ha='center', va='bottom', fontsize=F4_TICK,
             fontweight='bold', color='#555555')

    # Break indicator on y-axis (diagonal lines to indicate non-zero baseline)
    d = 0.015
    kwargs_break = dict(transform=ax2.transAxes, color='k', clip_on=False, lw=0.8)
    ax2.plot((-d, +d), (-d, +d), **kwargs_break)
    ax2.plot((-d, +d), (-d + 0.02, +d + 0.02), **kwargs_break)

    ax2.set_title('(b)', fontsize=F4_TITLE, fontweight='bold', loc='left')

    # =========================================================================
    # Panel (c): Latency Comparison - Grouped bar chart (full width)
    # =========================================================================
    metrics = ['Median', 'P95', 'Mean']
    prng_vals = [prng_median, prng_p95, prng_mean]
    qrng_vals = [qrng_median, qrng_p95, qrng_mean]
    pct_changes = [pct_median, pct_p95, pct_mean]

    x = np.arange(len(metrics))
    bar_w = 0.32

    bars_prng = ax3.bar(x - bar_w/2, prng_vals, bar_w, color=F4_COLOR_PRNG,
                        edgecolor='none', alpha=0.9, label='PRNG')
    bars_qrng = ax3.bar(x + bar_w/2, qrng_vals, bar_w, color=F4_COLOR_QRNG,
                        edgecolor='none', alpha=0.9, label='QRNG')

    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, fontsize=F4_TICK)
    ax3.set_ylabel('Latency (ms)', fontsize=F4_LABEL)
    ax3.tick_params(axis='both', labelsize=F4_TICK)

    # Dynamic y-limit
    max_val = max(max(prng_vals), max(qrng_vals))
    ax3.set_ylim(0, max_val * 1.35)

    # Value labels on each bar
    for bar in bars_prng:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, h + 0.2,
                 f"{h:.2f}", ha='center', va='bottom', fontsize=F4_TICK, color=F4_COLOR_PRNG)
    for bar in bars_qrng:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, h + 0.2,
                 f"{h:.2f}", ha='center', va='bottom', fontsize=F4_TICK, color=F4_COLOR_QRNG)

    # Percentage change annotations above each group
    for i, pct in enumerate(pct_changes):
        y_annot = max(prng_vals[i], qrng_vals[i]) + max_val * 0.12
        ax3.text(x[i], y_annot, f"+{pct:.1f}%",
                 ha='center', va='bottom', fontsize=F4_TICK,
                 fontweight='bold', color='#555555')

    ax3.legend(fontsize=F4_TICK, frameon=False, loc='upper left')
    ax3.set_title('(c)', fontsize=F4_TITLE, fontweight='bold', loc='left')

    save_figure(fig, "figure4_defense_performance")


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("NMI Paper Figure Generation")
    print("=" * 60)
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Figures dir: {FIGURES_DIR}")

    draw_figure2()
    # draw_figure3()  # Commented out — Figure 3 no longer needed
    draw_figure4()

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)

    # List generated files
    print("\nGenerated files:")
    for f in sorted(os.listdir(FIGURES_DIR)):
        fpath = os.path.join(FIGURES_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
