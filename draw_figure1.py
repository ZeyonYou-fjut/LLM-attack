"""
Figure 1: PRNG Attack & QRNG Defense Schematic
Nature Machine Intelligence style figure.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
import os

# === Nature MI Style Configuration ===
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 6,
    'axes.linewidth': 0.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.dpi': 300,
    'savefig.transparent': False,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
})

# === Color Palette (Okabe-Ito) ===
C_ATTACK = '#CC0000'   # red - attack/PRNG vulnerability
C_DEFENSE = '#029E73'  # green - defense/QRNG
C_NEUTRAL = '#0173B2'  # blue - normal flow
C_ACCENT = '#DE8F05'   # orange - emphasis
C_BG = 'white'
C_TEXT = 'black'
C_LIGHTGRAY = '#F0F0F0'


def draw_box(ax, xy, width, height, text, color=C_NEUTRAL, fontsize=5.5,
             linestyle='-', linewidth=0.8, alpha=1.0, textcolor='white',
             bold=False):
    """Draw a rounded rectangle box with centered text."""
    x, y = xy
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor=color if linestyle == '-' else color,
        linewidth=linewidth, linestyle=linestyle, alpha=alpha
    )
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + width / 2, y + height / 2, text,
            ha='center', va='center', fontsize=fontsize,
            color=textcolor, weight=weight, zorder=5)
    return box


def draw_arrow(ax, start, end, color=C_TEXT, linewidth=0.8, style='-',
               arrowstyle='->', connectionstyle='arc3,rad=0'):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=arrowstyle,
        connectionstyle=connectionstyle,
        color=color, linewidth=linewidth,
        linestyle=style,
        mutation_scale=8,
        zorder=3
    )
    ax.add_patch(arrow)
    return arrow


def draw_figure1a(ax):
    """Draw PRNG Attack (DeterministicHijacker) schematic."""
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis('off')

    # --- Top flow: normal inference pipeline ---
    flow_y = 0.80
    box_h = 0.09
    boxes = [
        (0.0, 'User\nPrompt', 0.12),
        (0.16, 'LLM\nModel', 0.12),
        (0.32, 'Logits', 0.10),
        (0.46, 'Softmax', 0.10),
        (0.60, 'Prob\nDist.', 0.10),
        (0.74, 'Sampling\nLayer', 0.12),
        (0.90, 'Output\nToken', 0.10),
    ]

    for i, (bx, label, bw) in enumerate(boxes):
        if i == len(boxes) - 1:
            # Last box: hijacked token in red
            draw_box(ax, (bx, flow_y), bw, box_h, label,
                     color=C_ATTACK, fontsize=5, textcolor='white', bold=True)
        else:
            draw_box(ax, (bx, flow_y), bw, box_h, label,
                     color=C_NEUTRAL, fontsize=5, textcolor='white')

    # Arrows between boxes (use triangle arrow character for Nature style)
    arrow_positions = [
        (0.12, 0.16), (0.28, 0.32), (0.42, 0.46),
        (0.56, 0.60), (0.70, 0.74), (0.86, 0.90)
    ]
    for (x1, x2) in arrow_positions:
        mid_x = (x1 + x2) / 2
        ax.text(mid_x, flow_y + box_h / 2, '>', ha='center', va='center',
                fontsize=6, color=C_TEXT, weight='bold', zorder=4)

    # --- Bottom: Attacker injection path ---
    # PRNG box
    prng_x, prng_y = 0.52, 0.50
    prng_w, prng_h = 0.30, 0.12
    # Dashed red border
    prng_box = FancyBboxPatch(
        (prng_x, prng_y), prng_w, prng_h,
        boxstyle="round,pad=0.02",
        facecolor='#FFF0F0', edgecolor=C_ATTACK,
        linewidth=1.0, linestyle='--'
    )
    ax.add_patch(prng_box)
    ax.text(prng_x + prng_w / 2, prng_y + prng_h * 0.7,
            'PRNG (MT19937)', ha='center', va='center',
            fontsize=5, color=C_ATTACK, weight='bold')
    ax.text(prng_x + prng_w / 2, prng_y + prng_h * 0.3,
            'seed → r1, r2, r3...', ha='center', va='center',
            fontsize=4.5, color=C_ATTACK)

    # Arrow from PRNG to Sampling Layer
    draw_arrow(ax, (prng_x + prng_w / 2, prng_y + prng_h),
               (0.80, flow_y), color=C_ATTACK, linewidth=0.8, style='--',
               connectionstyle='arc3,rad=-0.15')

    # Attacker box
    atk_x, atk_y = 0.50, 0.22
    atk_w, atk_h = 0.32, 0.16
    atk_box = FancyBboxPatch(
        (atk_x, atk_y), atk_w, atk_h,
        boxstyle="round,pad=0.02",
        facecolor='#FFF0F0', edgecolor=C_ATTACK,
        linewidth=1.2, linestyle='-'
    )
    ax.add_patch(atk_box)
    ax.text(atk_x + atk_w / 2, atk_y + atk_h * 0.78,
            'Attacker: knows seed', ha='center', va='center',
            fontsize=5, color=C_ATTACK, weight='bold')
    ax.text(atk_x + atk_w / 2, atk_y + atk_h * 0.5,
            '→ precomputes CDF intervals', ha='center', va='center',
            fontsize=4.5, color=C_ATTACK)
    ax.text(atk_x + atk_w / 2, atk_y + atk_h * 0.22,
            '→ forces target token', ha='center', va='center',
            fontsize=4.5, color=C_ATTACK)

    # Arrow from attacker to PRNG (curved for aesthetics)
    draw_arrow(ax, (atk_x + atk_w / 2, atk_y + atk_h),
               (prng_x + prng_w / 2, prng_y), color=C_ATTACK, linewidth=0.8,
               connectionstyle='arc3,rad=0.2')

    # --- CDF inset (small) ---
    # Draw a mini CDF diagram on the left side
    inset_ax_x, inset_ax_y = 0.02, 0.20
    inset_w, inset_h = 0.26, 0.28
    # Border
    inset_border = FancyBboxPatch(
        (inset_ax_x, inset_ax_y), inset_w, inset_h,
        boxstyle="round,pad=0.01",
        facecolor=C_LIGHTGRAY, edgecolor='#AAAAAA',
        linewidth=0.5, linestyle='-'
    )
    ax.add_patch(inset_border)
    ax.text(inset_ax_x + inset_w / 2, inset_ax_y + inset_h - 0.03,
            'CDF Sampling', ha='center', va='center',
            fontsize=4.5, color=C_TEXT, weight='bold')

    # Draw simple CDF steps
    cdf_left = inset_ax_x + 0.03
    cdf_bottom = inset_ax_y + 0.03
    cdf_w = inset_w - 0.06
    cdf_h = inset_h - 0.10
    steps = [0, 0.15, 0.35, 0.55, 0.75, 0.90, 1.0]
    for i in range(len(steps) - 1):
        y_low = cdf_bottom + steps[i] * cdf_h
        y_high = cdf_bottom + steps[i + 1] * cdf_h
        x_pos = cdf_left + (i / (len(steps) - 1)) * cdf_w
        x_next = cdf_left + ((i + 1) / (len(steps) - 1)) * cdf_w
        # Highlight the hijacked interval
        if i == 2:  # hijacked
            ax.fill_between([x_pos, x_next], y_low, y_high,
                            color=C_ATTACK, alpha=0.3, zorder=2)
            ax.plot([x_pos, x_next], [y_high, y_high],
                    color=C_ATTACK, linewidth=0.8, zorder=3)
            ax.plot([x_pos, x_next], [y_low, y_low],
                    color=C_ATTACK, linewidth=0.8, zorder=3)
        else:
            ax.plot([x_pos, x_next], [y_high, y_high],
                    color=C_NEUTRAL, linewidth=0.5, alpha=0.6, zorder=2)

    ax.text(cdf_left + cdf_w * 0.45, cdf_bottom + cdf_h * 0.48,
            'hijacked\ninterval', ha='center', va='center',
            fontsize=3.5, color=C_ATTACK, weight='bold')

    # "Hijacked!" label near output
    ax.text(0.95, flow_y + box_h + 0.02, 'Hijacked!',
            ha='center', va='bottom', fontsize=5.5, color=C_ATTACK,
            weight='bold')


def draw_figure1b(ax):
    """Draw QRNG Defense schematic."""
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis('off')

    # --- Top flow: normal inference pipeline ---
    flow_y = 0.80
    box_h = 0.09
    boxes = [
        (0.0, 'User\nPrompt', 0.12),
        (0.16, 'LLM\nModel', 0.12),
        (0.32, 'Logits', 0.10),
        (0.46, 'Softmax', 0.10),
        (0.60, 'Prob\nDist.', 0.10),
        (0.74, 'Sampling\nLayer', 0.12),
        (0.90, 'Output\nToken', 0.10),
    ]

    for i, (bx, label, bw) in enumerate(boxes):
        if i == len(boxes) - 1:
            # Last box: safe token in green
            draw_box(ax, (bx, flow_y), bw, box_h, label,
                     color=C_DEFENSE, fontsize=5, textcolor='white', bold=True)
        else:
            draw_box(ax, (bx, flow_y), bw, box_h, label,
                     color=C_NEUTRAL, fontsize=5, textcolor='white')

    # Arrows (use triangle arrow character for Nature style)
    arrow_positions = [
        (0.12, 0.16), (0.28, 0.32), (0.42, 0.46),
        (0.56, 0.60), (0.70, 0.74), (0.86, 0.90)
    ]
    for (x1, x2) in arrow_positions:
        mid_x = (x1 + x2) / 2
        ax.text(mid_x, flow_y + box_h / 2, '>', ha='center', va='center',
                fontsize=6, color=C_TEXT, weight='bold', zorder=4)

    # --- QRNG box (green, solid) ---
    qrng_x, qrng_y = 0.50, 0.50
    qrng_w, qrng_h = 0.32, 0.12
    qrng_box = FancyBboxPatch(
        (qrng_x, qrng_y), qrng_w, qrng_h,
        boxstyle="round,pad=0.02",
        facecolor='#E8F8F0', edgecolor=C_DEFENSE,
        linewidth=1.2, linestyle='-'
    )
    ax.add_patch(qrng_box)
    ax.text(qrng_x + qrng_w / 2, qrng_y + qrng_h * 0.7,
            'QRNG600 PCIe (Hardware)', ha='center', va='center',
            fontsize=4.8, color=C_DEFENSE, weight='bold')
    ax.text(qrng_x + qrng_w / 2, qrng_y + qrng_h * 0.3,
            'quantum noise → true random', ha='center', va='center',
            fontsize=4.5, color=C_DEFENSE)

    # Arrow from QRNG to Sampling Layer
    draw_arrow(ax, (qrng_x + qrng_w / 2, qrng_y + qrng_h),
               (0.80, flow_y), color=C_DEFENSE, linewidth=1.0,
               connectionstyle='arc3,rad=-0.15')

    # --- Attacker box (blocked) ---
    atk_x, atk_y = 0.50, 0.20
    atk_w, atk_h = 0.34, 0.16
    atk_box = FancyBboxPatch(
        (atk_x, atk_y), atk_w, atk_h,
        boxstyle="round,pad=0.02",
        facecolor='#FFF5F5', edgecolor='#CCCCCC',
        linewidth=0.8, linestyle='--'
    )
    ax.add_patch(atk_box)
    ax.text(atk_x + atk_w / 2, atk_y + atk_h * 0.78,
            'Attacker: CANNOT predict', ha='center', va='center',
            fontsize=5, color='#999999', weight='bold')
    ax.text(atk_x + atk_w / 2, atk_y + atk_h * 0.5,
            '→ CDF interval unknown', ha='center', va='center',
            fontsize=4.5, color='#999999')
    ax.text(atk_x + atk_w / 2, atk_y + atk_h * 0.22,
            '→ hijacking fails', ha='center', va='center',
            fontsize=4.5, color='#999999')

    # Big red X over attacker path
    cross_cx = atk_x + atk_w / 2
    cross_cy = atk_y + atk_h / 2
    ax.text(atk_x + atk_w + 0.02, atk_y + atk_h / 2, 'X',
            ha='center', va='center', fontsize=14, color=C_ATTACK,
            weight='bold', zorder=10)

    # Blocked arrow (faded, with X, curved)
    draw_arrow(ax, (atk_x + atk_w / 2, atk_y + atk_h),
               (qrng_x + qrng_w / 2, qrng_y),
               color='#CCCCCC', linewidth=0.6, style='--',
               connectionstyle='arc3,rad=0.2')

    # --- Uniform distribution inset ---
    inset_ax_x, inset_ax_y = 0.02, 0.20
    inset_w, inset_h = 0.26, 0.28
    inset_border = FancyBboxPatch(
        (inset_ax_x, inset_ax_y), inset_w, inset_h,
        boxstyle="round,pad=0.01",
        facecolor=C_LIGHTGRAY, edgecolor='#AAAAAA',
        linewidth=0.5, linestyle='-'
    )
    ax.add_patch(inset_border)
    ax.text(inset_ax_x + inset_w / 2, inset_ax_y + inset_h - 0.03,
            'Uniform Random Sampling', ha='center', va='center',
            fontsize=4.5, color=C_TEXT, weight='bold')

    # Draw uniform bars
    bar_left = inset_ax_x + 0.03
    bar_bottom = inset_ax_y + 0.03
    bar_area_w = inset_w - 0.06
    bar_area_h = inset_h - 0.10
    n_bars = 8
    bar_w = bar_area_w / (n_bars * 1.5)
    np.random.seed(42)

    for i in range(n_bars):
        x_pos = bar_left + i * (bar_area_w / n_bars)
        h = bar_area_h * (0.7 + 0.3 * np.random.random())
        rect = plt.Rectangle(
            (x_pos, bar_bottom), bar_w, h,
            facecolor=C_DEFENSE, alpha=0.6, edgecolor=C_DEFENSE,
            linewidth=0.3
        )
        ax.add_patch(rect)

    # "?" markers to show unpredictability
    ax.text(bar_left + bar_area_w * 0.5, bar_bottom + bar_area_h * 0.5,
            '?  ?  ?', ha='center', va='center',
            fontsize=6, color=C_DEFENSE, weight='bold', alpha=0.7)

    # "Safe" label near output
    ax.text(0.95, flow_y + box_h + 0.02, 'Safe',
            ha='center', va='bottom', fontsize=5.5, color=C_DEFENSE,
            weight='bold')


# === Main Figure ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
fig.patch.set_facecolor('white')
plt.subplots_adjust(wspace=0.08, left=0.02, right=0.98, top=0.90, bottom=0.02)

# Draw panels
draw_figure1a(ax1)
draw_figure1b(ax2)

# Panel labels (8pt bold lowercase)
ax1.text(-0.02, 1.02, 'a', transform=ax1.transAxes,
         fontsize=8, fontweight='bold', va='bottom', ha='left')
ax2.text(-0.02, 1.02, 'b', transform=ax2.transAxes,
         fontsize=8, fontweight='bold', va='bottom', ha='left')

# Panel titles
ax1.text(0.5, 1.0, 'PRNG Attack (DeterministicHijacker)',
         transform=ax1.transAxes, ha='center', va='bottom',
         fontsize=7, color=C_ATTACK, weight='bold')
ax2.text(0.5, 1.0, 'QRNG Defense',
         transform=ax2.transAxes, ha='center', va='bottom',
         fontsize=7, color=C_DEFENSE, weight='bold')

# Save
output_dir = r'C:\Users\Administrator\Desktop\NMI\figures'
os.makedirs(output_dir, exist_ok=True)

pdf_path = os.path.join(output_dir, 'figure1_attack_defense.pdf')
png_path = os.path.join(output_dir, 'figure1_attack_defense.png')

fig.savefig(pdf_path, format='pdf', dpi=1200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
fig.savefig(png_path, format='png', dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none')

plt.close()
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
