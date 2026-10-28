import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_sdlg_flowchart(save_paths):
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'axes.edgecolor': '#cbd5e1',
        'axes.linewidth': 1.2
    })

    fig, ax = plt.subplots(figsize=(15, 7.5), dpi=300)
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 8.6)
    ax.axis('off')

    c_input = "#F0F9FF"
    c_input_border = "#0284C7"
    c_swa = "#FEF3C7"
    c_swa_border = "#D97706"
    c_gla = "#DCFCE7"
    c_gla_border = "#16A34A"
    c_gate = "#F3E8FF"
    c_gate_border = "#9333EA"
    c_ggs = "#FFEDD5"
    c_ggs_border = "#EA580C"
    c_head = "#FEE2E2"
    c_head_border = "#DC2626"
    c_text = "#0F172A"

    ax.text(8.1, 8.15, "SdLG: Sequential Decoupled Local-Global Architecture for POMDP RL", 
            ha='center', va='center', fontsize=15.5, fontweight='bold', color=c_text)
    ax.text(8.1, 7.72, "Decoupled Local SWA & Global GLA Scan with GGS Trunk Shielding ($\\alpha=0.1$) and GTrXL Gating", 
            ha='center', va='center', fontsize=11, fontstyle='italic', color="#475569")

    def draw_box(x, y, w, h, bg, border, title, subtitle="", rx=0.25):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.08,rounding_size={rx}",
                                     facecolor=bg, edgecolor=border, linewidth=2.0)
        ax.add_patch(rect)
        if subtitle:
            ax.text(x + w/2, y + h/2 + 0.22, title, ha='center', va='center', fontsize=11, fontweight='bold', color=c_text)
            ax.text(x + w/2, y + h/2 - 0.24, subtitle, ha='center', va='center', fontsize=9, color="#334155")
        else:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center', fontsize=11, fontweight='bold', color=c_text)

    def draw_arrow(x1, y1, x2, y2, label="", style="->", color="#475569", lw=1.8, ls="-"):
        ax.annotate(label, xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls),
                    ha='center', va='center', fontsize=9, color="#1E293B")

    draw_box(0.5, 3.2, 1.8, 1.6, c_input, c_input_border, "Observation $x_t$", "$(B, T, D_{\\text{obs}})$")

    draw_arrow(2.38, 4.0, 3.0, 4.0)
    draw_box(3.0, 3.2, 2.3, 1.6, c_input, c_input_border, "Dense MLP Trunk", "Linear $\\to$ ReLU $\\to$ Linear\n$\\to x_t \\in \\mathbb{R}^{d_{\\text{model}}}$")

    draw_arrow(5.38, 4.35, 6.1, 5.75)
    draw_arrow(5.38, 3.65, 6.1, 2.25)

    draw_box(6.1, 4.9, 3.4, 1.7, c_swa, c_swa_border, "Local SWA Stream", "Sliding Window ($W=16$)\nLocal Temporal Context\nPrevents Recency Interference")

    draw_box(6.1, 1.3, 3.4, 1.9, c_gla, c_gla_border, "Global GLA Scan Core", "Parallel Prefix Scan\n$M_t = \\gamma_t M_{t-1} + \\beta_t \\phi(k_t)v_t^\\top$\n$z_t = \\gamma_t z_{t-1} + \\beta_t \\phi(k_t)$")

    draw_arrow(9.58, 5.75, 10.3, 4.35)
    draw_arrow(9.58, 2.25, 10.3, 3.65)

    draw_box(10.3, 3.2, 2.4, 1.6, c_gate, c_gate_border, "GTrXL Gate + FFN", "Gated Residual ($b_{\\text{gru}}=2.0$)\nWide FFN ($d_{\\text{ffc}}=1024$)")

    draw_arrow(12.78, 4.0, 13.4, 4.0)

    draw_box(13.4, 3.0, 2.3, 2.0, c_ggs, c_ggs_border, "GGS Shielding", "Actor: $\\text{sg}(x)(1-\\alpha) + x\\alpha$\n$(\\alpha=0.1)$\nCritic: Full Gradients")

    draw_arrow(14.55, 5.08, 14.55, 6.15)
    draw_arrow(14.55, 2.92, 14.55, 1.85)

    draw_box(13.5, 6.15, 2.1, 1.2, c_head, c_head_border, "Policy Head $\\pi(a|s)$", "Action Distribution")
    draw_box(13.5, 0.65, 2.1, 1.2, c_head, c_head_border, "Value Head $V(s)$", "Critic Baseline")

    ax.annotate("", xy=(6.1, 1.6), xytext=(5.4, 0.6),
                arrowprops=dict(arrowstyle="->", color="#16A34A", lw=2.2, connectionstyle="arc3,rad=-0.3"))
    ax.annotate("", xy=(5.4, 0.6), xytext=(9.5, 1.6),
                arrowprops=dict(arrowstyle="-", color="#16A34A", lw=2.2, connectionstyle="arc3,rad=-0.3"))
    ax.text(7.8, 0.35, "Recurrent Carry State $M_{t-1}, z_{t-1} \\to M_t, z_t$ (Un-truncated FP32 / Low-Rank)", 
            ha='center', va='center', fontsize=9.5, fontweight='bold', color="#15803D")

    plt.tight_layout()
    for p in save_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        plt.savefig(p, bbox_inches='tight', dpi=300)
        print(f"Saved figure to {p}")
    plt.close()

if __name__ == "__main__":
    paths = [
        "papers/latex/icmlV2/plots/sdlgflowchart.png",
        "papers/latex/icml_style/plots/sdlgflowchart.png"
    ]
    generate_sdlg_flowchart(paths)
