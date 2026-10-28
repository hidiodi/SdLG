import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 1.2
})

def draw_exact_sdlg_flow(save_path):
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    # Color palette
    c_input = "#E0F2FE"       # Light blue
    c_input_border = "#0284C7"
    c_swa = "#FEF3C7"         # Light amber
    c_swa_border = "#D97706"
    c_gla = "#DCFCE7"         # Light green
    c_gla_border = "#16A34A"
    c_gate = "#F3E8FF"        # Light purple
    c_gate_border = "#9333EA"
    c_ggs = "#FFEDD5"         # Light orange
    c_ggs_border = "#EA580C"
    c_head = "#FEE2E2"        # Light red
    c_head_border = "#DC2626"
    c_text = "#1E293B"

    # Title
    ax.text(8, 8.1, "SdLG-Exact: Un-truncated FP32 Linear Recurrence Architecture", 
            ha='center', va='center', fontsize=16, fontweight='bold', color=c_text)
    ax.text(8, 7.7, "Full-Rank FP32 State Matrix Accumulation ($M_t \\in \\mathbb{R}^{d_k \\times d_v}$) with Dual SWA Stream ($W=32$) & GGS Shielding", 
            ha='center', va='center', fontsize=11, fontstyle='italic', color="#475569")

    # Box drawer helper
    def box(x, y, w, h, bg, border, title, subtitle="", rx=0.2):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.1,rounding_size={rx}",
                                     facecolor=bg, edgecolor=border, linewidth=1.8)
        ax.add_patch(rect)
        if subtitle:
            ax.text(x + w/2, y + h/2 + 0.15, title, ha='center', va='center', fontsize=11, fontweight='bold', color=c_text)
            ax.text(x + w/2, y + h/2 - 0.18, subtitle, ha='center', va='center', fontsize=8.5, color="#334155")
        else:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center', fontsize=11, fontweight='bold', color=c_text)

    # Arrow drawer helper
    def arrow(x1, y1, x2, y2, label="", style="->", color="#475569", lw=1.5, ls="-"):
        ax.annotate(label, xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls),
                    ha='center', va='center', fontsize=8.5, color="#334155")

    # 1. Input Layer
    box(0.5, 3.2, 2.0, 1.6, c_input, c_input_border, "Observation $x_t$", "$(B, T, D_{\\text{obs}})$")

    # 2. Representation Encoder
    arrow(2.6, 4.0, 3.2, 4.0)
    box(3.2, 3.2, 2.2, 1.6, c_input, c_input_border, "Spatial Encoder", "Conv2D / Dense\n$\\to x_t \\in \\mathbb{R}^{d_{\\text{model}}}$")

    # Split into Streams
    arrow(5.5, 4.3, 6.2, 5.8)
    arrow(5.5, 3.7, 6.2, 2.2)

    # 3. Stream A: Local SWA
    box(6.2, 5.0, 3.2, 1.6, c_swa, c_swa_border, "Local SWA Stream", "Sliding Window ($W=32$)\nHigh-Resolution Visual Keys")

    # 4. Stream B: Exact GLA Memory Core
    box(6.2, 1.4, 3.2, 1.6, c_gla, c_gla_border, "Exact GLA Core", "Parallel Prefix Scan\n$M_t = \\gamma_t M_{t-1} + \\beta_t \\phi(k_t)v_t^\\top$\n$z_t = \\gamma_t z_{t-1} + \\beta_t \\phi(k_t)$")

    # Merging with GTrXL Gates
    arrow(9.5, 5.8, 10.2, 4.3)
    arrow(9.5, 2.2, 10.2, 3.7)

    # 5. GTrXL Residual Integration & FFN
    box(10.2, 3.2, 2.4, 1.6, c_gate, c_gate_border, "GTrXL Gate + FFN", "Gated Residual ($b_{\\text{gru}}=2.0$)\nWide FFN ($d_{\\text{ffc}}=1024$)")

    # Arrow to GGS Decoupling
    arrow(12.7, 4.0, 13.3, 4.0)

    # 6. GGS Actor-Critic Decoupling
    box(13.3, 3.0, 2.2, 2.0, c_ggs, c_ggs_border, "GGS Decoupling", "Actor: $\\text{sg}(x)(1-\\alpha) + x\\alpha$\n$(\\alpha=0.1)$\nCritic: Full Gradients")

    # Heads
    arrow(14.4, 5.1, 14.4, 6.2)
    arrow(14.4, 2.9, 14.4, 1.8)

    box(13.4, 6.2, 2.0, 1.2, c_head, c_head_border, "Policy Head $\\pi(a|s)$", "Action Logits")
    box(13.4, 0.6, 2.0, 1.2, c_head, c_head_border, "Value Head $V(s)$", "Value Bootstrap")

    # Add State Memory Carry Loop
    ax.annotate("", xy=(6.2, 1.7), xytext=(5.6, 0.7),
                arrowprops=dict(arrowstyle="->", color="#16A34A", lw=2, connectionstyle="arc3,rad=-0.3"))
    ax.annotate("", xy=(5.6, 0.7), xytext=(9.4, 1.7),
                arrowprops=dict(arrowstyle="-", color="#16A34A", lw=2, connectionstyle="arc3,rad=-0.3"))
    ax.text(7.8, 0.4, "Carry State $M_{t-1}, z_{t-1} \\to M_t, z_t$ (Un-truncated FP32 Memory)", 
            ha='center', va='center', fontsize=9, fontweight='bold', color="#15803D")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Exact-SdLG flow diagram to {save_path}")

def draw_approx_sdlg_flow(save_path):
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    # Color palette
    c_input = "#E0F2FE"       # Light blue
    c_input_border = "#0284C7"
    c_swa = "#FEF3C7"         # Light amber
    c_swa_border = "#D97706"
    c_approx = "#E0E7FF"      # Light indigo
    c_approx_border = "#4F46E5"
    c_gate = "#F3E8FF"        # Light purple
    c_gate_border = "#9333EA"
    c_ggs = "#FFEDD5"         # Light orange
    c_ggs_border = "#EA580C"
    c_head = "#FEE2E2"        # Light red
    c_head_border = "#DC2626"
    c_text = "#1E293B"

    # Title
    ax.text(8, 8.1, "SdLG-Approx: Low-Rank Rotary Position Projection (RoPE) Architecture", 
            ha='center', va='center', fontsize=16, fontweight='bold', color=c_text)
    ax.text(8, 7.7, "Sub-Quadratic Low-Rank Recurrence ($r \\ll d_k$) with Phase-Rotated Keys/Values & 1.5x Hardware Speedup ($14,520$ SPS)", 
            ha='center', va='center', fontsize=11, fontstyle='italic', color="#475569")

    # Helper box drawer
    def box(x, y, w, h, bg, border, title, subtitle="", rx=0.2):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.1,rounding_size={rx}",
                                     facecolor=bg, edgecolor=border, linewidth=1.8)
        ax.add_patch(rect)
        if subtitle:
            ax.text(x + w/2, y + h/2 + 0.15, title, ha='center', va='center', fontsize=11, fontweight='bold', color=c_text)
            ax.text(x + w/2, y + h/2 - 0.18, subtitle, ha='center', va='center', fontsize=8.5, color="#334155")
        else:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center', fontsize=11, fontweight='bold', color=c_text)

    # Arrow helper
    def arrow(x1, y1, x2, y2, label="", style="->", color="#475569", lw=1.5, ls="-"):
        ax.annotate(label, xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls),
                    ha='center', va='center', fontsize=8.5, color="#334155")

    # 1. Input Layer
    box(0.5, 3.2, 2.0, 1.6, c_input, c_input_border, "Observation $x_t$", "$(B, T, D_{\\text{obs}})$")

    # 2. Representation Encoder
    arrow(2.6, 4.0, 3.2, 4.0)
    box(3.2, 3.2, 2.2, 1.6, c_input, c_input_border, "Spatial Encoder", "Conv2D / Dense\n$\\to x_t \\in \\mathbb{R}^{d_{\\text{model}}}$")

    # Split into Streams
    arrow(5.5, 4.3, 6.2, 5.8)
    arrow(5.5, 3.7, 6.2, 2.2)

    # 3. Stream A: Local SWA
    box(6.2, 5.0, 3.2, 1.6, c_swa, c_swa_border, "Local SWA Stream", "Sliding Window ($W=32$)\nHigh-Resolution Visual Keys")

    # 4. Stream B: Approx Low-Rank RoPE Core
    box(6.2, 1.4, 3.2, 1.6, c_approx, c_approx_border, "Approx RoPE Core", "Low-Rank Projection ($r \\ll d_k$)\nRoPE Phase Rotation $R_t \\phi(k_t)$\nState Footprint $\\mathcal{O}(r \\cdot d_{\\text{model}})$")

    # Merging with GTrXL Gates
    arrow(9.5, 5.8, 10.2, 4.3)
    arrow(9.5, 2.2, 10.2, 3.7)

    # 5. GTrXL Residual Integration & FFN
    box(10.2, 3.2, 2.4, 1.6, c_gate, c_gate_border, "GTrXL Gate + FFN", "Gated Residual ($b_{\\text{gru}}=2.0$)\nWide FFN ($d_{\\text{ffc}}=1024$)")

    # Arrow to GGS Decoupling
    arrow(12.7, 4.0, 13.3, 4.0)

    # 6. GGS Actor-Critic Decoupling
    box(13.3, 3.0, 2.2, 2.0, c_ggs, c_ggs_border, "GGS Decoupling", "Actor: $\\text{sg}(x)(1-\\alpha) + x\\alpha$\n$(\\alpha=0.1)$\nCritic: Full Gradients")

    # Heads
    arrow(14.4, 5.1, 14.4, 6.2)
    arrow(14.4, 2.9, 14.4, 1.8)

    box(13.4, 6.2, 2.0, 1.2, c_head, c_head_border, "Policy Head $\\pi(a|s)$", "Action Logits")
    box(13.4, 0.6, 2.0, 1.2, c_head, c_head_border, "Value Head $V(s)$", "Value Bootstrap")

    # Add State Memory Carry Loop
    ax.annotate("", xy=(6.2, 1.7), xytext=(5.6, 0.7),
                arrowprops=dict(arrowstyle="->", color="#4F46E5", lw=2, connectionstyle="arc3,rad=-0.3"))
    ax.annotate("", xy=(5.6, 0.7), xytext=(9.4, 1.7),
                arrowprops=dict(arrowstyle="-", color="#4F46E5", lw=2, connectionstyle="arc3,rad=-0.3"))
    ax.text(7.8, 0.4, "Carry State $\\tilde{M}_{t-1} \\to \\tilde{M}_t$ (Low-Rank Compressed Memory)", 
            ha='center', va='center', fontsize=9, fontweight='bold', color="#4338CA")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Approx-SdLG flow diagram to {save_path}")

if __name__ == "__main__":
    p1 = "papers/latex/icml_style/plots/exact_sdlg_flow_diagram.png"
    p2 = "papers/latex/icml_style/plots/approx_sdlg_flow_diagram.png"
    draw_exact_sdlg_flow(p1)
    draw_approx_sdlg_flow(p2)
