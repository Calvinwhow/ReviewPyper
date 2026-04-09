import matplotlib.pyplot as plt
import numpy as np
import os

# Headless backend
plt.switch_backend('Agg')

def create_stunning_time_moat():
    # DATA CONSTANTS
    TOKENS_PER_RECORD = 15000
    HUMAN_MIN_PER_RECORD = 15.0 
    TPS_STANDARD = 4000
    TPS_RP = 32000
    
    patient_counts = np.linspace(100, 1000000, 100)
    
    def calc_days(p, tps):
        total_tokens = p * TOKENS_PER_RECORD
        return (total_tokens / tps) / 3600 / 24

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor('#0d0e12')
    ax.set_facecolor('#0f0f15')
    
    y_std = np.array([calc_days(p, TPS_STANDARD) for p in patient_counts])
    y_rp = np.array([calc_days(p, TPS_RP) for p in patient_counts])
    
    # 1. Neon Glow Lines (Stacked for neon effect)
    ax.plot(patient_counts, y_std, color='#ff0066', lw=10, alpha=0.1) # Glow
    ax.plot(patient_counts, y_std, color='#ff0066', lw=3, label='Standard Brute-Force AI')
    
    ax.plot(patient_counts, y_rp, color='#00ffcc', lw=12, alpha=0.1) # Glow
    ax.plot(patient_counts, y_rp, color='#00ffcc', lw=5, label='ReviewPyper™ Core Optimization')

    # 2. Shaded Advantage Area
    ax.fill_between(patient_counts, y_rp, y_std, color='#00ffcc', alpha=0.06)

    # 3. Framing & Aesthetics
    ax.set_title('THE THROUGHPUT MOAT: FROM DECADES TO DAYS', fontsize=34, fontweight='bold', pad=50, color='#ffffff')
    ax.set_xlabel('Scale: Number of Patient Records', fontsize=18, color='#888888', labelpad=20)
    ax.set_ylabel('Project Latency (Total Days)', fontsize=18, color='#888888', labelpad=20)
    
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.grid(color='#25262c', linestyle=':', linewidth=1, alpha=0.4)
    
    # 4. End State Badges (Neon Outlines)
    p_last = patient_counts[-1]
    s_last = y_std[-1]
    r_last = y_rp[-1]
    
    # Standard AI Badge
    ax.text(p_last, s_last, f' {s_last:.1f} DAYS ', color='#ff0066', fontsize=18, fontweight='bold',
            ha='right', va='bottom', bbox=dict(facecolor='#0d0e12', edgecolor='#ff0066', boxstyle='round,pad=0.5', lw=2))

    # ReviewPyper Badge
    ax.text(p_last, r_last, f' {r_last:.1f} DAYS ', color='#00ffcc', fontsize=22, fontweight='bold',
            ha='right', va='top', bbox=dict(facecolor='#0d0e12', edgecolor='#00ffcc', boxstyle='round,pad=0.6', lw=3))

    # 5. High-Impact Callouts
    total_human_hours = (p_last * HUMAN_MIN_PER_RECORD) / 60
    total_human_years = total_human_hours / 24 / 365.25
    ax.text(0.05, 0.95, f"MANUAL CLINICIAN REVIEW: {total_human_years:.1f} YEARS", 
            transform=ax.transAxes, color='#ffffff', fontsize=20, fontweight='bold', alpha=0.5, ha='left')

    # 6. Efficiency Badge (Saves X Hours)
    ax.text(0.5, 0.5, f"SAVES {int(total_human_hours - (r_last*24)):,} HOURS\nOF HUMAN LABOR", 
            transform=ax.transAxes, color='white', fontsize=24, fontweight='bold',
            bbox=dict(facecolor='#00ffcc', alpha=0.15, boxstyle='round,pad=1.5', edgecolor='#00ffcc'), ha='center')

    for spine in ax.spines.values():
        spine.set_edgecolor('#25262c')
    
    ax.legend(loc='upper left', fontsize=18, frameon=False, labelcolor='#888888')
    
    plt.tight_layout(pad=6.0)
    save_path = os.path.join(os.path.dirname(__file__), 'ReviewPyper_Time_Moat.png')
    plt.savefig(save_path, dpi=300)
    print(f"Stunning Time Moat saved: {save_path}")

if __name__ == "__main__":
    create_stunning_time_moat()
