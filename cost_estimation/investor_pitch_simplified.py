import matplotlib.pyplot as plt
import numpy as np
import os

# Headless backend
plt.switch_backend('Agg')

def create_white_background_pitch_assets():
    # SETTINGS
    MAX_PATIENTS = 1000
    patient_counts = np.linspace(0, MAX_PATIENTS, 100)
    
    # ---------------------------------------------------------
    # 1. COST MOAT DATA
    # ---------------------------------------------------------
    COST_1Q = 1.00    # $1.00/pt for 1 question
    COST_5Q = 5.00    # $5.00/pt for 5 questions
    COST_RP = 0.20    # $0.20/pt for ReviewPyper (Total)
    
    y_cost_1q = patient_counts * COST_1Q
    y_cost_5q = patient_counts * COST_5Q
    y_cost_rp = patient_counts * COST_RP

    # ---------------------------------------------------------
    # 2. TIME MOAT DATA (Units: Hours)
    # ---------------------------------------------------------
    TIME_HUMAN = 0.25
    TIME_STD_AI = 0.05
    TIME_RP = 0.005
    
    y_time_human = patient_counts * TIME_HUMAN
    y_time_std = patient_counts * TIME_STD_AI
    y_time_rp = patient_counts * TIME_RP

    # ---------------------------------------------------------
    # PLOTTING (White Background / Clean Aesthetic)
    # ---------------------------------------------------------
    plt.style.use('default')
    fig, axes = plt.subplots(2, 1, figsize=(14, 20))
    fig.patch.set_facecolor('#ffffff')

    # Colors
    color_std_dark = '#cc0044' # Darker red for light background
    color_std_light = '#ff3366'
    color_rp = '#008877'       # Darker teal for light background
    color_human = '#666666'

    # --- TOP PANEL: COST ---
    ax1 = axes[0]
    ax1.set_facecolor('#ffffff')
    ax1.grid(color='#eeeeee', linestyle='-', linewidth=1, alpha=0.8)
    
    ax1.plot(patient_counts, y_cost_5q, color=color_std_dark, lw=3, ls='--', alpha=0.4, label='Standard LLM API (5 Questions)')
    ax1.plot(patient_counts, y_cost_1q, color=color_std_light, lw=4, label='Standard LLM API (1 Question)')
    ax1.plot(patient_counts, y_cost_rp, color='#333333', lw=6, label='ReviewPyper Core') # Dark Gray / Black for RP
    ax1.fill_between(patient_counts, y_cost_rp, y_cost_5q, color='#333333', alpha=0.03)
    
    ax1.set_title('THE COST MOAT: Scale Without Spending', fontsize=28, fontweight='bold', pad=30, color='#222222')
    ax1.set_ylabel('Total Project Cost (USD)', fontsize=16, color='#555555')
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'${int(x):,}'))
    ax1.legend(loc='upper left', fontsize=16, framealpha=0.0, labelcolor='#333333')
    
    # Cost Badges
    ax1.text(MAX_PATIENTS, y_cost_5q[-1], f'${y_cost_5q[-1]:,.0f}', color=color_std_dark, fontweight='bold', fontsize=16, ha='right', va='bottom')
    ax1.text(MAX_PATIENTS, y_cost_rp[-1], f'${y_cost_rp[-1]:,.0f}', color='#333333', fontweight='bold', fontsize=22, ha='right', va='top', bbox=dict(facecolor='#eeeeee', alpha=0.5, boxstyle='round,pad=0.5', edgecolor='none'))

    # --- BOTTOM PANEL: TIME ---
    ax2 = axes[1]
    ax2.set_facecolor('#ffffff')
    ax2.grid(color='#eeeeee', linestyle='-', linewidth=1, alpha=0.8)
    
    ax2.plot(patient_counts, y_time_human, color=color_human, lw=3, ls=':', label='Manual Clinical Review')
    ax2.plot(patient_counts, y_time_std, color=color_std_light, lw=4, label='Standard LLM API')
    ax2.plot(patient_counts, y_time_rp, color=color_rp, lw=6, label='ReviewPyper')
    ax2.fill_between(patient_counts, y_time_rp, y_time_human, color=color_rp, alpha=0.03)

    ax2.set_title('THE TIME MOAT: From Weeks to Minutes', fontsize=28, fontweight='bold', pad=30, color='#222222')
    ax2.set_ylabel('Project Latency (Total Hours)', fontsize=16, color='#555555')
    ax2.set_xlabel('Scale: Number of Patient Records', fontsize=18, color='#555555', labelpad=20)
    ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}h'))
    ax2.legend(loc='upper left', fontsize=16, framealpha=0.0, labelcolor='#333333')

    # Time Badges
    ax2.text(MAX_PATIENTS, y_time_human[-1], f'{int(y_time_human[-1])} Hours', color=color_human, fontweight='bold', fontsize=16, ha='right', va='bottom')
    ax2.text(MAX_PATIENTS, y_time_rp[-1], f'{int(y_time_rp[-1]*60)} Mins', color=color_rp, fontweight='bold', fontsize=22, ha='right', va='top', bbox=dict(facecolor='#e6f9f7', alpha=0.5, boxstyle='round,pad=0.5', edgecolor='none'))

    # THE ROI PUNCHLINE (High Visibility)
    time_advantage = y_time_human[-1] / y_time_rp[-1]
    ax2.text(0.5, 0.45, f'{int(time_advantage)}x Speed Advantage', transform=ax2.transAxes, 
            color='#1a6d63', fontsize=26, fontweight='bold', 
            bbox=dict(facecolor='#e6f9f7', alpha=0.8, boxstyle='round,pad=1.2', edgecolor='#008877'),
            ha='center')

    for ax in axes:
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        for spine in ax.spines.values():
            spine.set_color('#cccccc')
            spine.set_linewidth(1.5)

    plt.tight_layout(pad=8.0)
    base_path = os.path.join(os.path.dirname(__file__), 'ReviewPyper_investorPitch')
    plt.savefig(base_path + '.png', dpi=300)
    plt.savefig(base_path + '.svg')
    print(f"White-background investor pitch visual saved as PNG and SVG: {base_path}")

if __name__ == "__main__":
    create_white_background_pitch_assets()
