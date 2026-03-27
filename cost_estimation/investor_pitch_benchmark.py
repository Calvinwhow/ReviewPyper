import matplotlib.pyplot as plt
import numpy as np
import os

def create_investor_ready_pitch_v2():
    # ASSUMPTIONS derived from local_test_analyze
    # 4.4 MB file / 74 records ≈ 60,000 chars/record
    # 1 token ≈ 4 chars => 15,000 tokens per clinical history
    TOKENS_PER_PATIENT = 15000  
    
    # SCALE: From clinic pilot (100 patients) to hospital system (1,000,000 patients)
    patient_counts = np.linspace(100, 1000000, 50) 
    OVERHEAD = 1000 # Recurring question/prompt tax per chunk

    # ERA SPECIFICATIONS
    eras = {
        "Present: Health System Infrastructure (GPT-4o)": {
            "p_high": 5.00, "p_low": 0.15, 
            "ctx_high": 128000, "ctx_low": 32000
        },
        "Future: Global Research Scale (GPT-5.1+)": {
            "p_high": 1.25, "p_low": 0.05, 
            "ctx_high": 272000, "ctx_low": 272000
        }
    }

    def calc_cost(num_patients, era_key, is_rp):
        spec = eras[era_key]
        total_tokens = num_patients * TOKENS_PER_PATIENT
        
        if is_rp:
            # Pass 1: Global Scan (Nano-Model)
            num_scan_chunks = np.ceil(total_tokens / spec["ctx_low"])
            scan_cost = (total_tokens + num_scan_chunks * OVERHEAD) * (spec["p_low"] / 1e6)
            
            # Pass 2: Precision hit (Selected High-Value Content)
            relevant_tokens = total_tokens * 0.05
            num_ext_chunks = np.ceil(relevant_tokens / spec["ctx_high"])
            ext_cost = (relevant_tokens + num_ext_chunks * OVERHEAD) * (spec["p_high"] / 1e6)
            return scan_cost + ext_cost
        else:
            # Traditional Brute Force:
            num_chunks = np.ceil(total_tokens / spec["ctx_high"])
            return (total_tokens + num_chunks * OVERHEAD) * (spec["p_high"] / 1e6)

    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 1, figsize=(16, 20))
    fig.patch.set_facecolor('#0d0e12')

    for i, (era_name, spec) in enumerate(eras.items()):
        ax = axes[i]
        y_b = [calc_cost(p, era_name, False) for p in patient_counts]
        y_r = [calc_cost(p, era_name, True) for p in patient_counts]
        
        ax.set_facecolor('#0f0f15')
        ax.grid(color='#25262c', linestyle='-', linewidth=1, alpha=0.3)
        
        # Plot linear lines
        ax.plot(patient_counts, y_b, color='#ff0066', lw=4, label='Standard Brute Force Cost Scaling', marker='o', markevery=8)
        ax.plot(patient_counts, y_r, color='#00ffcc', lw=6, label='ReviewPyper™ Core Optimization Advantage', marker='s', markevery=8)
        
        # Shade the gap
        ax.fill_between(patient_counts, y_r, y_b, color='#00ffcc', alpha=0.08)
        
        ax.set_title(f'{era_name}', fontsize=28, fontweight='bold', color='white', pad=35)
        ax.set_ylabel('Total Project Cost (USD)', fontsize=16, color='#888888')
        
        # Axe formatters
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:,.1f}M' if x >= 1e6 else f'${x:,.0f}'))

        # CONTAINED PRICE BADGES
        # Brute Force Price Badge
        ax.text(patient_counts[48], y_b[48], f'${y_b[48]:,.0f}', color='#ff0066', 
                fontweight='bold', fontsize=18, ha='right', va='bottom',
                bbox=dict(facecolor='#ff0066', alpha=0.15, boxstyle='round,pad=0.5'))

        # ReviewPyper Price Badge
        ax.text(patient_counts[48], y_r[48], f'${y_r[48]:,.0f}', color='#00ffcc', 
                fontweight='bold', fontsize=18, ha='right', va='top',
                bbox=dict(facecolor='#00ffcc', alpha=0.15, boxstyle='round,pad=0.5'))

        # Standard legend
        ax.legend(loc='upper left', fontsize=18, framealpha=0.1)
        
        # Advantage Badge (Centered visually in the diverging gap)
        margin = y_b[-1] / y_r[-1]
        ax.text(0.5, 0.45, f'{margin:.1f}x ROI Advantage', transform=ax.transAxes, color='white', fontsize=24, fontweight='bold', 
                bbox=dict(facecolor='#00ffcc', alpha=0.2, boxstyle='round,pad=1.0'), ha='center')

    axes[1].set_xlabel('Scale: Number of Patient Records (Assuming 15k Tokens Avg)', fontsize=18, color='#888888', labelpad=25)
    
    plt.tight_layout(pad=10.0)
    save_path = os.path.join(os.path.dirname(__file__), 'ReviewPyper_Billion_Dollar_Moat.png')
    plt.savefig(save_path, dpi=300)
    print(f"Updated Moat visual with contained price badges saved: {save_path}")

if __name__ == "__main__":
    create_investor_ready_pitch_v2()
