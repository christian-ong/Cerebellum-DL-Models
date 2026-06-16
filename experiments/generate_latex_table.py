import os
import glob
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# Set your "Solved" threshold here! 
# 1e-2 is visually indistinguishable. 1e-3 is mathematical perfection.
# ---------------------------------------------------------
SOLVED_THRESHOLD = 1e-2

def _pretty_model_name(model_name: str) -> str:
    names = {
        "ml_dmd": "NN-EDMD",
        "ml_linear_dynamics": "NN-LINOP",
        "mlp_baseline": "MLP",
        "linear_baseline": "LSTSQ",
        "dmd_baseline": "DMD",
        "regression_dmd": "EDMD",
        "sindy_baseline": "SINDy",
    }
    return names.get(model_name, model_name.replace("_", " ").title())

def format_latex_scientific(val, is_solved=False, is_best=False):
    """Formats values, applies bolding and background color to the best, and colors text."""
    if pd.isna(val):
        return "-"
    val_str = f"{val:.2e}"
    base, exp = val_str.split('e')
    exp = int(exp) # removes leading zeros like -04 -> -4
    
    # Base mathematical string
    inner_str = f"{base} \\times 10^{{{exp}}}"
    
    # Apply bolding if it is the best in the row
    if is_best:
        inner_str = f"\\mathbf{{{inner_str}}}"
        
    # Determine the color theme for the cell based on the solved threshold
    color_theme = "goodgreen" if is_solved else "badred"
    
    # Wrap the math string in the text color
    math_str = f"$\\textcolor{{{color_theme}}}{{{inner_str}}}$"
    
    # If it is the best model, inject the background color command before the math
    if is_best:
        return f"\\cellcolor{{{color_theme}!15}} {math_str}"
        
    return math_str

def format_rmse_cell(row):
    """Return ONLY the formatted RMSE."""
    return format_latex_scientific(row['target_rmse'], row['is_solved'], row['is_best'])

def format_param_cell(row):
    """Expanded, multi-line formatting for hyperparameters."""
    if pd.isna(row['target_rmse']):
        return "-"
        
    # Note: 'model' is already passed through _pretty_model_name before this runs,
    # so the values here are 'nn-edmd', 'nn-linop', 'mlp', 'sindy', 'edmd', etc.
    model = str(row.get('model_name', '')).lower()
    exp_type = str(row.get('expansion_type', 'none')).lower()

    lines = [] 

    # Helper: Print 0 for 0, "10" for 10.0, and "1e-06" for small values
    def fmt_val(v):
        f = float(v)
        if f == 0.0: return "0"
        if f >= 1 and f.is_integer(): return str(int(f))
        return f"{f:.0e}"

# 2. SINDy specific parameters
    if model == 'sindy':
        raw_lib = str(row.get('sindy_library_type', 'polynomial'))
        if raw_lib.lower() in ["nan", "none"]: raw_lib = "polynomial"
        lib = raw_lib.replace('_', ' ').title()
        
        if 'Poly' in lib and 'Fourier' not in lib:
            deg = row.get('sindy_poly_order')
            lines.append(f"Polynomial Degree: {int(deg)}" if pd.notna(deg) else "Polynomial Basis")
        elif 'Poly Fourier' in lib:
            lines.append("Poly-Fourier Basis")
            deg = row.get('sindy_poly_order')
            if pd.notna(deg): lines.append(f"Polynomial Degree: {int(deg)}")
            freq = row.get('sindy_fourier_n_frequencies')
            if pd.notna(freq): lines.append(f"Fourier Freqs: {int(freq)}")
        elif 'Specific' in lib:
            lines.append("Specific Basis")
            size = row.get('sindy_specific_basis_size')
            if pd.notna(size) and size > 0: lines.append(f"Basis Size: {int(size)}")
            
        # ALWAYS show SINDy threshold and alpha
        thresh = row.get('sindy_threshold')
        lines.append(f"Threshold: {fmt_val(thresh) if pd.notna(thresh) else '0'}")
        
        alpha = row.get('sindy_alpha')
        lines.append(f"Alpha: {fmt_val(alpha) if pd.notna(alpha) else '0'}")

    # 3. MLP specific parameters
    elif model == 'mlp':
        h = row.get('hidden_dim')
        l = row.get('num_layers')
        lines.append("Dense Network")
        if pd.notna(l): lines.append(f"Layers: {int(l)}")
        if pd.notna(h): lines.append(f"Hidden Dim: {int(h)}")

    # 4. LSTSQ / DMD (Pure linear baselines)
    elif model in ['lstsq', 'dmd']:
        r = row.get('rank')
        lines.append(f"Rank: {int(r)}" if pd.notna(r) and r != -1 else "Full Rank")
        
        ridge = row.get('ridge')
        if pd.notna(ridge) and float(ridge) > 0.0: lines.append(f"Ridge Penalty: {fmt_val(ridge)}")

    # 5. EDMD and Neural Koopman Models
    else:
        # Basis Expansion
        if exp_type not in ['nan', 'none', 'default']:
            if 'hankel' in exp_type:
                lines.append("Hankel Expansion")
                d = row.get('delay_depth')
                if pd.notna(d): lines.append(f"Delay Depth: {int(d)}")
                r = row.get('hankel_rank')
                if pd.notna(r): lines.append(f"Hankel Rank: {int(r)}")
            elif 'rbf' in exp_type:
                lines.append("RBF Expansion")
                c = row.get('rbf_n_centers')
                if pd.notna(c): lines.append(f"Centers: {int(c)}")
            elif 'specific' in exp_type:
                lines.append("Specific Basis")
            else:
                deg = row.get('expansion_degree')
                # CHANGED: Now uses "General Expansion" instead of "Polynomial Expansion"
                lines.append("General Expansion")
                if pd.notna(deg): lines.append(f"Degree: {int(deg)}")
        
        # Penalties
        if model == 'edmd':
            ridge = row.get('ridge')
            if pd.notna(ridge) and float(ridge) > 0.0: lines.append(f"Ridge Penalty: {fmt_val(ridge)}")
        
        elif 'nn-edmd' in model:
            # ALWAYS show L1 and Biorthogonal weights for NN-EDMD
            l1 = row.get('l1_weight')
            lines.append(f"L1 Penalty: {fmt_val(l1) if pd.notna(l1) else '0'}")
            biorth = row.get('biorth_weight')
            lines.append(f"Biorthogonal Pen: {fmt_val(biorth) if pd.notna(biorth) else '0'}")

    # 6. ALL Neural Networks: Always show Weight Decay
    if model in ['mlp', 'nn-linop'] or 'nn-edmd' in model:
        wd = row.get('weight_decay')
        lines.append(f"Weight Decay: {fmt_val(wd) if pd.notna(wd) else '0'}")

    if not lines:
        lines.append("Default Parameters")

    # 1. ALWAYS show dt at the very bottom
    dt_val = row.get('dt')
    if pd.notna(dt_val):
        lines.append(f"dt: {dt_val}")

    # Stack the lines vertically inside the cell, left-aligned
    joined_lines = " \\\\[0.5ex] ".join(lines)
    return f"\\makecell[l]{{{joined_lines}}}"


def build_latex_table(pivot_df, caption, label, is_param_table=False):
    """Helper function to generate the raw LaTeX code for a given pivot dataframe."""
    num_cols = len(pivot_df.columns) + 1
    lines = []
    
    lines.append("\\begin{table}[h!]")
    lines.append("\\centering")
    lines.append("\\renewcommand{\\arraystretch}{1.4}") 
    
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\resizebox{\\textwidth}{!}{%") 
    lines.append(f"\\begin{{tabular}}{{l{'c' * (num_cols - 1)}}}")
    lines.append("\\toprule")
    
    header_str = " & " + " & ".join(pivot_df.columns) + " \\\\"
    lines.append(header_str)
    lines.append("\\midrule")
    
    categories = [
        ["Degenerate Node", "Harmonic Oscillator", "Inward Spiral", "Inward Spiral Cw", "Saddle Point"],
        ["Closed Small", "Closed Large", "Closed Trig Small", "Closed Trig Medium", "Closed Trig Large"],
        ["Duffing", "Lotka Volterra", "Pendulum", "Vanderpol", "Lorenz"]
    ]
    
    added_systems = set()
    first_group = True
    
    # NEW: Define a light, semi-transparent line for row separation
    light_line = "\\arrayrulecolor{black!20}\\midrule\\arrayrulecolor{black}"
    
    for sys_names in categories:
        present_sys = [s for s in sys_names if s in pivot_df.index]
        if not present_sys:
            continue
            
        if not first_group:
            lines.append("\\midrule") # Solid line between categories
        first_group = False
        
        for i, sys in enumerate(present_sys):
            row_vals = pivot_df.loc[sys].values
            row_str = f"{sys} & " + " & ".join(row_vals) + " \\\\"
            lines.append(row_str)
            added_systems.add(sys)
            
            # Add the light line after every row EXCEPT the last one in the block
            if i < len(present_sys) - 1:
                lines.append(light_line)
            
    remaining = [s for s in pivot_df.index if s not in added_systems]
    if remaining:
        lines.append("\\midrule")
        for i, sys in enumerate(remaining):
            row_vals = pivot_df.loc[sys].values
            row_str = f"{sys} & " + " & ".join(row_vals) + " \\\\"
            lines.append(row_str)
            if i < len(remaining) - 1:
                lines.append(light_line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)

if __name__ == "__main__":
    search_pattern = "experiments/wandb/*/test_results_dt_*.csv"
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"No files found matching: {search_pattern}")
        exit()

    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    
    def get_1s_rmse(row):
        if row.get('dt') == 0.05:
            return row.get('h20_rmse', np.nan)
        return row.get('h100_rmse', np.nan)

    df['target_rmse'] = pd.to_numeric(df.apply(get_1s_rmse, axis=1), errors='coerce')

    best_runs = df.sort_values('target_rmse').drop_duplicates(
        subset=['system_name', 'model_name'], 
        keep='first'
    ).copy()
    
    best_runs['model_name'] = best_runs['model_name'].apply(_pretty_model_name)
    best_runs['system_name'] = best_runs['system_name'].str.replace("_", " ").str.title()

    # --- NEW: Compute row minimums to find the best model per system ---
    # Use idxmin() to strictly pick the FIRST occurrence of the minimum, breaking exact ties
    best_idx = best_runs.groupby('system_name')['target_rmse'].idxmin()
    best_runs['is_best'] = best_runs.index.isin(best_idx)
    
    best_runs['is_solved'] = best_runs['target_rmse'] <= SOLVED_THRESHOLD

    # Generate cells
    best_runs['rmse_cell'] = best_runs.apply(format_rmse_cell, axis=1)
    best_runs['param_cell'] = best_runs.apply(format_param_cell, axis=1)
    
    # ---------------------------------------------------------
    # Render Tables
    # ---------------------------------------------------------
    
    desired_order_all = ['LSTSQ', 'DMD', 'EDMD', 'SINDy', 'MLP', 'NN-LINOP', 'NN-EDMD', 'NN-EDMD (Drop)']
    
    # 1. Generate the single, full-page RMSE Table
    pivot_rmse = best_runs.pivot(index='system_name', columns='model_name', values='rmse_cell').fillna("-")
    actual_order_rmse = [col for col in desired_order_all if col in pivot_rmse.columns] + [c for c in pivot_rmse.columns if c not in desired_order_all]
    pivot_rmse = pivot_rmse[actual_order_rmse]
    
    latex_rmse = build_latex_table(pivot_rmse, "1.0s Rollout RMSE Performance", "tab:rmse_results")

    # 2. Prepare the Parameters Pivot Table
    pivot_param = best_runs.pivot(index='system_name', columns='model_name', values='param_cell').fillna("-")
    actual_order_param = [col for col in desired_order_all if col in pivot_param.columns] + [c for c in pivot_param.columns if c not in desired_order_all]
    pivot_param = pivot_param[actual_order_param]
    
    # Define our 3 system categories explicitly
    sys_linear = ["Degenerate Node", "Harmonic Oscillator", "Inward Spiral", "Inward Spiral Cw", "Saddle Point"]
    sys_closed = ["Closed Small", "Closed Large", "Closed Trig Small", "Closed Trig Medium", "Closed Trig Large"]
    sys_open = ["Duffing", "Lotka Volterra", "Pendulum", "Vanderpol", "Lorenz"]
    
    # Catch any unexpected datasets so they don't get accidentally dropped!
    known_sys = set(sys_linear + sys_closed + sys_open)
    sys_other = [s for s in pivot_param.index if s not in known_sys]
    
    # 3. Slice the dataframe into 3 smaller chunks
    param_linear = pivot_param[pivot_param.index.isin(sys_linear)]
    param_closed = pivot_param[pivot_param.index.isin(sys_closed)]
    param_open = pivot_param[pivot_param.index.isin(sys_open + sys_other)]
    
    # 4. Render the 3 separate tables (they will automatically drop empty sections)
    latex_param_linear = build_latex_table(param_linear, "Optimal Hyperparameters: Linear Systems", "tab:hyperparams_linear", is_param_table=True)
    latex_param_closed = build_latex_table(param_closed, "Optimal Hyperparameters: Closed (Limit Cycle) Systems", "tab:hyperparams_closed", is_param_table=True)
    latex_param_open = build_latex_table(param_open, "Optimal Hyperparameters: Open & Chaotic Systems", "tab:hyperparams_open", is_param_table=True)

    # ---------------------------------------------------------
    # Print Outputs
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(" TABLE 1: RMSE RESULTS")
    print("="*50 + "\n")
    print(latex_rmse)

    print("\n" + "="*50)
    print(" TABLE 2A: HYPERPARAMETERS (LINEAR)")
    print("="*50 + "\n")
    print(latex_param_linear)
    
    print("\n" + "="*50)
    print(" TABLE 2B: HYPERPARAMETERS (CLOSED / LIMIT CYCLE)")
    print("="*50 + "\n")
    print(latex_param_closed)

    print("\n" + "="*50)
    print(" TABLE 2C: HYPERPARAMETERS (OPEN / CHAOTIC)")
    print("="*50 + "\n")
    print(latex_param_open)