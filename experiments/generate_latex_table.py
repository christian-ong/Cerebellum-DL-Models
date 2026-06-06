import os
import glob
import pandas as pd
import numpy as np

def _pretty_model_name(model_name: str) -> str:
    names = {
        "ml_dmd": "NN-EDMD",
        "ml_dmd_drop": "NN-EDMD (Drop)",
        "ml_linear_dynamics": "NN-LINOP",
        "mlp_baseline": "MLP",
        "linear_baseline": "LSTSQ",
        "dmd_baseline": "DMD",
        "regression_dmd": "EDMD",
        "sindy_baseline": "SINDy",
    }
    return names.get(model_name, model_name.replace("_", " ").title())

def format_latex_scientific(val):
    """Format 1.23e-04 into LaTeX $1.23 \times 10^{-4}$"""
    if pd.isna(val):
        return "-"
    val_str = f"{val:.2e}"
    base, exp = val_str.split('e')
    exp = int(exp) # removes leading zeros like -04 -> -4
    return f"${base} \\times 10^{{{exp}}}$"

def format_cell(row):
    """Combine the RMSE and expansion type into a single LaTeX cell string."""
    rmse_latex = format_latex_scientific(row['h100_rmse'])
    
    # Retrieve expansion type
    exp_type = str(row.get('expansion_type', 'None'))
    
    if exp_type.lower() in ['nan', 'none']:
        exp_type = "-" # Use a dash instead of writing "None" to keep it cleaner
    else:
        # Clean up expansion names (e.g., 'hankel_svd' -> 'Hankel SVD')
        exp_type = exp_type.replace('_', ' ').title()
        
    # Put the RMSE on top, and expansion type below it in smaller text
    return f"\\makecell{{{rmse_latex} \\\\ {{\\scriptsize ({exp_type})}}}}"

if __name__ == "__main__":
    # 1. Find all overview CSVs for dt=0.01
    search_pattern = "experiments/wandb/*/test_results_dt_0.01.csv"
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"No files found matching: {search_pattern}")
        exit()

    # 2. Load and concatenate all data
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    
    if 'h100_rmse' not in df.columns:
        print("Error: 'h100_rmse' column not found in the results.")
        exit()

    # 3. Find the best run for each (system_name, model_name) combo
    best_runs = df.sort_values('h100_rmse').drop_duplicates(
        subset=['system_name', 'model_name'], 
        keep='first'
    )
    
    # 4. Format the cells and model/system names
    best_runs['formatted_cell'] = best_runs.apply(format_cell, axis=1)
    best_runs['model_name'] = best_runs['model_name'].apply(_pretty_model_name)
    best_runs['system_name'] = best_runs['system_name'].str.replace("_", " ").str.title()
    
    # 5. Pivot the table
    pivot_df = best_runs.pivot(
        index='system_name', 
        columns='model_name', 
        values='formatted_cell'
    )
    
    pivot_df.fillna("-", inplace=True)
    
    # 6. Reorder the columns to the requested layout
    desired_order = ['LSTSQ', 'DMD', 'EDMD', 'SINDy', 'MLP', 'NN-LINOP', 'NN-EDMD']
    actual_order = [col for col in desired_order if col in pivot_df.columns]
    
    # Append any extra models that might exist
    for col in pivot_df.columns:
        if col not in actual_order:
            actual_order.append(col)
            
    pivot_df = pivot_df[actual_order]
    
    # 7. Define System Categories (just lists now, no header names)
    categories = [
        # Linear Systems
        ["Degenerate Node", "Harmonic Oscillator", "Inward Spiral", "Inward Spiral Cw", "Saddle Point"],
        # Closed Nonlinear Systems
        ["Closed Large", "Closed Small", "Closed Trig Large", "Closed Trig Medium", "Closed Trig Small"],
        # Open Nonlinear Systems
        ["Duffing", "Lorenz", "Lotka Volterra", "Pendulum", "Vanderpol"]
    ]

    # 8. Manually Generate LaTeX Code
    num_cols = len(pivot_df.columns) + 1
    lines = []
    
    lines.append("\\begin{table}[h!]")
    lines.append("\\centering")
    lines.append("\\resizebox{\\textwidth}{!}{%") 
    lines.append(f"\\begin{{tabular}}{{l{'c' * (num_cols - 1)}}}")
    lines.append("\\toprule")
    
    # Build header row
    header_str = " & " + " & ".join(pivot_df.columns) + " \\\\"
    lines.append(header_str)
    lines.append("\\midrule")
    
    added_systems = set()
    first_group = True
    
    for sys_names in categories:
        # Check which systems from this category actually exist in the dataframe
        present_sys = [s for s in sys_names if s in pivot_df.index]
        if not present_sys:
            continue
            
        # Add a midrule between groups (but not before the first group, as there's already one under the header)
        if not first_group:
            lines.append("\\midrule")
        first_group = False
        
        # Add rows
        for sys in present_sys:
            row_vals = pivot_df.loc[sys].values
            row_str = f"{sys} & " + " & ".join(row_vals) + " \\\\"
            lines.append(row_str)
            added_systems.add(sys)
            
    # Add any remaining systems that didn't fit into the predefined categories
    remaining = [s for s in pivot_df.index if s not in added_systems]
    if remaining:
        lines.append("\\midrule")
        for sys in remaining:
            row_vals = pivot_df.loc[sys].values
            row_str = f"{sys} & " + " & ".join(row_vals) + " \\\\"
            lines.append(row_str)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")
    
    latex_str = "\n".join(lines)
    
    print("\n--- LaTeX Table Output ---\n")
    print("% Remember to add these to your LaTeX preamble if you haven't already:")
    print("% \\usepackage{booktabs}")
    print("% \\usepackage{makecell}")
    print("% \\usepackage{graphicx} % for \\resizebox\n")
    print(latex_str)