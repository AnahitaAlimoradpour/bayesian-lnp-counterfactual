import os
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

def main():
    os.makedirs("assets", exist_ok=True)
    
    # 1. Reproducible benchmark dummy generation
    np.random.seed(42)
    n_samples = 120
    # Features: [Ionizable_Lipid_ratio, Helper_Lipid, Cholesterol, PEG_Lipid]
    X = np.random.uniform(0.1, 0.9, size=(n_samples, 4))
    true_beta = np.array([2.5, -1.2, 0.8, -0.5])
    y = np.dot(X, true_beta) + np.random.normal(0, 0.25, size=n_samples)
    
    # 2. Bayesian Model
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta = pm.Normal("beta", mu=0, sigma=1, shape=4)
        sigma = pm.HalfNormal("sigma", sigma=1)
        
        mu = alpha + pm.math.dot(X, beta)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
        
        trace = pm.sample(
            draws=400,
            tune=400,
            chains=1,
            random_seed=42,
            return_inferencedata=True,
            progressbar=False
        )

    # 3. Extract Posterior Draws
    beta_samples = trace.posterior["beta"].values[0] # shape: (400, 4)
    feature_names = ["Ionizable Lipid", "Helper Lipid", "Cholesterol", "PEG-Lipid"]
    
    # 4. Clean Publication-Quality Visualization
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Subplot 1: Posterior Distributions
    for i in range(4):
        axes[0].hist(beta_samples[:, i], bins=25, alpha=0.6, label=feature_names[i], density=True)
    axes[0].set_title("Posterior Distributions for LNP Features", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Effect Size (Beta Weight)", fontsize=10)
    axes[0].set_ylabel("Density", fontsize=10)
    axes[0].legend(frameon=True)

    # Subplot 2: 89% Credible Intervals (HDI)
    means = np.mean(beta_samples, axis=0)
    hdi_low = np.percentile(beta_samples, 5.5, axis=0)
    hdi_high = np.percentile(beta_samples, 94.5, axis=0)
    y_pos = np.arange(len(feature_names))

    axes[1].errorbar(means, y_pos, xerr=[means - hdi_low, hdi_high - means], fmt='o', color='#1f77b4',
                     ecolor='#ff7f0e', elinewidth=2.5, capsize=5, markersize=7)
    axes[1].axvline(0, color='gray', linestyle='--', alpha=0.7)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(feature_names, fontsize=10, fontweight="bold")
    axes[1].set_title("89% Highest Density Intervals (HDI)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Estimated Parameter Value", fontsize=10)

    plt.tight_layout()
    output_path = os.path.join("assets", "posterior_plot.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f">>> Success! Plot generated and saved at: {output_path}")

if __name__ == "__main__":
    main()
