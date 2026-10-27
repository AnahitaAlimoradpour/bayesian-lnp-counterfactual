import numpy as np
import pymc as pm
import arviz as az

if __name__ == '__main__':
    print(">>> 1. Re-running Bayesian LNP Fitting for Counterfactual Engine...")
    np.random.seed(42)
    n_samples = 120
    
    ionizable = np.random.uniform(35, 55, n_samples)
    helper = np.random.uniform(5, 20, n_samples)
    cholesterol = np.random.uniform(30, 45, n_samples)
    np_ratio = np.random.uniform(3, 10, n_samples)
    
    total_lipid = ionizable + helper + cholesterol
    X_data = np.column_stack([ionizable / total_lipid, helper / total_lipid, cholesterol / total_lipid, np_ratio / 10.0])
    true_weights = np.array([3.2, -1.1, 1.8, 2.5])
    y_data = np.dot(X_data, true_weights) + np.random.normal(0, 0.4, n_samples)

    with pm.Model() as lnp_model:
        beta = pm.Normal("beta", mu=0, sigma=2, shape=4)
        alpha = pm.Normal("alpha", mu=0, sigma=5)
        sigma = pm.HalfNormal("sigma", sigma=1)
        mu = alpha + pm.math.dot(X_data, beta)
        likelihood = pm.Normal("obs", mu=mu, sigma=sigma, observed=y_data)
        idata = pm.sample(draws=300, tune=300, chains=1, return_inferencedata=True, progressbar=False)

    print("\n>>> 2. Performing Counterfactual / What-If Optimization...")
    # تعریف دو سناریو فرضی (فرمولاسیون ضدواقعی برای تست رفتار مدل)
    # سناریوی الف: افزایش کاتیونیک لیپید و نسبت N/P
    # سناریوی ب: حالت استاندارد فعلی
    counterfactual_scenarios = np.array([
        [0.55, 0.10, 0.35, 0.8],  # سناریوی A (پیشنهادی جدید)
        [0.45, 0.15, 0.40, 0.5]   # سناریوی B (پایه)
    ])

    # استخراج نمونه‌های پسین (Posterior Samples) برای محاسبه عدم‌قطعیت پیش‌بینی
    posterior = idata.posterior
    beta_samples = posterior['beta'].values.reshape(-1, 4)
    alpha_samples = posterior['alpha'].values.reshape(-1)
    sigma_samples = posterior['sigma'].values.reshape(-1)

    print("\n>>> 3. Counterfactual Predictions & Uncertainty Quantification:")
    for i, sc in enumerate(counterfactual_scenarios):
        # محاسبه پیش‌بینی برای تمام نمونه‌های پسین
        pred_samples = alpha_samples + np.dot(beta_samples, sc)
        pred_mean = np.mean(pred_samples)
        pred_uncertainty = np.std(pred_samples) # Epistemic Uncertainty
        
        print(f"Scenario {chr(65+i)} [Ionizable={sc[0]:.2f}, Helper={sc[1]:.2f}, Chol={sc[2]:.2f}, N/P={sc[3]*10:.1f}]:")
        print(f"  -> Predicted Efficacy (Mean): {pred_mean:.3f}")
        print(f"  -> Epistemic Uncertainty (SD): {pred_uncertainty:.3f}\n")

    print(">>> SUCCESS: Counterfactual Uncertainty Pipeline Operational! <<<")
