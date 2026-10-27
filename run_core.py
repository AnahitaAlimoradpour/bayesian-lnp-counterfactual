import sys
import subprocess

# بررسی و نصب خودکار پکیج‌ها در صورت نیاز
required_packages = {'numpy': 'numpy', 'matplotlib': 'matplotlib', 'arviz': 'arviz', 'pymc': 'pymc'}
for pkg, import_name in required_packages.items():
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing missing package: {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import numpy as np
import pymc as pm
import arviz as az

# محافظت پایتون در ویندوز برای Multiprocessing
if __name__ == '__main__':
    print(">>> ALL PACKAGES LOADED SUCCESSFULLY (0 Errors)! <<<")

    # تست اجرایی مدل بیزین برای نانوذرات
    np.random.seed(42)
    X_dummy = np.random.randn(50, 2)
    y_dummy = 2.0 * X_dummy[:, 0] - 1.5 * X_dummy[:, 1] + np.random.normal(0, 0.5, 50)

    with pm.Model() as nanoparticle_model:
        beta = pm.Normal("beta", mu=0, sigma=2, shape=2)
        sigma = pm.HalfNormal("sigma", sigma=1)
        mu = pm.math.dot(X_dummy, beta)
        likelihood = pm.Normal("obs", mu=mu, sigma=sigma, observed=y_dummy)
        
        # استفاده از chains=1 برای جلوگیری از درگیری موازی در تست اولیه ویندوز
        idata = pm.sample(draws=100, tune=100, chains=1, return_inferencedata=True, progressbar=False)

    print(">>> MCMC Sampling Test Passed Successfully! <<<")
    print(az.summary(idata))
