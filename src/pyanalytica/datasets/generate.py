"""Generate bundled datasets for PyAnalytica.

Run: python -m pyanalytica.datasets.generate
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

_DIR = Path(__file__).parent
_RNG = np.random.default_rng(42)


def generate_diamonds() -> None:
    """Generate a diamonds dataset similar to the ggplot2 classic."""
    n = 53940
    rng = _RNG

    cuts = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    colors = ["D", "E", "F", "G", "H", "I", "J"]
    clarities = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

    carat = rng.lognormal(mean=-0.3, sigma=0.5, size=n).clip(0.2, 5.01).round(2)
    cut = rng.choice(cuts, size=n, p=[0.03, 0.09, 0.22, 0.26, 0.40])
    color = rng.choice(colors, size=n, p=[0.10, 0.14, 0.16, 0.17, 0.16, 0.14, 0.13])
    clarity = rng.choice(clarities, size=n, p=[0.03, 0.15, 0.20, 0.22, 0.16, 0.12, 0.08, 0.04])

    depth = rng.normal(61.7, 1.4, size=n).clip(43, 79).round(1)
    table = rng.normal(57.5, 2.2, size=n).clip(43, 95).round(0)

    x = (carat ** (1/3) * 4.0 + rng.normal(0, 0.15, size=n)).clip(0, 11).round(2)
    y = (x + rng.normal(0, 0.1, size=n)).clip(0, 59).round(2)
    z = (x * depth / 100 + rng.normal(0, 0.05, size=n)).clip(0, 32).round(2)

    # Price model
    cut_mult = np.array([cuts.index(c) for c in cut]) * 0.05 + 0.8
    color_mult = np.array([colors.index(c) for c in color]) * -0.03 + 1.1
    clarity_mult = np.array([clarities.index(c) for c in clarity]) * 0.04 + 0.85
    base_price = carat ** 1.5 * 3000
    price = (base_price * cut_mult * color_mult * clarity_mult
             + rng.normal(0, 200, size=n)).clip(326, 18823).astype(int)

    df = pd.DataFrame({
        "carat": carat, "cut": cut, "color": color, "clarity": clarity,
        "depth": depth, "table": table, "price": price,
        "x": x, "y": y, "z": z,
    })

    out = _DIR / "diamonds"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "diamonds.csv", index=False)
    print(f"  diamonds: {len(df)} rows -> {out / 'diamonds.csv'}")


def generate_tips() -> None:
    """Generate a tips dataset similar to the seaborn classic."""
    n = 244
    rng = _RNG

    sex = rng.choice(["Female", "Male"], size=n, p=[0.36, 0.64])
    smoker = rng.choice(["No", "Yes"], size=n, p=[0.62, 0.38])
    day = rng.choice(["Thur", "Fri", "Sat", "Sun"], size=n, p=[0.25, 0.08, 0.36, 0.31])
    time = np.where(day == "Thur", rng.choice(["Lunch", "Dinner"], size=n, p=[0.6, 0.4]), "Dinner")
    size = rng.choice([1, 2, 3, 4, 5, 6], size=n, p=[0.03, 0.63, 0.14, 0.15, 0.03, 0.02])

    total_bill = (rng.lognormal(2.8, 0.5, size=n) + size * 3).clip(3.07, 50.81).round(2)
    tip_pct = rng.normal(0.16, 0.06, size=n).clip(0.01, 0.7)
    tip = (total_bill * tip_pct).clip(1.0, 10.0).round(2)

    df = pd.DataFrame({
        "total_bill": total_bill, "tip": tip, "sex": sex,
        "smoker": smoker, "day": day, "time": time, "size": size,
    })

    out = _DIR / "tips"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "tips.csv", index=False)
    print(f"  tips: {len(df)} rows -> {out / 'tips.csv'}")


def generate_jobmatch() -> None:
    """Generate the JobMatch recruiting simulation (4 tables)."""
    rng = _RNG

    # --- Companies (200) ---
    n_companies = 200
    industries = ["Technology", "Healthcare", "Finance", "Retail", "Manufacturing",
                  "Consulting", "Education", "Energy", "Media", "Real Estate"]
    sizes = ["Small", "Medium", "Large", "Enterprise"]

    companies = pd.DataFrame({
        "company_id": range(1, n_companies + 1),
        "company_name": [f"Company_{i:03d}" for i in range(1, n_companies + 1)],
        "industry": rng.choice(industries, size=n_companies),
        "company_size": rng.choice(sizes, size=n_companies, p=[0.30, 0.35, 0.25, 0.10]),
        "founded_year": rng.integers(1950, 2024, size=n_companies),
        "headquarters": rng.choice(
            ["New York", "San Francisco", "Chicago", "Austin", "Boston",
             "Seattle", "Denver", "Atlanta", "Miami", "Los Angeles"],
            size=n_companies
        ),
    })

    # --- Candidates (5000) ---
    n_candidates = 5000
    seniority_levels = ["Entry", "Junior", "Mid", "Senior", "Executive"]
    degrees = ["High School", "Associate", "Bachelor", "Master", "PhD"]

    age = rng.normal(32, 8, size=n_candidates).clip(22, 65).astype(int)
    seniority = rng.choice(seniority_levels, size=n_candidates, p=[0.20, 0.25, 0.30, 0.18, 0.07])
    degree = rng.choice(degrees, size=n_candidates, p=[0.05, 0.08, 0.45, 0.32, 0.10])

    experience_base = {"Entry": 1, "Junior": 3, "Mid": 6, "Senior": 10, "Executive": 15}
    experience = np.array([
        max(0, rng.normal(experience_base[s], 2)) for s in seniority
    ]).clip(0, 40).round(1)

    salary_base = {"Entry": 52000, "Junior": 65000, "Mid": 82000, "Senior": 105000, "Executive": 145000}
    salary = np.array([
        rng.normal(salary_base[s], salary_base[s] * 0.15) for s in seniority
    ]).clip(30000, 300000).round(0).astype(int)

    gender = rng.choice(["Male", "Female", "Non-binary"], size=n_candidates, p=[0.48, 0.48, 0.04])

    # Add some missing values (~3% in salary, ~5% in experience)
    salary_float = salary.astype(float)
    salary_float[rng.choice(n_candidates, size=int(n_candidates * 0.03), replace=False)] = np.nan
    experience_arr = experience.copy()
    experience_arr[rng.choice(n_candidates, size=int(n_candidates * 0.05), replace=False)] = np.nan

    candidates = pd.DataFrame({
        "candidate_id": range(1, n_candidates + 1),
        "age": age,
        "gender": gender,
        "degree": degree,
        "seniority": seniority,
        "experience_years": experience_arr,
        "current_salary": salary_float,
        "city": rng.choice(
            ["New York", "San Francisco", "Chicago", "Austin", "Boston",
             "Seattle", "Denver", "Atlanta", "Remote", "Los Angeles"],
            size=n_candidates
        ),
        "skills_count": rng.poisson(5, size=n_candidates).clip(1, 20),
    })

    # --- Jobs (500) ---
    n_jobs = 500
    job_seniority = rng.choice(seniority_levels, size=n_jobs, p=[0.20, 0.25, 0.30, 0.18, 0.07])
    job_company = rng.integers(1, n_companies + 1, size=n_jobs)

    min_salary = np.array([salary_base[s] * 0.8 for s in job_seniority]).astype(int)
    max_salary = np.array([salary_base[s] * 1.3 for s in job_seniority]).astype(int)

    jobs = pd.DataFrame({
        "job_id": range(1, n_jobs + 1),
        "company_id": job_company,
        "title": [f"{s} Analyst" if rng.random() > 0.5 else f"{s} Engineer"
                  for s in job_seniority],
        "seniority": job_seniority,
        "min_salary": min_salary,
        "max_salary": max_salary,
        "remote_ok": rng.choice([True, False], size=n_jobs, p=[0.40, 0.60]),
        "posted_date": pd.date_range("2025-01-01", periods=n_jobs, freq="4h").strftime("%Y-%m-%d"),
    })

    # --- Events (~50K for manageability) ---
    event_types = ["application", "screen", "interview", "offer", "hire", "reject"]
    event_probs = [0.40, 0.20, 0.15, 0.10, 0.08, 0.07]

    n_events = 50000
    events = pd.DataFrame({
        "event_id": range(1, n_events + 1),
        "candidate_id": rng.integers(1, n_candidates + 1, size=n_events),
        "job_id": rng.integers(1, n_jobs + 1, size=n_events),
        "event_type": rng.choice(event_types, size=n_events, p=event_probs),
        "event_date": pd.date_range("2025-01-01", periods=n_events, freq="10min").strftime("%Y-%m-%d"),
        "rating": rng.choice([np.nan, 1, 2, 3, 4, 5], size=n_events, p=[0.40, 0.05, 0.10, 0.20, 0.15, 0.10]),
    })

    # Save all
    out = _DIR / "jobmatch"
    out.mkdir(exist_ok=True)
    companies.to_csv(out / "companies.csv", index=False)
    candidates.to_csv(out / "candidates.csv", index=False)
    jobs.to_csv(out / "jobs.csv", index=False)
    events.to_csv(out / "events.csv", index=False)
    print(f"  companies: {len(companies)} rows")
    print(f"  candidates: {len(candidates)} rows")
    print(f"  jobs: {len(jobs)} rows")
    print(f"  events: {len(events)} rows")


def main():
    print("Generating PyAnalytica bundled datasets...")
    generate_diamonds()
    generate_tips()
    generate_jobmatch()
    print("Done!")


if __name__ == "__main__":
    main()
