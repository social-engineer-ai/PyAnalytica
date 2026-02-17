"""Generate bundled datasets for PyAnalytica.

Run: python -m pyanalytica.datasets.generate
"""

from __future__ import annotations

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


def generate_titanic() -> None:
    """Generate a Titanic dataset similar to the classic Kaggle version."""
    n = 891
    rng = np.random.default_rng(1912)

    # Passenger class distribution (approx historical)
    pclass = rng.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55])

    # Sex distribution
    sex = rng.choice(["male", "female"], size=n, p=[0.65, 0.35])

    # Age: varies by class
    age = np.zeros(n)
    for i in range(n):
        if pclass[i] == 1:
            age[i] = rng.normal(38, 14)
        elif pclass[i] == 2:
            age[i] = rng.normal(30, 13)
        else:
            age[i] = rng.normal(25, 12)
    age = age.clip(0.42, 80).round(1)
    # ~20% missing ages
    age_missing = rng.choice(n, size=int(n * 0.20), replace=False)
    age_float = age.copy()

    # SibSp and Parch
    sibsp = rng.choice([0, 1, 2, 3, 4, 5, 8], size=n,
                       p=[0.68, 0.23, 0.03, 0.02, 0.02, 0.01, 0.01])
    parch = rng.choice([0, 1, 2, 3, 4, 5, 6], size=n,
                       p=[0.76, 0.12, 0.08, 0.01, 0.01, 0.01, 0.01])

    # Fare: correlated with class
    fare = np.zeros(n)
    for i in range(n):
        if pclass[i] == 1:
            fare[i] = rng.lognormal(3.8, 0.7)
        elif pclass[i] == 2:
            fare[i] = rng.lognormal(2.7, 0.5)
        else:
            fare[i] = rng.lognormal(2.0, 0.6)
    fare = fare.clip(0, 512.33).round(4)

    # Embarked
    embarked = rng.choice(["S", "C", "Q"], size=n, p=[0.72, 0.19, 0.09])

    # Survival: depends on sex, class, age
    survived = np.zeros(n, dtype=int)
    for i in range(n):
        base_prob = 0.38  # overall survival rate
        if sex[i] == "female":
            base_prob += 0.35
        if pclass[i] == 1:
            base_prob += 0.15
        elif pclass[i] == 3:
            base_prob -= 0.12
        if age_float[i] < 10:
            base_prob += 0.15
        elif age_float[i] > 60:
            base_prob -= 0.10
        base_prob = np.clip(base_prob, 0.05, 0.95)
        survived[i] = 1 if rng.random() < base_prob else 0

    # Names (synthetic but realistic format)
    first_names_m = [
        "James", "John", "William", "Robert", "Charles", "George", "Edward",
        "Thomas", "Henry", "Arthur", "Joseph", "Albert", "Harry", "Walter",
        "Frederick", "Frank", "Samuel", "Patrick", "Daniel", "Michael",
        "Oscar", "Ernest", "Herbert", "Alfred", "Bernard", "Peter", "Louis",
        "Alexander", "Victor", "Eugene", "Leo", "Karl", "Ivan", "Erik",
        "Hans", "Olaf", "Lars", "Nils", "Johan", "Gustav",
    ]
    first_names_f = [
        "Mary", "Anna", "Elizabeth", "Margaret", "Ellen", "Catherine", "Alice",
        "Edith", "Florence", "Helen", "Sarah", "Jane", "Julia", "Bertha",
        "Emma", "Clara", "Mabel", "Lillian", "Eva", "Rosa",
        "Agnes", "Elsa", "Ingrid", "Maria", "Helga", "Hilda",
        "Amelia", "Dorothy", "Ethel", "Gladys",
    ]
    last_names = [
        "Smith", "Johnson", "Brown", "Williams", "Jones", "Davis", "Wilson",
        "Anderson", "Taylor", "Thomas", "Moore", "Martin", "Thompson",
        "White", "Harris", "Clark", "Lewis", "Hall", "Allen", "Young",
        "King", "Wright", "Hill", "Green", "Baker", "Nelson", "Carter",
        "Mitchell", "Roberts", "Turner", "Phillips", "Campbell", "Parker",
        "Evans", "Edwards", "Collins", "Stewart", "Morris", "Murphy",
        "Cook", "Rogers", "Morgan", "Cooper", "Reed", "Bailey",
        "Kelly", "Howard", "Ward", "Cox", "Peterson", "Gray",
        "O'Brien", "Sullivan", "McCarthy", "O'Connor", "Fitzgerald",
        "Andersson", "Johansson", "Eriksson", "Lindqvist", "Olsen",
        "Hansen", "Pedersen", "Larsen", "Jensen", "Nielsen",
        "Mueller", "Schmidt", "Fischer", "Weber", "Meyer",
        "Rossi", "Bianchi", "Marino", "Ricci", "Romano",
    ]
    titles_m = ["Mr.", "Master", "Rev.", "Dr.", "Col.", "Major", "Capt."]
    titles_f = ["Miss", "Mrs.", "Ms.", "Lady", "Countess"]
    title_prob_m = [0.82, 0.06, 0.04, 0.04, 0.02, 0.01, 0.01]
    title_prob_f = [0.45, 0.45, 0.05, 0.03, 0.02]

    names = []
    for i in range(n):
        last = rng.choice(last_names)
        if sex[i] == "male":
            first = rng.choice(first_names_m)
            title = rng.choice(titles_m, p=title_prob_m)
            if age_float[i] < 14:
                title = "Master"
        else:
            first = rng.choice(first_names_f)
            title = rng.choice(titles_f, p=title_prob_f)
        names.append(f"{last}, {title} {first}")

    # Ticket numbers
    ticket_prefixes = ["", "A/5", "PC", "STON/O", "S.O.C.", "C.A.", "W./C.", "SOTON/OQ", "F.C.C."]
    tickets = []
    for _ in range(n):
        prefix = rng.choice(ticket_prefixes, p=[0.60, 0.06, 0.08, 0.05, 0.04, 0.06, 0.04, 0.04, 0.03])
        num = rng.integers(1000, 999999)
        if prefix:
            tickets.append(f"{prefix} {num}")
        else:
            tickets.append(str(num))

    # Cabin (mostly missing, more present for 1st class)
    cabin_letters = ["A", "B", "C", "D", "E", "F", "G", "T"]
    cabin = []
    for i in range(n):
        if pclass[i] == 1:
            has_cabin = rng.random() < 0.60
        elif pclass[i] == 2:
            has_cabin = rng.random() < 0.15
        else:
            has_cabin = rng.random() < 0.05
        if has_cabin:
            letter = rng.choice(cabin_letters[:6] if pclass[i] <= 2 else cabin_letters)
            room = rng.integers(1, 150)
            cabin.append(f"{letter}{room}")
        else:
            cabin.append(np.nan)

    # Build DataFrame
    df = pd.DataFrame({
        "PassengerId": range(1, n + 1),
        "Survived": survived,
        "Pclass": pclass,
        "Name": names,
        "Sex": sex,
        "Age": age_float,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": tickets,
        "Fare": fare,
        "Cabin": cabin,
        "Embarked": embarked,
    })

    # Set missing ages to NaN
    df.loc[age_missing, "Age"] = np.nan

    out = _DIR / "titanic"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "titanic.csv", index=False)
    print(f"  titanic: {len(df)} rows -> {out / 'titanic.csv'}")


def main():
    print("Generating PyAnalytica bundled datasets...")
    generate_diamonds()
    generate_tips()
    generate_titanic()
    print("Done!")


if __name__ == "__main__":
    main()
