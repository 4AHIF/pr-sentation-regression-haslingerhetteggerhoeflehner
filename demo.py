import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

# 1. Synthetische RAM-Daten erstellen
# Features: Kapazität (GB), Takt (MHz), DDR-Version, RGB (0/1)
data = {
    'gb': [8, 16, 32, 64, 8, 16, 32, 64, 16, 32],
    'mhz': [2400, 3200, 3600, 5200, 2666, 3600, 6000, 6400, 3200, 5600],
    'ddr': [4, 4, 4, 5, 4, 4, 5, 5, 4, 5],
    'rgb': [0, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    'preis': [35, 55, 95, 210, 38, 75, 140, 280, 60, 125]
}
df = pd.DataFrame(data)

# Features (X) und Zielvariable (y)
X = df.drop('preis', axis=1)
y = df['preis']

# 2. Modell initialisieren und trainieren
# HistGradientBoosting ist extrem schnell und effizient
model = HistGradientBoostingRegressor(max_iter=100)
model.fit(X, y)

# 3. Vorhersage für einen neuen RAM-Riegel
# Test-Szenario: 32GB, 5200MHz, DDR5, mit RGB
neuer_ram = pd.DataFrame([[32, 5200, 5, 1]], columns=X.columns)
vorhersage = model.predict(neuer_ram)

print("-" * 30)
print(f"INPUT: 32GB, 5200MHz, DDR5, RGB")
print(f"PROGNOSE: {vorhersage[0]:.2f} €")
print("-" * 30)