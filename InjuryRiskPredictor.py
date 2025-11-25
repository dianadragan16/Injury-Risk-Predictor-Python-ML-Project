import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

# 1. SIMULARE 12 SĂPTĂMÂNI EXISTENTE
num_weeks = 12
future_weeks = 4  

age = np.random.randint(18, 36)
previous_injuries = np.random.poisson(1)
position_factor = np.random.choice([0.8, 1.0, 1.2])

weekly_data = []
cumulative_load = 0
last_recovery = 0

for week in range(1, num_weeks + 1):
    games_7d = np.random.randint(0, 3)
    minutes_played = np.random.randint(0, 300) + games_7d * 20
    average_speed = np.random.normal(7, 1) * position_factor
    recovery_days = np.random.randint(0, 7)
    training_load = minutes_played * average_speed * np.random.uniform(0.8, 1.1)
    cumulative_load = cumulative_load * 0.7 + training_load
    
    risk_score = (
        0.0008 * cumulative_load + 
        0.1 * games_7d + 
        0.08 * previous_injuries + 
        0.05 * (age - 24) - 
        0.15 * recovery_days -
        0.05 * last_recovery +
        np.random.normal(0, 1.0)
    )
    injury_prob = 1 / (1 + np.exp(-risk_score))
    injury_risk = int(injury_prob > 0.7)
    
    if injury_risk:
        previous_injuries += 1
    
    last_recovery = recovery_days
    
    weekly_data.append([
        week, age, previous_injuries, games_7d, minutes_played,
        average_speed, training_load, recovery_days, cumulative_load,
        injury_risk, injury_prob
    ])

columns = [
    "week", "age", "previous_injuries", "games_last_7d",
    "minutes_played_last_7d", "average_speed", "training_load",
    "recovery_days", "cumulative_load", "injury_risk", "injury_prob"
]

df_player = pd.DataFrame(weekly_data, columns=columns)

# 2. ANTRENARE MODELE
X = df_player.drop(["week", "injury_risk", "injury_prob"], axis=1)
y = df_player["injury_risk"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

log_model = LogisticRegression()
log_model.fit(X_scaled, y)

rf_model = RandomForestClassifier(n_estimators=200)
rf_model.fit(X, y)

# 3. PREDICȚIE PENTRU SĂPTĂMÂNILE VIITOARE
pred_weeks = []
cumulative_load_future = df_player["cumulative_load"].iloc[-1]
last_recovery_future = df_player["recovery_days"].iloc[-1]
prev_injuries_future = df_player["previous_injuries"].iloc[-1]

for week in range(num_weeks + 1, num_weeks + future_weeks + 1):
    games_7d = np.random.randint(0, 3)
    minutes_played = np.random.randint(0, 300) + games_7d * 20
    average_speed = np.random.normal(7, 1) * position_factor
    recovery_days = np.random.randint(0, 7)
    training_load = minutes_played * average_speed * np.random.uniform(0.8, 1.1)
    cumulative_load_future = cumulative_load_future * 0.7 + training_load
    
    X_future = np.array([[age, prev_injuries_future, games_7d, minutes_played,
                          average_speed, training_load, recovery_days, cumulative_load_future]])
    X_future_scaled = scaler.transform(X_future)
    
    log_prob = log_model.predict_proba(X_future_scaled)[0,1]
    rf_prob = rf_model.predict_proba(X_future)[0,1]
    
    pred_weeks.append([
        week, games_7d, minutes_played, average_speed, training_load,
        recovery_days, cumulative_load_future, log_prob, rf_prob
    ])
    
    if log_prob > 0.7 or rf_prob > 0.7:
        prev_injuries_future += 1
    
    last_recovery_future = recovery_days

df_future = pd.DataFrame(pred_weeks, columns=[
    "week", "games_last_7d", "minutes_played_last_7d", "average_speed",
    "training_load", "recovery_days", "cumulative_load",
    "logistic_prob", "rf_prob"
])

# 4. CATEGORIZARE RISC

def risk_category(prob):
    if prob > 0.7:
        return "Risc mare"
    elif prob > 0.4:
        return "Risc moderat"
    else:
        return "Risc scăzut"

df_future["max_prob"] = df_future[["logistic_prob", "rf_prob"]].max(axis=1)
df_future["risk_category"] = df_future["max_prob"].apply(risk_category)

df_report = df_future[[
    "week", "games_last_7d", "minutes_played_last_7d", "training_load",
    "recovery_days", "cumulative_load", "max_prob", "risk_category"
]]
df_report["max_prob"] = df_report["max_prob"].round(2)

# 5. GRAFIC + TABEL COLORAT
fig = plt.figure(figsize=(12,8))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

# Grafic
ax = fig.add_subplot(gs[0])
ax.plot(df_player["week"], df_player["injury_prob"], marker='o', linestyle='-', label="Probabilitate istoric")
ax.plot(df_future["week"], df_future["logistic_prob"], marker='s', linestyle='--', label="Logistic Regression (prev)")
ax.plot(df_future["week"], df_future["rf_prob"], marker='d', linestyle='--', label="Random Forest (prev)")

# Highlight risc mare
high_risk_weeks = df_future[df_future["max_prob"] > 0.7]
for idx, row in high_risk_weeks.iterrows():
    ax.axvspan(row["week"]-0.5, row["week"]+0.5, color='red', alpha=0.2)

ax.set_xlabel("Săptămâna")
ax.set_ylabel("Probabilitate accidentare")
ax.set_ylim(0,1.05)
ax.set_title("Predicție risc accidentare pentru un jucător + tabel risc")
ax.grid(True)
ax.legend(loc='upper left')

# Tabel
ax_table = fig.add_subplot(gs[1])
ax_table.axis('off')

table_data = df_report[["week", "max_prob", "risk_category"]].values
table_columns = ["Săptămâna", "Probabilitate", "Risc"]
tbl = ax_table.table(cellText=table_data, colLabels=table_columns, cellLoc='center', loc='center')

for i, risk in enumerate(df_report["risk_category"]):
    color = '#CCFFCC'  # verde implicit
    if risk == "Risc mare":
        color = '#FF9999'
    elif risk == "Risc moderat":
        color = '#FFF2CC'
    for j in range(3):
        tbl[(i+1,j)].set_facecolor(color)

tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.5)

plt.tight_layout()
plt.show()

