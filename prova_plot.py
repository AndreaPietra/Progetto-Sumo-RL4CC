import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
import os

# ============================================================
# CONFIGURAZIONE — modifica questi percorsi
# ============================================================"

# Opzione B: pattern file singoli (commentare DQN_CSV e decommentare questa)
DQN_CSV_PATTERN = r"C:\Users\andre\OneDrive\Desktop\Sumo\Tesi-Sumo\logs\DQN_Agente_warmup_load\DQN_SumoEnv_2026-06-11_17-06-45.429077\sumo_csv\sumo_csv_*ep*.csv"

#Baseline: 1 episodio (verrà ripetuto automaticamente N_EPISODES volte)
BASELINE_CSV    = r"C:\Users\andre\OneDrive\Desktop\Sumo\Tesi-Sumo\logs\DQN_Baseline_warmup_load\DQN_SumoEnv_2026-06-11_16-48-16.166997\sumo_csv\sumo_csv_conn0_ep1.csv"

OUTPUT_DIR      = r"C:\Users\andre\OneDrive\Desktop"

SCENARIO        = "Singolo Incrocio"
ROWS_PER_EP     = 181   # step 0-1800 con delta=10 → 181 righe per episodio
N_EPISODES      = 3
MA_WINDOW       = 10
# ============================================================

DQN_COLOR   = "#E84855"
BASE_COLOR  = "#3A86FF"
BG_COLOR    = "#000005"
PANEL_COLOR = "#e0e0ed"
GRID_COLOR  = "#2a2a3a"

metrics = [
    ("system_mean_waiting_time",              "Tempo medio di attesa (s)"),
    ("system_mean_speed",                     "Velocita media (m/s)"),
    ("system_total_stopped",                  "Numero Veicoli fermi"),
    ("system_total_waiting_time",             "Totale attesa veicoli (s)"),
    ("agents_total_stopped",                  "Veicoli fermi ai semafori"),
    ("agents_total_accumulated_waiting_time", "Tempo di attesa ai semafori (s)"),
]


def ma(series, w):
    return series.rolling(w, min_periods=1, center=True).mean()


def load_and_concat(pattern):
    """Carica e concatena tutti i CSV singoli per episodio trovati con il pattern."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Nessun file trovato con pattern: {pattern}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Caricati {len(files)} file → {len(df)} righe totali")
    return df


# --- Carica DQN ---
if "DQN_CSV_PATTERN" in dir():
    df_dqn = load_and_concat(DQN_CSV_PATTERN)


df_dqn["episode"]     = df_dqn.index // ROWS_PER_EP
df_dqn["global_step"] = df_dqn["episode"] * 1800 + df_dqn["step"]

# --- Carica Baseline e ripeti N_EPISODES volte ---
df_base_single = pd.read_csv(BASELINE_CSV)
df_base = pd.concat([df_base_single] * N_EPISODES, ignore_index=True)
df_base["global_step"] = df_base.index * (1800 / (ROWS_PER_EP - 1))
print(f"Baseline ripetuta {N_EPISODES} volte → {len(df_base)} righe totali")

# Asse x: tick ogni 10 episodi con label in secondi
ep_ticks  = [ep * 1800 for ep in range(0, N_EPISODES + 1, 10)]
ep_labels = [f"{ep * 1800:,}" for ep in range(0, N_EPISODES + 1, 10)]

# --- Genera un PNG per ogni metrica ---
for col, ylabel in metrics:
    fig, ax = plt.subplots(figsize=(18, 7), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    x_dqn  = df_dqn["global_step"]
    y_dqn  = df_dqn[col]
    x_base = df_base["global_step"]
    y_base = df_base[col]

    # DQN raw + smoothed
    ax.plot(x_dqn, y_dqn, color=DQN_COLOR, alpha=0.08, linewidth=0.4)
    ax.plot(x_dqn, ma(y_dqn, MA_WINDOW), color=DQN_COLOR, linewidth=0.5,
            label="DQN", zorder=5)

    # Baseline raw + smoothed
    ax.plot(x_base, y_base, color=BASE_COLOR, alpha=0.08, linewidth=0.4)
    ax.plot(x_base, ma(y_base, MA_WINDOW), color=BASE_COLOR, linewidth=0.5,
            linestyle="--", label="Baseline", zorder=5)

    # Separatori episodi DQN
    for ep in range(1, N_EPISODES):
        ax.axvline(x=ep * 1800, color="white", linewidth=0.2, alpha=0.12, zorder=2)

    # Bande alternate per leggibilità
    for ep in range(0, N_EPISODES, 2):
        ax.axvspan(ep * 1800, (ep + 1) * 1800, alpha=0.03, color="white", zorder=1)

    ax.set_xlabel("Step di simulazione", fontsize=14, color="#aaaacc",
                  fontfamily="monospace", labelpad=10)
    ax.set_ylabel(ylabel, fontsize=14, color="#aaaacc",
                  fontfamily="monospace", labelpad=10)
    ax.set_title(f"{SCENARIO} — {ylabel}", fontsize=18, fontweight="bold",
                 color="white", pad=15, fontfamily="monospace")
    ax.tick_params(colors="#888899", labelsize=10)
    ax.grid(color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax.set_xlim(0, N_EPISODES * 1800)
    ax.set_xticks(ep_ticks)
    ax.set_xticklabels(ep_labels, fontsize=9, color="#888899")

    p1 = mpatches.Patch(color=DQN_COLOR,  label="DQN")
    p2 = mpatches.Patch(color=BASE_COLOR, label="Baseline")
    ax.legend(handles=[p1, p2], fontsize=13, facecolor="#1e1e30",
              edgecolor="#444466", labelcolor="white", loc="upper right")

    fname = os.path.join(OUTPUT_DIR, f"plot_{col}.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"Salvato: {fname}")
    plt.show()
    plt.close()