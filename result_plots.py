import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# =========================
# 1. Training curves
# =========================

d3_dem_t10 = np.load("jobs/stats/d3_stats_dem_real_phaseA_t10.npy")
d3_real_t10 = np.load("jobs/stats/d3_stats_dem_real_phaseB_t10.npy")
d3_dem_t20 = np.load("jobs/stats/d3_stats_dem_real_phaseA_t20.npy")
d3_real_t20 = np.load("jobs/stats/d3_stats_dem_real_phaseB_t20.npy")

d5_real_t10 = np.load("jobs/stats/d5_stats_real_t10.npy")

# Rad 4 = train accuracy
# Rad 10 = validation accuracy

d3_dem_train_t10 = 1 - d3_dem_t10[4]
d3_dem_val_t10 = 1 - d3_dem_t10[10]
d3_dem_train_t20 = 1 - d3_dem_t20[4]
d3_dem_val_t20 = 1 - d3_dem_t20[10]

d3_real_train_t10 = 1 - d3_real_t10[4]
d3_real_val_t10 = 1 - d3_real_t10[10]
d3_real_train_t20 = 1 - d3_real_t20[4]
d3_real_val_t20 = 1 - d3_real_t20[10]


d5_real_train_t10 = 1 - d5_real_t10[4]
d5_real_val_t10 = 1 - d5_real_t10[10]


# =========================
# DEM training plot
# =========================

epochs_d3_t10 = np.arange(1, len(d3_dem_train_t10) + 1)
epochs_d3_t20 = np.arange(1, len(d3_dem_train_t20) + 1)

plt.figure(figsize=(8, 5))

plt.plot(epochs_d3_t10, d3_dem_train_t10, linewidth=2, color="darkgreen", label="d=3 T=10 Träning")
plt.plot(epochs_d3_t10, d3_dem_val_t10, "--", linewidth=2, color="darkgreen", label="d=3 T=10 Validering")
plt.plot(epochs_d3_t20, d3_dem_train_t20, linewidth=2, color="darkorange", label="d=3 T=20 Träning")
plt.plot(epochs_d3_t20, d3_dem_val_t20, "--", linewidth=2, color="darkorange", label="d=3 T=20 Validering")

plt.xlabel("Epok")
plt.ylabel("Logisk felfrekvens")
plt.title("Träning på DEM-data")
# plt.legend(frameon=False)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("logical_error_dem_training.png", bbox_inches="tight")
plt.show()


# =========================
# Real-data fine-tuning plot
# =========================

epochs_d3_t10 = np.arange(1, len(d3_real_train_t10) + 1)
epochs_d3_t20 = np.arange(1, len(d3_real_train_t20) + 1)
epochs_d5_t10 = np.arange(1, len(d5_real_train_t10) + 1)

plt.figure(figsize=(10, 5))

plt.plot(epochs_d3_t10, d3_real_train_t10, linewidth=2, color="darkgreen", label="d=3 T=10 Träning")
plt.plot(epochs_d3_t10, d3_real_val_t10, "--", linewidth=2, color="darkgreen", label="d=3 T=10 Validering")
plt.plot(epochs_d3_t20, d3_real_train_t20, linewidth=2, color="darkorange", label="d=3 T=20 Träning")
plt.plot(epochs_d3_t20, d3_real_val_t20, "--", linewidth=2, color="darkorange", label="d=3 T=20 Validering")
plt.plot(epochs_d5_t10, d5_real_train_t10, linewidth=2, color="darkblue", label="d=5 T=10 Träning")
plt.plot(epochs_d5_t10, d5_real_val_t10, "--", linewidth=2, color="darkblue", label="d=5 T=10 Validering")

plt.xlabel("Epok")
plt.ylabel("Logisk felfrekvens")
plt.title("Finjustering på verklig IBM-data")
plt.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("logical_error_real_training.png", bbox_inches="tight")
plt.show()


# =========================
# 2. Final benchmark plot
# =========================

labels = [
    "d=3, T=10",
    "d=3, T=20",
    "d=5, T=10",
    "d=5, T=20",
]

# Fyll i värden när de kommer
trained_gnn = [
    0.1352,   # d3 T10
    0.2154,     # d3 T20
    0.2393,     # d5 T10
    np.nan,     # d5 T20
]

mwpm = [
    0.1844,   # d3 T10
    0.2618,   # d3 T20
    0.3397,     # d5 T10
    0.4465,     # d5 T20
]

untrained_gnn = [
    0.2378,   # d3 T10
    0.3666,   # d3 T20
    0.2791,     # d5 T10
    0.4090,     # d5 T20
]

x = np.arange(len(labels))
width = 0.1

plt.figure(figsize=(8, 5))

plt.bar(x - width, trained_gnn, width, color="darkgreen", label="Tränat GNN-RNN")
plt.bar(x, mwpm, width, color="darkblue", label="MWPM")
plt.bar(x + width, untrained_gnn, width, color="darkorange", label="Otränat GNN-RNN")

plt.xticks(x, labels)
plt.ylabel("Logisk felfrekvens")
plt.title("Slutlig avkodningsprestanda på IBM-testdata")
plt.legend(frameon=False)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("final_decoder_comparison.png", bbox_inches="tight")
plt.show()
