# %%
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

plt.style.use(hep.style.ROOT)
plt.rcParams["legend.edgecolor"] = "black"
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = .8
plt.rcParams["legend.borderpad"] = 0.4
plt.rcParams["legend.borderaxespad"] = 0.5
plt.rcParams["legend.handlelength"] = 1.0
plt.rcParams['xtick.major.width'] = 0.0
plt.rcParams['ytick.major.width'] = 0.0
plt.rcParams['xtick.minor.width'] = 0.0
plt.rcParams['ytick.minor.width'] = 0.0
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['xtick.minor.size'] = 0
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 10,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})
# %%

def derivative_pricing_layout():
    V0 = .4
    X = np.linspace(0, .8, 100)

    fig, ax = plt.subplots(figsize=(2.5, 2))

# Move axes to origin
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")

# Hide the other two spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

# Optional: set limits
    ax.set_xlim(-0.1, 1)
    ax.set_ylim(-0.1, 1)

    ax.set_xlabel("t", labelpad=-10)
    ax.set_ylabel("S", rotation=0, labelpad=-15)

    ax.set_xticks([.8], labels=[r"$T$"])
    ax.set_yticks([.425], labels=[r"$S(0)$"])

    ax.scatter(0, V0, marker="o", color="orange", s=20)
    ax.axhline(V0, xmin=.1, xmax=.8, color="orange", linewidth=1, ls='--')

    ax.plot(X, V0 + .3*np.sin(X), color="orange")
    ax.plot(X, V0 + -.3*np.sin(3*X), color="orange")
    ax.plot(X, V0 + X/3 + .3*np.sin(3*X), color="orange")
    ax.plot(X, V0 + X/3 + .1*np.sin(10*X), color="orange")

    return fig, ax

# %%
derivative_pricing_layout()
plt.savefig("./derivative_pricing.pdf", bbox_inches="tight")
# plt.close()
# %%
ax, fig = derivative_pricing_layout()
ax.text(
    .85, 0.53,
    r"$\left.\rule{0pt}{3em}\right\}$",
    fontsize=12
)
ax.text(
    .92, 0.53,
    r"$\mathbb{E}[S(T)]$",
    fontsize=12
)
    
V0 = .4
X = np.linspace(0, .8, 100)
plt.scatter([X[-1]]*4, 
    [
        V0 + .3*np.sin(X[-1]),
        V0 + -.3*np.sin(3*X[-1]),
        V0 + X[-1]/3 + .3*np.sin(3*X[-1]),
        V0 + X[-1]/3 + .1*np.sin(10*X[-1]),
    ],
            marker="o", color="orange", s=20
)
plt.savefig("./derivative_pricing_european.pdf", bbox_inches="tight")
# plt.close()
# %%
ax, fig = derivative_pricing_layout()
ax.text(
    .85, 0.53,
    r"$\left.\rule{0pt}{3em}\right\}$",
    fontsize=12
)
ax.text(
    .92, 0.53,
    r"$\mathbb{E}[\bar{S}]$",
    fontsize=12
)
V0 = .4
X = np.linspace(0, .8, 100)
plt.scatter([X[-1]]*4, 
    [
        V0 + .3*np.sin(X[-1]),
        V0 + -.3*np.sin(3*X[-1]),
        V0 + X[-1]/3 + .3*np.sin(3*X[-1]),
        V0 + X[-1]/3 + .1*np.sin(10*X[-1]),
    ],
            marker="o", color="orange", s=20
)
for t in np.arange(0, 100, 10):
    plt.scatter([X[t]]*4, 
        [
            V0 + .3*np.sin(X[t]),
            V0 + -.3*np.sin(3*X[t]),
            V0 + X[t]/3 + .3*np.sin(3*X[t]),
            V0 + X[t]/3 + .1*np.sin(10*X[t]),
        ],
                marker="o", color="orange", s=20
    )
plt.savefig("./derivative_pricing_asian.pdf", bbox_inches="tight")
# plt.close()
# %%
