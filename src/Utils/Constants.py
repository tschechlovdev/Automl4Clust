import seaborn as sns

k_deviation_label = r'$\Delta k$'
# relative_k_dev_label = "\u03B4k"
relative_k_dev_label = r'$\delta k$'
relative_k_dev_label_change_rate = r'$\delta k$%'
opt_loops_label = r'Optimizer Loops $l_{i}$'
noise_label = r'$r$'

runtime_label = "Runtime (s)"

HPO_EXPERIMENT = "hpo"
CASH_EXPERIMENT = "cash"
WARMSTART_STRATEGY = "warmstart"
COLDSTART_STRATEGY = "coldstart"
OFFLINE_PHASE = "offline"
ONLINE_PHASE = "online"

HPO_RANK = "hpo rank"
CASH_RANK = "cash rank"
SYNTHETIC = "synthetic"
REAL_WORLD = "real_world"
def set_style(font_scale=2):
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")

    # Set the font to be serif, rather than sans
    sns.set(font='serif', font_scale=font_scale)

    # Make the background white, and specify the
    # specific font family
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })


set_style()
