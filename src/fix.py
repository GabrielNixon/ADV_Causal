import numpy as np
import matplotlib.pyplot as plt

from train_adversarial_world import run_training, make_rl_adversary_config


def moving_average(x, window=100):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x.copy()
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def build_normalized_heatmap(df, col, action_order=None, round_order=None):
    heat = (
        df.dropna(subset=[col])
        .groupby([col, "round"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    if action_order is not None:
        heat = heat.reindex(action_order, fill_value=0)
    if round_order is not None:
        heat = heat.reindex(columns=round_order, fill_value=0)

    if heat.empty:
        return heat, heat

    heat_norm = heat.div(heat.sum(axis=0), axis=1).fillna(0.0)
    return heat, heat_norm


def format_heatmap_ticks(ax, heat_norm):
    ax.set_xticks(range(len(heat_norm.columns)))
    ax.set_xticklabels([str(c) for c in heat_norm.columns])
    ax.set_yticks(range(len(heat_norm.index)))
    ylabels = []
    for v in heat_norm.index:
        if np.isclose(v, 0.0):
            ylabels.append("0%")
        elif np.isclose(v, 0.25):
            ylabels.append("25%")
        elif np.isclose(v, 0.5):
            ylabels.append("50%")
        elif np.isclose(v, 0.75):
            ylabels.append("75%")
        elif np.isclose(v, 1.0):
            ylabels.append("100%")
        else:
            ylabels.append(str(v))
    ax.set_yticklabels(ylabels)


def plot_world_panel(results, world_label, outname_prefix, window=100):
    df = results["round_logs"]

    policy_specs = [
        ("fair_scores", "fair_repay_prop", "Fair"),
        ("mid_scores", "mid_repay_prop", "Mid"),
        ("max_scores", "max_repay_prop", "Max"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    fig.suptitle(f"{world_label}: Adversary Policies and Repayment Distributions", fontsize=20, y=0.98)

    vmin = 0.0
    vmax = 1.0
    action_order = [0.0, 0.25, 0.5, 0.75, 1.0]
    round_order = list(range(1, 11))
    ims = []

    for j, (score_key, heat_col, title) in enumerate(policy_specs):
        ax = axes[0, j]
        series = moving_average(results[score_key], window=window)
        x = np.arange(1, len(series) + 1)

        ax.plot(x, series, linewidth=2)
        ax.set_title(f"{title} adversary", fontsize=15)
        ax.set_xlabel("Episode", fontsize=12)
        if j == 0:
            ax.set_ylabel("Chosen repayment", fontsize=12)
        else:
            ax.set_ylabel("")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.2)

        _, heat_norm = build_normalized_heatmap(
            df,
            heat_col,
            action_order=action_order,
            round_order=round_order,
        )
        axh = axes[1, j]

        if heat_norm.empty:
            axh.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
            axh.set_title(f"{title} heatmap", fontsize=15)
            axh.set_xlabel("Round", fontsize=12)
            if j == 0:
                axh.set_ylabel("Repayment proportion", fontsize=12)
            else:
                axh.set_ylabel("")
            continue

        im = axh.imshow(
            heat_norm.values,
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        ims.append(im)

        axh.set_title(f"{title} heatmap", fontsize=15)
        axh.set_xlabel("Round", fontsize=12)
        if j == 0:
            axh.set_ylabel("Repayment proportion", fontsize=12)
        else:
            axh.set_ylabel("")
        format_heatmap_ticks(axh, heat_norm)

    fig.subplots_adjust(left=0.06, right=0.90, top=0.87, bottom=0.08, wspace=0.18, hspace=0.24)

    cax = fig.add_axes([0.92, 0.18, 0.015, 0.62])
    cbar = fig.colorbar(ims[-1], cax=cax)
    cbar.set_label("Normalized frequency within round", fontsize=12)

    fig.savefig(f"{outname_prefix}.png", dpi=240, bbox_inches="tight")
    fig.savefig(f"{outname_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_difference_heatmaps(coord_results, ind_results, outname_prefix):
    coord_df = coord_results["round_logs"]
    ind_df = ind_results["round_logs"]

    policy_specs = [
        ("fair_repay_prop", "Fair"),
        ("mid_repay_prop", "Mid"),
        ("max_repay_prop", "Max"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle("Difference Heatmaps: Coordinated − Independent", fontsize=20, y=1.02)

    action_order = [0.0, 0.25, 0.5, 0.75, 1.0]
    round_order = list(range(1, 11))
    diffs = []

    for heat_col, _ in policy_specs:
        _, coord_norm = build_normalized_heatmap(
            coord_df,
            heat_col,
            action_order=action_order,
            round_order=round_order,
        )
        _, ind_norm = build_normalized_heatmap(
            ind_df,
            heat_col,
            action_order=action_order,
            round_order=round_order,
        )
        diff = coord_norm - ind_norm
        diffs.append(diff)

    abs_max = max(np.abs(d.values).max() for d in diffs if not d.empty)
    abs_max = max(abs_max, 1e-8)

    ims = []
    for j, ((_, title), diff) in enumerate(zip(policy_specs, diffs)):
        ax = axes[j]

        im = ax.imshow(
            diff.values,
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            vmin=-abs_max,
            vmax=abs_max,
        )
        ims.append(im)

        ax.set_title(f"{title} adversary", fontsize=15)
        ax.set_xlabel("Round", fontsize=12)
        if j == 0:
            ax.set_ylabel("Repayment proportion", fontsize=12)
        else:
            ax.set_ylabel("")
        format_heatmap_ticks(ax, diff)

    fig.subplots_adjust(left=0.06, right=0.90, top=0.82, bottom=0.15, wspace=0.18)

    cax = fig.add_axes([0.92, 0.20, 0.015, 0.56])
    cbar = fig.colorbar(ims[-1], cax=cax)
    cbar.set_label("Difference in normalized frequency", fontsize=12)

    fig.savefig(f"{outname_prefix}.png", dpi=240, bbox_inches="tight")
    fig.savefig(f"{outname_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    coordinated_cfg = make_rl_adversary_config("coordinated", "reactive")
    independent_cfg = make_rl_adversary_config("independent", "reactive")

    print("Running coordinated world...")
    coord_results = run_training(coordinated_cfg, n_episodes=3000, seed=0)
    plot_world_panel(
        coord_results,
        world_label="Coordinated World",
        outname_prefix="world1_coordinated_panel_polished",
        window=100,
    )

    print("Running independent world...")
    ind_results = run_training(independent_cfg, n_episodes=3000, seed=50000)
    plot_world_panel(
        ind_results,
        world_label="Independent World",
        outname_prefix="world2_independent_panel_polished",
        window=100,
    )

    print("Building difference heatmaps...")
    plot_difference_heatmaps(
        coord_results,
        ind_results,
        outname_prefix="coord_minus_ind_difference_heatmaps",
    )

    print("\nSaved:")
    print(" - world1_coordinated_panel_polished.png")
    print(" - world1_coordinated_panel_polished.pdf")
    print(" - world2_independent_panel_polished.png")
    print(" - world2_independent_panel_polished.pdf")
    print(" - coord_minus_ind_difference_heatmaps.png")
    print(" - coord_minus_ind_difference_heatmaps.pdf")