import matplotlib.pyplot as plt
import numpy as np


def plot_cusum_results(results, threshold, filename=None, show=False, time_out=1):
    for i, (t, y_t, cp, pred, scores, regressor_preds) in enumerate(zip(*results)):
        fig, ax = plt.subplots(2, 1, figsize=(15, 10))
        ax[0].plot(t, y_t, label="data")
        # ax.plot(X_t[:len(cp)], cp, linestyle="-", linewidth=3, color="r", label="changepoint locs")
        ax[0].plot(
            t,
            regressor_preds,
            linestyle="-",
            linewidth=3,
            color="r",
            label="regressor prediction",
        )
        ax[1].plot(
            t, scores, linestyle=":", linewidth=3, color="b", label="cusum score"
        )
        ax[1].axhline(
            -np.log(threshold),
            linestyle="-",
            linewidth=3,
            color="k",
            label="wald constant threshold",
        )
        ax[0].set_xlabel("t", fontsize=30)
        ax[0].tick_params(axis="both", which="major", labelsize=15)
        ax[1].tick_params(axis="both", which="major", labelsize=15)
        ax[0].set_ylabel("y", fontsize=30)
        ax[1].set_xlabel("t", fontsize=30)
        ax[1].set_ylabel("score", fontsize=30)
        ax[0].fill_between(
            t.flatten(),
            ax[0].get_ylim()[0],
            ax[0].get_ylim()[1],
            where=scores > -np.log(threshold),
            color="red",
            alpha=0.3,
            label="CUSUM > threshold",
        )
        ax[1].fill_between(
            t.flatten(),
            ax[1].get_ylim()[0],
            ax[1].get_ylim()[1],
            where=scores > -np.log(threshold),
            color="red",
            alpha=0.3,
            label="CUSUM > threshold",
        )
        ax[1].legend(fontsize=15)
        ax[0].legend(fontsize=15)
        plt.tight_layout()
        if filename:
            plt.savefig(filename + f"plot_{i}")
        if show:
            plt.clf()
            plt.show(block=False)
            plt.pause(time_out)
            plt.close(fig)
