"""Cluster (question-level) percentile bootstrap for the pooled rates shown in the figures.

Rates pool counts over trials or messages, but trials of the same question (or messages of the same trial) are
correlated. The resampling unit is therefore the cluster: each resample draws clusters with replacement, within
each stratum when a stratum column is present (difficulty bucket for questions, model for trials), and keeps
every row of a drawn cluster. One ClusterBootstrap reuses the same draws for every rate it is asked about, so
all intervals in a figure come from identical resamples. Defaults match the heterogeneous-model analysis:
2,000 resamples, seed 20260921, 95% percentile intervals.
"""
import numpy as np

RESAMPLES = 2000
SEED = 20260921


def rate(frame, num="defections", den="initially_correct"):
    """Pooled rate in percent: sum(num) / sum(den)."""
    d = frame[den].sum()
    return 100 * frame[num].sum() / d if d else np.nan


class ClusterBootstrap:
    def __init__(self, frame, cluster="question_id", strata="bucket", resamples=RESAMPLES, seed=SEED):
        info = frame.drop_duplicates(cluster).set_index(cluster)
        self.cluster = cluster
        self.clusters = list(info.index)
        n = len(self.clusters)
        labels = info[strata].to_numpy() if strata in info.columns else np.zeros(n)
        rng = np.random.default_rng(seed)
        self.weights = np.zeros((resamples, n), dtype=np.int64)
        for s in np.unique(labels):
            members = np.flatnonzero(labels == s)
            draws = rng.integers(0, len(members), size=(resamples, len(members)))
            for b in range(resamples):
                self.weights[b] += np.bincount(members[draws[b]], minlength=n)

    def ci(self, frame, num="defections", den="initially_correct", level=0.95):
        """(low, high) percentile interval, in percent, for the pooled rate of `frame`."""
        g = frame.groupby(self.cluster)[[num, den]].sum().reindex(self.clusters).fillna(0)
        n = self.weights @ g[num].to_numpy(dtype=float)
        d = self.weights @ g[den].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            rates = np.where(d > 0, 100 * n / d, np.nan)
        q = 100 * (1 - level) / 2
        lo, hi = np.nanpercentile(rates, [q, 100 - q])
        return float(lo), float(hi)


ERRORBAR = dict(fmt="none", elinewidth=0.8, capsize=2, capthick=0.8, zorder=2)


def bars(ax, x, y, lo, hi, color):
    """Vertical 95% interval bars drawn behind the markers."""
    y, lo, hi = (np.asarray(v, dtype=float) for v in (y, lo, hi))
    ax.errorbar(x, y, yerr=[y - lo, hi - y], ecolor=color, **ERRORBAR)
