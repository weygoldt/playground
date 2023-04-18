import matplotlib.pyplot as plt
import numpy as np
from IPython import embed


def cluster_peaks(arr, thresh=0.5):
    """Clusters peaks of probabilitis between 0 and 1.
    Returns a list of lists where each list contains the indices of the
    all values belonging to a peak i.e. a cluster.

    Parameters
    ----------
    arr : np.ndarray
        Array of probabilities between 0 and 1.
    thresh : float, optional
        All values below are not peaks, by default 0.5

    Returns
    -------
    np.array(np.array(int))
        Each subarray contains the indices of the values belonging to a peak.
    """
    clusters = []
    cluster = []
    for i, val in enumerate(arr):
        # do nothing or append prev cluste if val is below threshold
        # then clear the current cluster
        if val <= thresh:
            if len(cluster) > 0:
                clusters.append(cluster)
                cluster = []
            continue

        # if larger than thresh
        # if first value in array, append to cluster
        # since there is no previous value to compare to
        if i == 0:
            cluster.append(i)

        # if this is the last value then there is no future value
        # to compare to so append to cluster
        elif i == len(arr) - 1:
            cluster.append(i)
            clusters.append(cluster)

        # if were at a trough then the diff between the current value and
        # the previous value will be negative and the diff between the
        # future value and the current value will be positive
        elif val - arr[i - 1] < 0 and arr[i + 1] - val > 0:
            cluster.append(i)
            clusters.append(cluster)
            cluster = []
            cluster.append(i)

        # if this is not the first value or the last value or a trough
        # then append to cluster
        else:
            cluster.append(i)

    return clusters


arr1 = np.asarray(
    [
        0,
        0,
        0,
        0,
        0,
        0.2,
        0.3,
        0.5,
        0.6,
        0.8,
        0.9,
        0.8,
        0.7,
        0.6,
        0.7,
        0.8,
        0.9,
        0.6,
        0.55,
        0.5,
        0.2,
        0,
        1,
        0,
        0,
    ]
)

arr2 = np.asarray([0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 0.9, 0.5, 0.1, 0])


arr = arr1
time = np.arange(len(arr))
indices = cluster_peaks(arr)

# embed()
# exit()

plt.plot(time, arr, "bo-")
plt.axhline(0.5, color="r")
colors = ["r", "g", "y", "c", "m", "k"]
for i, idx in enumerate(indices):
    c = colors[i % len(colors)]
    plt.plot(time[idx], arr[idx], "o", color=c)
plt.show()
