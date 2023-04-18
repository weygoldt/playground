import time

import numpy as np


def timer(func):
    """
    A decorator that measures and prints the time it takes for a function to run.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time elapsed for '{func.__name__}': {elapsed_time:.6f} seconds")
        return result

    return wrapper


@timer
def merge_duplicates1(timestamps, threshold):
    """
    Compute the mean of groups of timestamps that are closer to the previous
    or consecutive timestamp than the threshold, and return all timestamps that
    are further apart from the previous or consecutive timestamp than the
    threshold in a single list.
    Parameters
    ----------
    timestamps : List[float]
        A list of sorted timestamps
    threshold : float, optional
        The threshold to group the timestamps by, default is 0.5
    Returns
    -------
    List[float]
        A list containing a list of timestamps that are further apart than
        the threshold and a list of means of the groups of timestamps that
        are closer to the previous or consecutive timestamp than the threshold.
    """
    # Initialize an empty list to store the groups of timestamps that are
    # closer to the previous or consecutive timestamp than the threshold
    groups = []

    # initialize the first group with the first timestamp
    group = [timestamps[0]]

    for i in range(1, len(timestamps)):
        # check the difference between current timestamp and previous
        # timestamp is less than the threshold
        if timestamps[i] - timestamps[i - 1] < threshold:
            # add the current timestamp to the current group
            group.append(timestamps[i])
        else:
            # if the difference is greater than the threshold
            # append the current group to the groups list
            groups.append(group)

            # start a new group with the current timestamp
            group = [timestamps[i]]

    # after iterating through all the timestamps, add the last group to the
    # groups list
    groups.append(group)

    # get the mean of each group and only include the ones that have more
    # than 1 timestamp
    means = [np.mean(group) for group in groups if len(group) > 1]

    # get the timestamps that are outliers, i.e. the ones that are alone
    # in a group
    outliers = [ts for group in groups for ts in group if len(group) == 1]

    # return the outliers and means in a single list
    return np.sort(outliers + means)


@timer
def merge_duplicates2(timestamps, threshold):
    """
    Compute the mean of groups of timestamps that are closer to the previous
    or consecutive timestamp than the threshold, and return all timestamps that
    are further apart from the previous or consecutive timestamp than the
    threshold in a single list.
    Parameters
    ----------
    timestamps : List[float]
        A list of sorted timestamps
    threshold : float, optional
        The threshold to group the timestamps by, default is 0.5
    Returns
    -------
    List[float]
        A list containing a list of timestamps that are further apart than
        the threshold and a list of means of the groups of timestamps that
        are closer to the previous or consecutive timestamp than the threshold.
    """
    # Initialize an empty list to store the groups of timestamps that are
    # closer to the previous or consecutive timestamp than the threshold
    groups = []
    # Initialize an empty list to store timestamps that are further apart
    # than the threshold
    outliers = []

    # initialize the previous timestamp with the first timestamp
    prev_ts = timestamps[0]

    # initialize the first group with the first timestamp
    group = [prev_ts]

    for i in range(1, len(timestamps)):
        # check the difference between current timestamp and previous
        # timestamp is less than the threshold
        if timestamps[i] - prev_ts < threshold:
            # add the current timestamp to the current group
            group.append(timestamps[i])
        else:
            # if the difference is greater than the threshold
            # append the current group to the groups list
            groups.append(group)

            # if the group has only one timestamp, add it to outliers
            if len(group) == 1:
                outliers.append(group[0])

            # start a new group with the current timestamp
            group = [timestamps[i]]

        # update the previous timestamp for the next iteration
        prev_ts = timestamps[i]

    # after iterating through all the timestamps, add the last group to the
    # groups list
    groups.append(group)

    # if the last group has only one timestamp, add it to outliers
    if len(group) == 1:
        outliers.append(group[0])

    # get the mean of each group and only include the ones that have more
    # than 1 timestamp
    means = [np.mean(group) for group in groups if len(group) > 1]

    # return the outliers and means in a single list
    return np.sort(outliers + means)


x = np.array([1, 2, 2, 3, 3.01, 3.02, 3.05, 4, 4, 4.99, 5])

mx1 = merge_duplicates1(x, 0.02)
mx2 = merge_duplicates2(x, 0.02)

print(x)
print(mx1)
print(mx2)
