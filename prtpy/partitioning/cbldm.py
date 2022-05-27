"""
An implementation anytime balanced number partition form:
A complete anytime algorithm for balanced number partitioning
by Stephan Mertens 1999
https://arxiv.org/abs/cs/9903011

The algorithm gets a list of numbers and returns a partition with
the smallest sum difference between 2 groups the list is divided to.
The algorithm runs until it finds the optimal partition, or it runs out of time.
"""
import sys
from typing import Callable, List, Any
import numpy as np
from prtpy import outputtypes as out, objectives as obj, Bins, BinsKeepingContents
import time
import math


def cbldm(
        bins: Bins,
        items: List[float],
        valueof: Callable[[Any], float] = lambda x: x,
        time_in_seconds: float = np.inf,
        partition_difference: int = sys.maxsize
) -> Bins:
    """
    >>> from prtpy.bins import BinsKeepingContents, BinsKeepingSums
    >>> cbldm(BinsKeepingContents(2), items=[10], time_in_seconds=1).bins
    [[], [10]]
    >>> cbldm(BinsKeepingContents(2), items=[1/2,1/3,1/5], time_in_seconds=1, partition_difference=1).bins
    [[0.5], [0.2, 0.3333333333333333]]
    >>> cbldm(BinsKeepingContents(2), items=[8,7,6,5,4], time_in_seconds=1, partition_difference=1).bins
    [[4, 6, 5], [8, 7]]
    >>> cbldm(BinsKeepingContents(2), items=[6,6,5,5,5], time_in_seconds=1, partition_difference=1).bins
    [[6, 6], [5, 5, 5]]
    >>> cbldm(BinsKeepingContents(2), items=[4,1,1,1,1], time_in_seconds=1, partition_difference=1).bins
    [[1, 1, 1], [4, 1]]

    >>> from prtpy import partition
    >>> partition(algorithm=cbldm, numbins=2, items=[10], time_in_seconds=1)
    [[], [10]]
    >>> partition(algorithm=cbldm, numbins=2, items=[10,0], time_in_seconds=1, partition_difference=3)
    [[0], [10]]
    >>> partition(algorithm=cbldm, numbins=2, items=[1/2,1/3,1/5], time_in_seconds=1, partition_difference=1)
    [[0.5], [0.2, 0.3333333333333333]]
    >>> partition(algorithm=cbldm, numbins=2, items=[6,6,5,5,5], time_in_seconds=1, partition_difference=1)
    [[6, 6], [5, 5, 5]]
    >>> partition(algorithm=cbldm, numbins=2, items=[8,7,6,5,4], time_in_seconds=1, partition_difference=1)
    [[4, 6, 5], [8, 7]]
    >>> partition(algorithm=cbldm, numbins=2, items=[4,1,1,1,1], time_in_seconds=1, partition_difference=1)
    [[1, 1, 1], [4, 1]]

    >>> partition(algorithm=cbldm, numbins=3, items=[8,7,6,5,4], time_in_seconds=1, partition_difference=1)
    Traceback (most recent call last):
        ...
    ValueError: number of bins must be 2
    >>> partition(algorithm=cbldm, numbins=2, items=[8,7,6,5,4], time_in_seconds=-1, partition_difference=1)
    Traceback (most recent call last):
        ...
    ValueError: time_in_seconds must be positive
    >>> partition(algorithm=cbldm, numbins=2, items=[8,7,6,5,4], time_in_seconds=1, partition_difference=-1)
    Traceback (most recent call last):
        ...
    ValueError: partition_difference must be a complete number and >= 1
    >>> partition(algorithm=cbldm, numbins=2, items=[8,7,6,5,4], time_in_seconds=1, partition_difference=1.5)
    Traceback (most recent call last):
        ...
    ValueError: partition_difference must be a complete number and >= 1
    >>> partition(algorithm=cbldm, numbins=2, items=[8,7,6,5,-4], time_in_seconds=1, partition_difference=1)
    Traceback (most recent call last):
        ...
    ValueError: items must be none negative

    >>> partition(algorithm=cbldm, numbins=2, items={"a":1, "b":2, "c":3, "d":3, "e":5, "f":9, "g":9})
    [['g', 'd', 'c', 'a'], ['f', 'b', 'e']]
    """
    start = time.perf_counter()
    if bins.num != 2:
        raise ValueError("number of bins must be 2")
    if time_in_seconds <= 0:
        raise ValueError("time_in_seconds must be positive")
    if partition_difference < 1 or not isinstance(partition_difference, int):
        raise ValueError("partition_difference must be a complete number and >= 1")
    sorted_items = sorted(items, key=valueof, reverse=True)
    for i in reversed(sorted_items):
        if valueof(i) >= 0:
            break
        else:
            raise ValueError("items must be none negative")

    length = len(items)
    if length == 0:  # empty items returns empty partition
        return bins

    normalised_items = cbldm_bins(sorted_items, valueof)
    alg = CBLDM_algo(length=length, time_in_seconds=time_in_seconds, len_delta=partition_difference, start=start)
    alg.part(normalised_items)
    return alg.best


class CBLDM_algo:

    def __init__(self, length, time_in_seconds, len_delta, start):
        self.sum_delta = np.inf  # partition sum difference
        self.length = length
        self.time_in_seconds = time_in_seconds
        self.len_delta = len_delta  # partition cardinal difference
        self.start = start
        self.best = BinsKeepingContents(2)
        self.best.add_item_to_bin(np.inf, 1)
        self.opt = False

    def part(self, items):
        if time.perf_counter() - self.start >= self.time_in_seconds or self.opt:
            return
        if items.get_length() == 1:  # possible partition
            cur_bin = items.get_item(0)
            cur_sum_delta = items.get_sum(cur_bin)
            cur_len_delta = items.get_cardinal(cur_bin)
            if cur_len_delta <= self.len_delta and cur_sum_delta < self.sum_delta:
                self.best = cur_bin
                self.sum_delta = cur_sum_delta
                if self.sum_delta == 0:
                    self.opt = True
                return
        else:
            if 2 * items.max_x - items.sum_x >= self.sum_delta:
                return
            # despite being in the paper, or condition breaks algorithm. breaks on [1,1,1,1,1,1,1,1,1,1]
            if 2 * items.max_m - items.sum_m > self.len_delta:  # or items.sum_m < self.difference:
                return
            if items.get_length() <= math.ceil(self.length / 2):
                items.sort()

            self.part(items.get_left())
            self.part(items.get_right())


class cbldm_bins:

    def __init__(self, items, val):
        self.val = val
        self.items = []  # list of bins, each bin contain a sub partition
        for i in items:
            b = BinsKeepingContents(2)
            b.set_valueof(self.val)
            b.add_item_to_bin(item=i, bin_index=1)
            self.items.append(b)

        self.sum_m = 0
        self.max_m = 0
        self.sum_x = 0
        self.max_x = 0
        for i in self.items:
            xi = self.get_sum(i)
            mi = self.get_cardinal(i)
            self.sum_x += xi
            self.sum_m += mi
            if xi > self.max_x:
                self.max_x = xi
            if mi > self.max_m:
                self.max_m = mi

    def get_item(self, index):
        return self.items[index]

    def get_cardinal(self, b):
        return abs(len(b.bins[0]) - len(b.bins[1]))

    def get_sum(self, b):
        return abs(b.sums[0] - b.sums[1])

    def get_length(self):
        return len(self.items)

    def sort(self):
        self.items = sorted(self.items, key=lambda item: abs(item.sums[0] - item.sums[1]), reverse=True)

    def get_left(self):
        left = cbldm_bins([], self.val)
        new_items = self.items[2:]
        split = BinsKeepingContents(2)
        split.set_valueof(self.val)
        for i in self.items[0].bins[0]:  # [small, big] + [small, big] -> [small + big, small + big]
            split.add_item_to_bin(i, 1)
        for i in self.items[0].bins[1]:
            split.add_item_to_bin(i, 0)
        for bin_i in range(2):
            for i in self.items[1].bins[bin_i]:
                split.add_item_to_bin(i, bin_i)
        split.sort()
        new_items.append(split)
        left.items = new_items
        left.sum_x = self.sum_x - self.get_sum(self.items[0]) - self.get_sum(self.items[1]) + self.get_sum(split)
        left.sum_m = self.sum_m - self.get_cardinal(self.items[0]) - self.get_cardinal(self.items[1]) + self.get_cardinal(split)
        if self.max_x == self.get_sum(self.items[0]) or self.max_x == self.get_sum(self.items[1]):
            if self.get_length() >= 3:
                left.max_x = max(self.get_sum(split), self.get_sum(self.items[2]))
            else:
                left.max_x = self.get_sum(split)
        else:
            left.max_x = max(self.get_sum(split), self.max_x)
        if self.max_m != self.get_cardinal(self.items[0]) and self.max_m != self.get_cardinal(self.items[1]):
            left.max_m = max(self.get_cardinal(split), self.max_m)
        else:
            left.max_m = max(self.get_cardinal(i) for i in new_items)
        return left

    def get_right(self):
        right = cbldm_bins([], self.val)
        new_items = self.items[2:]
        combine = BinsKeepingContents(2)
        combine.set_valueof(self.val)
        for section in range(2):  # [small, big] + [small, big] -> [small + small, big + big]
            for bin_i in range(2):
                for i in self.items[section].bins[bin_i]:
                    combine.add_item_to_bin(i, bin_i)
        combine.sort()
        new_items.append(combine)
        right.items = new_items
        right.sum_x = self.sum_x - self.get_sum(self.items[0]) - self.get_sum(self.items[1]) + self.get_sum(combine)
        right.sum_m = self.sum_m - self.get_cardinal(self.items[0]) - self.get_cardinal(self.items[1]) + self.get_cardinal(combine)
        if self.max_x == self.get_sum(self.items[0]) or self.max_x == self.get_sum(self.items[1]):
            if self.get_length() >= 3:
                right.max_x = max(self.get_sum(combine), self.get_sum(self.items[2]))
            else:
                right.max_x = self.get_sum(combine)
        else:
            right.max_x = max(self.get_sum(combine), self.max_x)
        if self.max_m != self.get_cardinal(self.items[0]) and self.max_m != self.get_cardinal(self.items[1]):
            right.max_m = max(self.get_cardinal(combine), self.max_m)
        else:
            right.max_m = max(self.get_cardinal(i) for i in new_items)
        return right


if __name__ == "__main__":
    import doctest

    (failures, tests) = doctest.testmod(report=True)
    print("{} failures, {} tests".format(failures, tests))
