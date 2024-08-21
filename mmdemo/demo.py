"""
Demo class definition
"""

import logging
from collections import deque

from mmdemo.base_feature import BaseFeature
from mmdemo.base_interface import BaseInterface
from mmdemo.interfaces import EmptyInterface


class FeatureGraph:
    def __init__(self, targets: list[BaseFeature]) -> None:
        self.targets = targets

        self._find_all_features()
        self._find_required_features()

        if len(self.unused_features) > 0:
            logging.warning(
                f"Unused features in demo ({', '.join([str(i) for i in self.unused_features])})"
            )

        self._assert_no_cycles()

        self._find_feature_ordering()

        logging.info("Feature graph and topological sort complete")

    def _find_all_features(self):
        """
        Populate `self.features_by_id` using BFS with
        both dependencies and reverse dependencies
        """
        self.features_by_id = {id(n): n for n in self.targets}
        queue = deque(self.features_by_id.values())
        while len(queue) > 0:
            node = queue.popleft()

            for neighbor in node._deps + node._rev_deps:
                if id(neighbor) not in self.features_by_id:
                    self.features_by_id[id(neighbor)] = neighbor
                    queue.append(neighbor)

    def _find_required_features(self):
        """
        Populate `self.required_features` using BFS on targets.
        Also poupulates `self.unused_features` with all other features.
        """
        required_feature_ids = set([id(n) for n in self.targets])
        queue = deque(required_feature_ids)
        while len(queue) > 0:
            node = queue.popleft()

            for neighbor in self.features_by_id[node]._deps:
                n = id(neighbor)
                if n not in required_feature_ids:
                    required_feature_ids.add(n)
                    queue.append(n)

        unused_feature_ids = set(self.features_by_id.keys()) - required_feature_ids

        self.required_features = [self.features_by_id[i] for i in required_feature_ids]
        self.unused_features = [self.features_by_id[i] for i in unused_feature_ids]

    def _assert_no_cycles(self):
        # TODO: make sure there are no cycles and error if there are
        pass

    def _find_feature_ordering(self):
        """
        Perform topological sort on the required features. Outputs
        to `self.sorted_features`.
        """
        required_feature_ids = set([id(i) for i in self.required_features])
        sorted_feature_ids = []

        # until all required features are in the sorted list
        while set(sorted_feature_ids) != required_feature_ids:
            # add features which have all dependencies already in the sorted list
            remaining_features = required_feature_ids - set(sorted_feature_ids)
            for n in remaining_features:
                remaining_deps = [
                    i
                    for i in self.features_by_id[n]._deps
                    if id(i) not in sorted_feature_ids
                ]
                if len(remaining_deps) == 0:
                    sorted_feature_ids.append(n)

        self.sorted_features = [self.features_by_id[i] for i in sorted_feature_ids]


class Demo:
    """
    Create a runnable demo instance

    Arguments:
        targets -- list of features which will be updated
            during the demo. The dependencies of these
            features will also be updated.
    """

    def __init__(self, *, targets: list[BaseFeature]) -> None:
        self.graph = FeatureGraph(targets)

        self.interface_lookup: dict[int, BaseInterface] = {
            id(i): EmptyInterface() for i in self.graph.sorted_features
        }

    def run(self):
        """
        Run the demo
        """
        for f in self.graph.sorted_features:
            f.initialize()

        while True:
            done = False
            for f in self.graph.sorted_features:
                self.interface_lookup[id(f)]._new = False

                args = [self.interface_lookup[id(d)] for d in f._deps]
                output = f.get_output(*args)
                if output is not None:
                    assert (
                        output.is_new()
                    ), "interface._new was modified when it should not have been"
                    self.interface_lookup[id(f)] = output

                if f.is_done():
                    done = True

            if done:
                break

        for f in self.graph.sorted_features:
            f.finalize()
