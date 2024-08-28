"""
Demo class definition
"""

import logging
import time
from collections import deque

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

from mmdemo.base_feature import BaseFeature
from mmdemo.base_interface import BaseInterface
from mmdemo.interfaces import EmptyInterface

logger = logging.getLogger(__name__)


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

        self.evaluation_time = {k: 0.0 for k in self.interface_lookup.keys()}
        self.evaluation_time_on_new = {k: 0.0 for k in self.interface_lookup.keys()}
        self.new_count = {k: 0 for k in self.interface_lookup.keys()}

        self.has_run = False

    def run(self):
        """
        Run the demo
        """
        assert not self.has_run, "Demo has already been run"

        for f in self.graph.sorted_features:
            f.initialize()

        while True:
            done = False
            for f in self.graph.sorted_features:
                self.interface_lookup[id(f)]._new = False

                args = [self.interface_lookup[id(d)] for d in f._deps]

                start_time = time.time()
                output = f.get_output(*args)
                end_time = time.time()
                self.evaluation_time[id(f)] += end_time - start_time

                if output is not None:
                    self.evaluation_time_on_new[id(f)] += end_time - start_time
                    self.new_count[id(f)] += 1

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

        self.has_run = True

    def show_dependency_graph(self):
        """
        Show the dependency graph using networkx and matplotlib
        """
        self.graph.show()

    def print_time_benchmarks(self):
        """
        Print the time it takes for all features to be evaluated
        """
        assert self.has_run, "Run the demo first"
        abs_times = [
            (
                self.graph.features_by_id[i],
                self.evaluation_time[i],
                self.evaluation_time_on_new[i] / self.new_count[i],
            )
            for i in self.interface_lookup.keys()
            if self.new_count[i] > 0
        ]
        total_time = sum(map(lambda x: x[1], abs_times))
        rel_times = [(name, t / total_time, avg) for name, t, avg in abs_times]

        print("Total evaluation time (% of total):")
        rel_times.sort(key=lambda x: x[1], reverse=True)
        for name, t, _ in rel_times:
            if t * 100 < 1e-2:
                print("  ...")
                break
            print(f"  {name.__class__.__name__} -- {t*100:.2f}%")
        print()

        print("Average time per new output (seconds):")
        rel_times.sort(key=lambda x: x[2], reverse=True)
        for name, t, avg in rel_times:
            if avg < 1e-5:
                print("  ...")
                break
            print(f"  {name.__class__.__name__} -- {avg:.2E}")


class DemoError(Exception):
    pass


class FeatureGraph:
    def __init__(self, targets: list[BaseFeature]) -> None:
        self.targets = targets

        self._find_all_features()
        self._find_required_features()

        if len(self.unused_features) > 0:
            logger.warning(
                f"Unused features in demo ({', '.join([str(i) for i in self.unused_features])})"
            )

        self._find_feature_ordering()

        logger.info("Feature graph and topological sort complete")

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
            added_node = False
            for n in remaining_features:
                remaining_deps = [
                    i
                    for i in self.features_by_id[n]._deps
                    if id(i) not in sorted_feature_ids
                ]
                if len(remaining_deps) == 0:
                    sorted_feature_ids.append(n)
                    added_node = True

            if not added_node:
                raise DemoError(
                    "Cycle detected in dependency graph, so there is no valid topological sort"
                )

        self.sorted_features = [self.features_by_id[i] for i in sorted_feature_ids]

    def show(self):
        G = nx.DiGraph()
        for feature_id in self.features_by_id.keys():
            G.add_node(feature_id)
        for feature_id, feature in self.features_by_id.items():
            for dep in feature._deps:
                G.add_edge(id(dep), feature_id)

        # calculate classes of features and node ordering
        targets = {id(f) for f in self.targets}
        unused = {id(f) for f in self.unused_features}
        inputs = set()
        for layer, nodes in enumerate(nx.topological_generations(G)):
            if layer == 0:
                inputs.update(nodes)
            for node in nodes:
                G.nodes[node]["layer"] = layer
        pos = nx.multipartite_layout(G, subset_key="layer")

        # calculate colors of features
        INPUT_COLOR = "#aaffaa"
        TARGET_COLOR = "#ffaaaa"
        UNUSED_COLOR = "#aaaaaa"
        DEFAULT_COLOR = "#aaaaff"

        def get_node_color(f_id):
            if f_id in unused:
                return UNUSED_COLOR
            elif f_id in targets:
                return TARGET_COLOR
            elif f_id in inputs:
                return INPUT_COLOR
            else:
                return DEFAULT_COLOR

        color_map = list(map(get_node_color, self.features_by_id.keys()))

        # calculate labels of features
        labels_dict = {
            f_id: f.__class__.__name__ for f_id, f in self.features_by_id.items()
        }

        # define legend for plot
        plt.legend(
            handles=[
                mpatches.Patch(color=INPUT_COLOR, label="Inputs"),
                mpatches.Patch(color=TARGET_COLOR, label="Targets"),
                mpatches.Patch(color=DEFAULT_COLOR, label="Default"),
                mpatches.Patch(color=UNUSED_COLOR, label="Unused"),
            ]
        )

        # draw and show graph
        nx.draw_networkx(
            G,
            pos,
            node_color=color_map,
            labels=labels_dict,
            node_size=1000,
            font_size=6,
        )
        plt.show()
