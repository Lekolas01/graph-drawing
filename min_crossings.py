import json
import math
import numpy as np
from pathlib import Path
from shapely.geometry import LineString
import graphviz
from argparse import ArgumentParser
from itertools import permutations
import networkx as nx


class MinCrossingSolver:
    """Class for solving Minimum Crossing Problem"""

    def __init__(self, G: nx.Graph, points: np.ndarray):
        self.p = points  # (N, 2) - {x coord, y coord} N times

        self.G = G
        self.n_nodes = len(self.G.nodes)
        self.n_edges = len(self.G.edges)
        self.n_points = self.p.shape[0]

    def _any_solution(self) -> np.ndarray:
        return np.arange(len(self.G.nodes), dtype=int)

    def _brute_force_solution(self) -> np.ndarray:
        solutions = np.array(
            list(permutations(np.arange(self.n_points), r=self.n_nodes))
        )
        print(f"{len(solutions) = }")
        min_score, min_solution = float("inf"), np.empty(0)
        for solution in solutions:
            score = self.score(solution)
            if score >= min_score:
                continue
            min_score = score
            min_solution = solution

        return min_solution

    def solve(self) -> np.ndarray:
        self.n_checks = 0
        """Solve the minimum crossing problem with the given graph settings. For now, it just returns any solution."""
        return self._brute_force_solution()

    def score(self, solution: np.ndarray) -> int:
        """Score calculation as defined in https://mozart.diei.unipg.it/gdcontest/2024/live/"""
        pm = self.p[solution]
        ans = 0
        for i, edge1 in enumerate(self.G.edges):
            e1 = np.array(edge1)
            for edge2 in list(self.G.edges)[i + 1 :]:
                e2 = np.array(edge2)
                edge_score = self.cross(pm, e1, e2)
                ans += edge_score
        return ans

    def cross(self, points: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> int:
        self.n_checks += 1
        line1 = LineString(points[e1])
        line2 = LineString(points[e2])
        int_pt = line1.intersection(line2)

        # case 1a: no common point
        if int_pt.geom_type == "LineString" and int_pt.length == 0.0:
            return 0

        temp = np.concatenate([points[e1], points[e2]])
        n_overlaps = sum([all(int_pt.coords[0] == temp[i]) for i in range(4)])

        # case 1b: exactly one common point and that point is an endpoint of both segments
        if int_pt.geom_type == "Point" and n_overlaps == 2:
            return 0

        # case 2: the only common point is interior to both edges
        if int_pt.geom_type == "Point" and n_overlaps == 0:
            return 1

        # case 3a: the only common point lies interior to one edge, and on top of one of the other edge's end points
        # case 3b: all edge end points lie on one line, in the order src1 - src2 - target1 - target2
        # case 3c: all edge end points lie on one line, in the order src1 - src2 - target2 - target1
        # case 3d: all edge end points lie on one line, with two of the end points being the same point
        return self.n_nodes

    def visualize(self, solution: np.ndarray, path: str):
        """Draw a graph for a given solution, and save it."""
        assert len(solution) == self.n_nodes

        dot = graphviz.Graph(comment="Solution", engine="neato")
        # draw available points as squares
        for i, p in enumerate(self.p):
            dot.node(
                f"P_{i + 1}",
                f"p {i + 1}",
                shape="square",
                pos=f"{p[0]},{p[1]}!",
                fixedsize="True",
                style="filled",
                fillcolor="#FFFF9F",
            )

        # draw nodes on top of the available points
        for i in range(self.n_nodes):
            p = self.p[solution[i]]
            dot.node(
                f"N_{i + 1}",
                f"{i + 1}",
                shape="circle",
                pos=f"{p[0]},{p[1]}!",
                style="filled",
                fillcolor="#ADD8E6",
                fixedsize="True",
            )

        # draw edges
        for i, (source, target) in enumerate(self.G.edges):
            dot.edge(f"N_{source + 1}", f"N_{target + 1}")

        # draw x and y axis
        dot.render(path)


def load_problem(f: Path) -> MinCrossingSolver:
    with open(f) as json_file:
        x = json.load(json_file)

        G = nx.Graph()
        G.add_nodes_from(range(len(x["nodes"])))

        for edge in x["edges"]:
            G.add_edge(edge["source"], edge["target"])
        points = x["points"]
        p = np.array([[t["x"] for t in points], [t["y"] for t in points]]).T

        return MinCrossingSolver(G, p)


def n_checks(n: int, e: int, p: int) -> int:
    assert n >= 0 and e >= 0 and p >= 0
    return int(math.factorial(p) / (math.factorial(p - n)) * e * (e - 1) / 2)


def inspect_problem(s: MinCrossingSolver):
    print(f"Maximum number of checks: {n_checks(s.n_nodes, s.n_edges, s.n_points)}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("problem_name")
    args = parser.parse_args()

    problem_file_1 = Path(f"problems/{args.problem_name}.json")
    solver = load_problem(problem_file_1)
    inspect_problem(solver)
    solution = solver.solve()
    final_score = solver.score(solution)
    print(f"{final_score = }")
    solver.visualize(solution, f"solutions/{args.problem_name}/graph.gv")
