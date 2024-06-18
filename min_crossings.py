import json
import numpy as np
from pathlib import Path
from shapely.geometry import LineString
import graphviz
from argparse import ArgumentParser


class MinCrossingSolver:
    """Class for solving Minimum Crossing Problem"""

    def __init__(self, n_vertices: int, edges: np.ndarray, points: np.ndarray):
        self.e = edges  # (N, 2) - {source id, target id} N times
        self.p = points  # (N, 2) - {x coord, y coord} N times

        self.n_vertices = n_vertices  # vertices are implicitly indexed with 0 to n - 1
        self.n_edges = self.e.shape[0]
        self.n_points = self.p.shape[0]

    def __str__(self):
        ans = f"Graph(\n"
        ans += f"edges = {str(self.e)}\n"
        ans += f"points = {str(self.p)}\n)\n"
        return ans

    def solve(self) -> np.ndarray:
        """Solve the minimum crossing problem with the given graph settings. For now, it just returns any solution."""
        return np.arange(self.n_vertices, dtype=int)

    def score(self, drawing: np.ndarray) -> int:
        """Score calculation as defined in https://mozart.diei.unipg.it/gdcontest/2024/live/"""
        ans = 0
        for i, edge1 in enumerate(self.e):
            for edge2 in self.e[i + 1 :]:
                ans += self.cross(edge1, edge2)
        return ans

    def cross(self, e1: np.ndarray, e2: np.ndarray) -> int:
        line1 = LineString(self.p[e1])
        line2 = LineString(self.p[e2])
        int_pt = line1.intersection(line2)

        # case 1a: no common point
        if int_pt.geom_type == "LineString" and int_pt.length == 0.0:
            return 0

        temp = np.concatenate([self.p[e1], self.p[e2]])
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
        return self.n_vertices

    def visualize(self, drawing: np.ndarray, path: str):
        """Draw a graph for a given solution, and save it."""
        assert len(drawing) == self.n_vertices

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
        for i in range(self.n_vertices):
            p = self.p[drawing[i]]
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
        for i, (source, target) in enumerate(self.e):
            dot.edge(f"N_{source + 1}", f"N_{target + 1}")

        # draw x and y axis
        dot.render(path)


def load_problem(f: Path) -> MinCrossingSolver:
    with open(f) as json_file:
        x = json.load(json_file)

        n_vertices = len(x["nodes"])

        edges = x["edges"]
        sorted_edges = []
        for edge in edges:
            sorted_edges.append([min(edge.values()), max(edge.values())])
        sorted_edges.sort(key=lambda x: (x[0], x[1]))
        e = np.array(sorted_edges)
        points = sorted(x["points"], key=lambda x: x["id"])
        p = np.array([[t["x"] for t in points], [t["y"] for t in points]]).T

        return MinCrossingSolver(n_vertices, e, p)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("problem_name")
    args = parser.parse_args()

    problem_file_1 = Path(f"problems/{args.problem_name}.json")
    solver = load_problem(problem_file_1)
    solution = solver.solve()
    final_score = solver.score(solution)
    print(f"{final_score = }")
    solver.visualize(solution, f"solutions/{args.problem_name}/graph.gv")
