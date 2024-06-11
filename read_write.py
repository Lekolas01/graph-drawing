from collections.abc import Iterable
import json
from pathlib import Path


class Vertex:
    def __init__(self, id: int, x: int, y: int) -> None:
        self.id = id
        self.x = x
        self.y = y


class Edge:
    def __init__(self, source: int, target: int) -> None:
        self.source = source
        self.target = target


class Graph:
    def __init__(
        self,
        vertices: Iterable[Vertex],
        edges: Iterable[Edge],
        points: Iterable[Vertex],
    ) -> None:
        self.v = vertices
        self.e = edges
        self.p = points
