from shapely.geometry import LineString

lines = [
    ([[0, 0], [0, 1]], [[1, 0], [1, 1]]),
    ([[0, 0], [0, 1]], [[0, 0], [1, 1]]),
    ([[0, 0], [1, 1]], [[1, 0], [0, 1]]),
    ([[0, 0], [2, 2]], [[0, 2], [1, 1]]),
    ([[0, 0], [2, 2]], [[1, 1], [3, 3]]),
    ([[0, 0], [3, 3]], [[1, 1], [2, 2]]),
    ([[0, 0], [2, 2]], [[0, 0], [1, 1]]),
]

for l1, l2 in lines:
    line1 = LineString(l1)
    line2 = LineString(l2)
    int_pt = line1.intersection(line2)
    print(f"{line1 = }")
    print(f"{line2 = }")
    print(f"{int_pt = }")
    print(f"{int_pt.geom_type = }")
    print(f"{int_pt.length = }")
    print(f"{list(int_pt.coords) = }")
    print()
