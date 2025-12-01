from .constants import (
    Combi,
    COLOR_TO_NAME,
    SHAPE_TO_NAME,
    NUMBER_TO_NAME,
    FILL_TO_NAME,
)


class Solver:
    def __init__(self, shapes: list[Combi]):
        self.shapes = shapes
        self.dict = {}
        for i in range(len(self.shapes)):
            self.dict[self.shapes[i]] = i

    def solve(self) -> list[list[int]]:
        """Solve the Set game and return all valid sets."""
        for t in self.shapes:
            print(
                COLOR_TO_NAME[t[0]],
                SHAPE_TO_NAME[t[1]],
                NUMBER_TO_NAME[t[2]],
                FILL_TO_NAME[t[3]]
            )
        result = []
        for i in range(len(self.shapes)):
            for j in range(i + 1, len(self.shapes)):
                target = tuple((3 - (x + y) % 3) % 3 for x, y in zip(self.shapes[i], self.shapes[j]))
                if target in self.dict and self.dict[target] > j:
                    result.append([i, j, self.dict[target]])
        return result



if __name__ == "__main__":
    from .constants import COLOR_TO_NAME, SHAPE_TO_NAME, NUMBER_TO_NAME, FILL_TO_NAME
    
    task = [(1, 0, 0, 1), (1, 0, 0, 0), (0, 1, 1, 0), (0, 2, 0, 0), (0, 2, 2, 2), (2, 1, 0, 1), (0, 0, 1, 0), (0, 2, 2, 0), (0, 2, 0, 1), (1, 1, 0, 1), (2, 0, 1, 2), (0, 2, 0, 2)]
    for t in task:
        print(COLOR_TO_NAME[t[0]], SHAPE_TO_NAME[t[1]], NUMBER_TO_NAME[t[2]], FILL_TO_NAME[t[3]])
    solver = Solver(task)
    print(solver.solve())