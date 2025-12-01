from typing import Tuple, Literal

Combi = Tuple[Literal[0, 1, 2], Literal[0, 1, 2], Literal[0, 1, 2], Literal[0, 1, 2]]

COLOR  = {
    0: "green",
    2: "purple",
    1: "red",
}

SHAPE = {
    0: "flower",
    1: "star",
    2: "sandglass",
}

NUMBER = {
    0: 1,
    1: 2,
    2: 3,
}

FILL = {
    0: "solid",
    1: "striped",
    2: "outline",
}

class Solver:
  def __init__(self, shapes: list[Combi]):
    self.shapes = shapes
    self.dict = {}
    for i in range(len(self.shapes)):
      self.dict[self.shapes[i]] = i
  
  def solve(self) -> list[int]:
    for t in self.shapes:
      print(COLOR[t[0]], SHAPE[t[1]], NUMBER[t[2]], FILL[t[3]])
    result = []
    for i in range(len(self.shapes)):
      for j in range(i + 1, len(self.shapes)):
        target = tuple((3 - (x + y) % 3) % 3 for x, y in zip(self.shapes[i], self.shapes[j]))
        if target in self.dict and self.dict[target] > j:
          result.append([i, j, self.dict[target]])
    return result



if __name__ == "__main__":
  task = [(1, 0, 0, 1), (1, 0, 0, 0), (0, 1, 1, 0), (0, 2, 0, 0), (0, 2, 2, 2), (2, 1, 0, 1), (0, 0, 1, 0), (0, 2, 2, 0), (0, 2, 0, 1), (1, 1, 0, 1), (2, 0, 1, 2), (0, 2, 0, 2)]
  for t in task:
    print(COLOR[t[0]], SHAPE[t[1]], NUMBER[t[2]], FILL[t[3]])
  solver = Solver(task)
  print(solver.solve())