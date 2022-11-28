import stellargraph as SG
import pandas as pd

square_foo = pd.DataFrame(index=["a"])


square_bar = pd.DataFrame(
    {"y": [0.4, 0.1, 0.9], "z": [100, 200, 300]}, index=["b", "c", "d"]
)

square_edges = pd.DataFrame(
    {"source": ["a", "b", "c", "d", "a"], "target": ["b", "c", "d", "a", "c"]}
)

square_foo_and_bar = SG.StellarGraph({"foo": square_foo, "bar": square_bar}, square_edges)
print(square_foo_and_bar.info())

