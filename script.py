import numpy as np

for size in [
    # (int(1e5), "100k"),
    # (int(1e6), "1m"),
    (int(5e6), "5m"),
    (int(1e7), "10m"),
    (int(15e6), "15m"),
    (int(2e7), "20m"),
]:
    with open(f"./saved_db_{size[1]}.csv", "a") as fout:
        for sz in range(size[0] // int(1e5)):
            records_np = np.random.random((int(1e5), 70))
            assert(len(records_np) == int(1e5))
            records = [
                f"{i+(int(1e5) * sz)},{','.join([str(e) for e in row])}"
                for i, row in enumerate(records_np)
            ]
            fout.write("\n".join(records))
            fout.write("\n")
