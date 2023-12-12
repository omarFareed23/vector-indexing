from typing import Dict, List, Annotated
import numpy as np
from utils import VectorUtils
import os

# from scipy.cluster.vq import kmeans2
from sklearn.cluster import KMeans


class VecDB:
    def __init__(self, file_path="saved_db.csv", new_db=True) -> None:
        for file in os.listdir("./db"):
            os.remove(f"./db/{file}")
        self.file_path = file_path
        self.num_per_cluster = 1000
        self.top_level = 0
        if new_db:
            # just open new file to delete the old one
            with open(self.file_path, "w") as fout:
                # if you need to add any head to the file
                pass

    def string_rep(self, vec):
        return ",".join([str(int(e * 10)) for e in vec])

    def string_rep2(self, vec):
        return ",".join([str(e) for e in vec])

    def save_clusters(self, rows, labels, centroids, level):
        files = [
            open(f"./db/{level}_{self.string_rep(centroid['embed'])}", "a")
            for centroid in centroids
        ]
        # self.index[level + 1] = {
        #     self.string_rep(centroid): [] for centroid in centroids
        # }
        for i in range(len(rows)):
            _id = self.mp[tuple(rows[i])]
            files[labels[i]].write(f"{_id},{self.string_rep2(rows[i])}\n")
            # self.index[level + 1][centroids[labels[i]]].append()
        [f.close() for f in files]

    def num_clusters(self, rows_count):
        return int(np.ceil(rows_count / self.num_per_cluster))

    def cluster_centroids(self, centroids, level):
        if len(centroids) == 1:
            # self.top_level = level
            with open(f"./db/{level}.csv", "a") as fout:
                fout.write("last level")
            return
        kmeans = KMeans(
            n_clusters=self.num_clusters(len(centroids)), verbose=True, n_init=1
        ).fit(centroids)
        labels = kmeans.predict(centroids)
        centroids2 = kmeans.cluster_centers_
        self.save_clusters(centroids, labels, centroids2, level)
        self.cluster_centroids(centroids2, level + 1)

    def cluster_data(self, data, level):
        if len(data) == 1:
            self.top_level = level
            return
        self.mp = {tuple(row["embed"]): row["id"] for row in data}
        rows = list(map(lambda row: row["embed"], data))
        kmeans = KMeans(
            n_clusters=self.num_clusters(len(rows)), n_init=1, verbose=True
        ).fit(rows)
        labels = kmeans.predict(rows)
        centroids = kmeans.cluster_centers_
        centroids = [
            {"id": self.last_id + i + 1, "embed": centroid}
            for i, centroid in enumerate(centroids)
        ]
        self.last_id += len(centroids)
        self.save_clusters(rows, labels, centroids, level)
        self.cluster_data(centroids, level + 1)

    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        self.rows_map = {row["id"]: row["embed"] for row in rows}
        self.last_id = max(rows, key=lambda row: row["id"])["id"]

        rows = list(
            map(
                lambda row: {
                    "id": row["id"],
                    # "embed": VectorUtils.product_quantization(row["embed"], 10),
                    "embed": row["embed"],
                },
                rows,
            )
        )

        self.cluster_data(rows, 0)

    def retrive(self, query: Annotated[List[float], 70], top_k=5):
        files = os.listdir("./db")
        top_cluster = sorted(files)[-1]
        level = int(top_cluster.split("_")[0])
        # query = VectorUtils.product_quantization(query, 10)
        while True:
            # print("LEVEL : ", level, " TOP : ", top_cluster)
            with open(f"./db/{top_cluster}", "r") as fin:
                lines = fin.readlines()
                lines = [[float(num) for num in line.split(",")] for line in lines]
                scores = [(self._cal_score(query, line[1:]), line) for line in lines]
                if level == 0:
                    scores = sorted(scores, reverse=True)[:top_k]
                    return [s[1][0] for s in scores]
                else:
                    top_line = sorted(scores)[-1]
                    level -= 1
                    print("TOP LINE : ", top_line, self.string_rep(top_line[1][1:]))
                    top_cluster = f"{level}_{self.string_rep(top_line[1][1:])}"

        # top_cluster = sorted(os.listdir("./db"))[0]
        # print("TOP : ", top_cluster)

    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    def _build_index(self):
        pass
