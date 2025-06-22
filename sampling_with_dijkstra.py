import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import anndata as ad
import gzip
import pickle
import os

from openTSNE import affinity
from openTSNE import TSNE
from openTSNE import TSNEEmbedding
from openTSNE import initialization
from openTSNE.affinity import joint_probabilities_nn
from openTSNE.nearest_neighbors import Annoy

from scipy.sparse import csr_matrix

from pathlib import Path

from time import time

RANDOM_SEED = 42
EPSILON = 1e-12


class PrecomputedAffinities(affinity.Affinities):
    def __init__(self, affinities, normalize=True):
        if not isinstance(affinities, sp.csr_matrix):
            affinities = sp.csr_matrix(affinities)
        if normalize:
            affinities /= np.sum(affinities)
        self.P = affinities


    def to_new(self, data, return_distances=True):
        raise RuntimeError("Precomputed affinity matrices cannot be queried.")


def reconstruct_dijkstra_distances(distances, sampled_indices, perplexity):
    distances = distances.tocsr()

    dijkstra_distances = sp.csgraph.dijkstra(distances, directed=False, indices=sampled_indices,
                                             sample_indices=sampled_indices, k=np.ceil(3 * perplexity) + 1)
    dijkstra_distances = dijkstra_distances[:, sampled_indices]

    mask = np.isfinite(dijkstra_distances) & (dijkstra_distances > EPSILON)
    rows, cols = np.where(mask)
    values = dijkstra_distances[rows, cols]

    dijkstra_distances = sp.csr_matrix((values, (rows, cols)), shape=(len(sampled_indices), len(sampled_indices)))

    dijkstra_distances = dijkstra_distances.maximum(dijkstra_distances.T)

    dijkstra_distances.setdiag(0)
    dijkstra_distances.eliminate_zeros()

    return dijkstra_distances


def normal_affinities(data, knn_index=None, perplexity=30, metric="euclidean", method="exact"):
    return affinity.PerplexityBasedNN(
        data=data if knn_index is None else None,
        knn_index=knn_index if knn_index else None,
        perplexity=perplexity,
        metric=metric,
        method=method,
        random_state=RANDOM_SEED,
        n_jobs=8
    )


def full_data_to_sample_affinities(sampled_data, perplexity=30):
    return normal_affinities(data=sampled_data, perplexity=perplexity)


def full_distances_to_sample_distances(full_distances, sampling, perplexity=30):
    dijkstra_distances = reconstruct_dijkstra_distances(full_distances, sampling, perplexity)

    n = dijkstra_distances.shape[0]

    max_k = max(np.diff(dijkstra_distances.indptr))
    # max_k = int(np.ceil(perplexity*3))

    knn_indices = np.full((n, max_k), fill_value=0, dtype=int)
    knn_dists = np.full((n, max_k), fill_value=np.inf, dtype=float)

    for i in range(n):
        row_start = dijkstra_distances.indptr[i]
        row_end = dijkstra_distances.indptr[i + 1]
        indices = dijkstra_distances.indices[row_start:row_end]
        data = dijkstra_distances.data[row_start:row_end]

        sorted_idx = np.argsort(data)
        indices = indices[sorted_idx]
        data = data[sorted_idx]

        k_i = len(indices)
        num_neighbors = min(k_i, max_k)
        knn_indices[i, :k_i] = indices[:num_neighbors]
        knn_dists[i, :k_i] = data[:num_neighbors]

    P = joint_probabilities_nn(knn_indices, knn_dists, [perplexity], True)
    return PrecomputedAffinities(P, normalize=True)


def full_affinities_to_sample_affinities(full_affinities, sampling, perplexity=30):
    reversed_affinities = full_affinities.copy()
    reversed_affinities.data = 1.0 / (reversed_affinities.data + EPSILON)

    dijkstra_distances = reconstruct_dijkstra_distances(reversed_affinities, sampling, perplexity)

    dijkstra_affinities = dijkstra_distances.copy()
    dijkstra_affinities.data = 1.0 / (dijkstra_affinities.data + EPSILON)

    return PrecomputedAffinities(dijkstra_affinities, normalize=True)


def get_sampling(data, size_prop, exp_dir, rng, run_dir_name=None):
    if run_dir_name is None:
        sampling_path = exp_dir.joinpath(f"sampling.csv")
    else:
        sampling_path = exp_dir.joinpath(f"{run_dir_name}/sampling.csv")
    if sampling_path.exists():
        sampling = np.loadtxt(sampling_path, delimiter=",")
    else:
        sampling_path.parent.mkdir(exist_ok=True, parents=True)
        sampling = rng.choice(np.arange(data.shape[0]), size=int(data.shape[0] * size_prop), replace=False)
        np.savetxt(sampling_path, sampling, delimiter=",")
    sampling = sampling.astype(int)
    return sampling


def compare_affinities(first_affinities, second_affinities):
    first_affinities.P.sort_indices()
    second_affinities.P.sort_indices()

    # print("first affinities:")
    # print(first_affinities.P[0])
    # print(distances.P[1])
    # print(distances.P[-1])
    # print(distances.P[-2])

    # print("second affinities:")
    # print(second_affinities.P[0])
    # print(normal_affinities.P[1])
    # print(normal_affinities.P[-1])
    # print(normal_affinities.P[-2])

    shared_counts = []
    neighbours_diff = []
    affinity_similarities = []

    N = first_affinities.P
    D = second_affinities.P

    for i in range(D.shape[0]):
        d_row = D[i]
        n_row = N[i]

        d_indices = d_row.indices
        d_values = d_row.data
        n_indices = n_row.indices
        n_values = n_row.data

        # Map indices to values
        d_dict = dict(zip(d_indices, d_values))
        n_dict = dict(zip(n_indices, n_values))

        shared = set(d_indices) & set(n_indices)
        shared_counts.append(len(shared) / max(len(d_indices), len(n_indices)))
        # neighbours_diff.append(min(len(d_indices), len(n_indices)) / max(len(d_indices), len(n_indices)))
        neighbours_diff.append(len(d_indices) / len(n_indices))

        if shared:
            # diffs = [abs(d_dict[j] - n_dict[j]) for j in shared]  # absolute
            # diffs = [abs(d_dict[j] - n_dict[j]) / max(d_dict[j], n_dict[j], EPSILON) for j in shared]  # relative
            similarity = [min(d_dict[j], n_dict[j]) / max(d_dict[j], n_dict[j]) for j in shared]
            affinity_similarities.append(np.mean(similarity))

    avg_shared = np.mean(shared_counts)
    # print(f"Average shared indices per row: {avg_shared:.4%}")

    avg_rel_diff = np.mean(affinity_similarities)
    # print(f"Average relative affinity difference on shared points: {avg_rel_diff:.4%}")

    avg_neigh_count_diff = np.mean(neighbours_diff)

    return avg_shared, avg_rel_diff, avg_neigh_count_diff


def plot_comparison(results, perplexities):
    plt.figure(figsize=(10, 6))
    for comparison in results:
        avg_shared_list, avg_rel_sim_list, avg_neigh_count_diff = zip(*results[comparison])

        perplexities_count = range(len(perplexities))

        perplexities_sorted = [perplexities[i] for i in perplexities_count]
        avg_shared_sorted = [avg_shared_list[i] for i in perplexities_count]
        avg_rel_diff_sorted = [avg_rel_sim_list[i] for i in perplexities_count]
        avg_neigh_count_diff = [avg_neigh_count_diff[i] for i in perplexities_count]

        plt.plot(perplexities_sorted, avg_neigh_count_diff, marker='s',
                 label=f'Ratio of Count of Number of Neighbours')
        plt.plot(perplexities_sorted, avg_shared_sorted, marker='o',
                 label=f'Ratio of Shared Indices')
        plt.plot(perplexities_sorted, avg_rel_diff_sorted, marker='^',
                 label=f'Similarity')

    # plt.xlabel("Perplexity")
    # plt.ylabel("Average Value")
    # plt.title(f"Evolution of Affinity Metrics as Perplexity Increases for {list(results.keys())[0]}")

    plt.ylim(0, 1.1)

    min_perp = min(perplexities)
    max_perp = max(perplexities)
    plt.xticks(range(min_perp, max_perp + 1, 5), fontsize=20)
    plt.yticks(np.arange(0, 1.11, 0.1), fontsize=20)

    # plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("affinity_comparison_plot_same_2v3.png")
    plt.show()


def create_embedding(perp, init, affinities, exp_dir, run_dir_name=None):
    if run_dir_name is None:
        embedding_path = exp_dir.joinpath(f"embedding.csv")
    else:
        embedding_path = exp_dir.joinpath(f"{run_dir_name}/embedding.csv")

    embedding_path.parent.mkdir(exist_ok=True, parents=True)

    embedding = TSNEEmbedding(
        init,
        affinities,
        negative_gradient_method="fft",
        n_jobs=8
    )

    embedding = embedding.optimize(n_iter=250, exaggeration=12, momentum=0.5)
    embedding = embedding.optimize(n_iter=750, exaggeration=1, momentum=0.8)

    embedding = np.array(embedding)

    np.savetxt(embedding_path, embedding, delimiter=",")

    return embedding


def get_data(exp_dir, dataset_name, num_pca_components=50):
    data_path = exp_dir.joinpath("data.pkl")
    if data_path.exists():
        # data = np.loadtxt(data_path, delimiter=",")
        with gzip.open(data_path, "rb") as f:
            data = pickle.load(f)
    else:
        if dataset_name == "mnist":
            data = get_mnist("../data/mnist")[0]
        elif dataset_name == "celegans":
            data = np.asarray(load_celegans("../data")[0].todense())
        else:
            assert False, "Unsupported dataset ..."
        if num_pca_components is not None and num_pca_components > 0:
            print("Original data shape: ", data.shape)
            num_pca_components = np.min([num_pca_components, data.shape[0], data.shape[1]])
            data = initialization.pca(data, n_components=num_pca_components, random_state=RANDOM_SEED)
            data = np.array(data)
            print("Reduced data shape: ", data.shape)
        with gzip.open(data_path, "wb") as f:
            pickle.dump(data, f)
        # np.savetxt(data_path, data, delimiter=",")
    return data


def get_labels(exp_dir, dataset_name):
    labels_path = exp_dir.joinpath("labels.csv")
    if labels_path.exists():
        labels = np.loadtxt(labels_path, delimiter=",")
    else:
        if dataset_name == "mnist":
            labels = get_mnist("../data/mnist")[1]
        elif dataset_name == "celegans":
            labels = load_celegans("../data")[1]
        else:
            assert False, "Unsupported dataset ..."
        np.savetxt(labels_path, labels, delimiter=",")
    return labels


def get_init(data, exp_dir):
    init_path = exp_dir.joinpath("init.csv")
    if init_path.exists():
        init = np.loadtxt(init_path, delimiter=",")
    else:
        init = initialization.pca(data, random_state=RANDOM_SEED)
        np.savetxt(init_path, init, delimiter=",")
    return init


def load_celegans(data_home, return_X_y=True):
    """
    Loads C-ELEGANS data available at https://data.caltech.edu/records/1945

    Parameters
    __________
    data_home : str, optional
        Locations of the folder where the datasets are stored.
    return_X_y: bool, optional
        If True, method only returns tuple with the data and its labels.
    """
    # Use default location
    if data_home is None:
        data_home = Path.joinpath(Path(__file__).parent, "datasets")
    else:
        data_home = Path(str(data_home))  # quick fix to deal with incoming os.paths

    full_path = Path.joinpath(data_home, "c_elegans")

    ad_obj = ad.read_h5ad(str(Path.joinpath(full_path, "packer2019.h5ad")))
    X = ad_obj.X

    labels_str = np.array(ad_obj.obs.cell_type)

    _, labels = np.unique(labels_str, return_inverse=True)

    print(labels.shape)

    if return_X_y:
        return X, labels
    else:
        return X


def get_mnist(path, kind="all"):
    """
    TODO Docstring for get_mnist.
    """
    path_to_data = Path(path)
    if not path_to_data.exists():
        raise Exception("mnist data was not found at {}".format(path_to_data))

    labels_path_train = os.path.join(path_to_data, 'train-labels-idx1-ubyte.gz')
    labels_path_test = os.path.join(path_to_data, 't10k-labels-idx1-ubyte.gz')
    images_path_train = os.path.join(path_to_data, 'train-images-idx3-ubyte.gz')
    images_path_test = os.path.join(path_to_data, 't10k-images-idx3-ubyte.gz')

    labels_dict = dict()
    images_dict = dict()

    if kind == 'all' or kind == 'train':
        with gzip.open(labels_path_train, 'rb') as lbpath:
            br = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
            labels_dict["train"] = br

    if kind == 'all' or kind == 'test':
        with gzip.open(labels_path_test, 'rb') as lbpath:
            br = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
            labels_dict["test"] = br

    if kind == 'all' or kind == 'train':
        with gzip.open(images_path_train, 'rb') as imgpath:
            br = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)
            images = br.reshape(len(labels_dict["train"]), 784)
            images_dict["train"] = images

    if kind == 'all' or kind == 'test':
        with gzip.open(images_path_test, 'rb') as imgpath:
            br = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)
            images = br.reshape(len(labels_dict["test"]), 784)
            images_dict["test"] = images

    labels = np.concatenate(list(labels_dict.values()), axis=0)
    images = np.concatenate(list(images_dict.values()), axis=0)

    return images, labels


def plot_grid_embeddings(ncols, nrows, embeddings, titles=None, labels=None, fn=None):

    if not titles:
        titles = [i for i in enumerate(embeddings)]

    # print(type(labels) == np.ndarray)
    if type(labels) == np.ndarray:
        labels = [labels for _ in range(len(embeddings))]
    elif labels is None:
        labels = [np.zeros_like(embeddings[0].shape[0]) for _ in range(len(embeddings))]

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10,10))

    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
        s = 0.1
    else:
        axs = [axs]
        s = 7

    for i, emb in enumerate(embeddings):
        title = titles[i]
        # print(title)

        # axs[i].set_title(title)
        axs[i].scatter(emb[:, 1], emb[:, 0], s=s, c=labels[i], cmap=plt.get_cmap("tab10"), alpha=0.5)

        # Remove the ticks on both axes
        axs[i].tick_params(axis='both', which='both', length=0)

        # Remove the labels on both axes
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])

    if fn:
        fig.savefig(fn, bbox_inches='tight', dpi=300)
        plt.close()
        # print(f"Done saving embeddings grid")
    else:
        plt.show()


def distance_matrix_and_annoy(data, num_neighbors, epsilon=False, num_threads=1, rnd_state=42):
    """
    TODO Docstring for distance_matrix_and_annoy
    """
    # Create an Annoy object to get the neighborhood indices and the corresponding distances
    annoy = Annoy(
        data=data,
        k=num_neighbors,
        metric="euclidean",
        n_jobs=num_threads,
        random_state=rnd_state,
        verbose=False
    )
    neighbors, distances = annoy.build()
    # Remove explicit zero entries
    if epsilon:
        distances[distances == 0.0] = epsilon
    # Convert the information to a CSR matrix
    row_indices = np.repeat(
        np.arange(data.shape[0]),
        num_neighbors
    )
    data_distance_matrix = csr_matrix(
        (distances.flatten(), (row_indices, neighbors.flatten())),
        shape=(data.shape[0], data.shape[0])
    )
    return data_distance_matrix, annoy


# if __name__ == "__main__":
#     dataset_name = "mnist"
#
#     exp_dir = Path(f"sampling_with_dijkstra_test/{dataset_name}")
#     exp_dir.mkdir(exist_ok=True, parents=True)
#
#     data = get_data(exp_dir=exp_dir, dataset_name=dataset_name)
#     labels = get_labels(exp_dir=exp_dir, dataset_name=dataset_name)
#     init = get_init(data=data, exp_dir=exp_dir)
#
#     rng = np.random.default_rng(seed=RANDOM_SEED)
#
#     initial_perplexity = 300
#     sample_rate = 0.1
#     # perplexities = [100, 50, 20, 10, 5]
#     # perplexities = [5, 10]
#     perplexities = [5, 10, 20, 50, 100]
#     # perplexities = [5]
#     # perplexities = [30]
#
#     # t_setup = time()
#
#     # print(f"setup: {time() - t_setup}")
#
#     results = {}
#
#     plot_labels = [
#                    # "Normal vs Dijkstra on Distances",
#                    # "Normal vs Dijkstra on Affinities",
#                    "Dijkstra on Distances vs Dijkstra on Affinities"
#                    ]
#     #
#     results[plot_labels[0]] = []
#     # results[plot_labels[1]] = []
#     # results[plot_labels[2]] = []
#     for perplexity in perplexities:
#
#         full_distances, annoy = distance_matrix_and_annoy(data, perplexity, num_threads=8)
#         full_affinities = normal_affinities(data=None, knn_index=annoy, perplexity=perplexity)
#         full_P = full_affinities.P
#
#         run_dir_name = f"{str(perplexity).replace('.', 'p')}"
#
#         sampling = get_sampling(data, size_prop=sample_rate, exp_dir=exp_dir, rng=rng, run_dir_name=run_dir_name)
#         num_points = sampling.size
#
#         sampled_data = data[sampling]
#         sampled_labels = labels[sampling]
#         sampled_init = init[sampling]
#
#         t_total = time()
#         run_count = 1
#         for i in range(run_count):
#             # t_affinities = time()
#
#             # aff1 = full_data_to_sample_affinities(sampled_data, perplexity)  # normal dijkstra
#             aff2 = full_distances_to_sample_distances(full_distances, sampling, perplexity)  # dijkstra on distances
#             aff3 = full_affinities_to_sample_affinities(full_P, sampling, perplexity)  # dijkstra on affinities
#
#             # print(f"affinities for perplexity {perplexity}: {time() - t_affinities}")
#
#             results[plot_labels[0]].append(compare_affinities(aff2, aff3))
#             # results[plot_labels[0]].append(compare_affinities(aff1, aff3))
#             # results[plot_labels[0]].append(compare_affinities(aff2, aff3))
#
#             # t_embeddings = time()
#
#             # grid_path = exp_dir.joinpath(f"vis1_{initial_perplexity}_{perplexity}_{i}.png")
#             # embeddings = create_embedding(perplexity, init=sampled_init, affinities=aff1, exp_dir=exp_dir,
#             #                            run_dir_name=run_dir_name)
#             # # embeddings = get_embedding(perplexity, init=sampled_init, affinities=aff2, exp_dir=exp_dir,
#             # #                            run_dir_name=run_dir_name)
#             # plot_grid_embeddings(1, 1, embeddings=[embeddings],
#             #                      titles=["Visualisation for Normal"],
#             #                      labels=labels[sampling], fn=grid_path)
#
#             # print(f"embeddings for perplexity {perplexity}: {time() - t_embeddings}")
#         # print(f"average time for {run_count} runs for perplexity {perplexity}: {(time() - t_total)/run_count}")
#     plot_comparison(results, perplexities)


if __name__ == "__main__":
    dataset_name = "celegans"

    exp_dir = Path(f"sampling_with_dijkstra_test/{dataset_name}")
    exp_dir.mkdir(exist_ok=True, parents=True)

    data = get_data(exp_dir=exp_dir, dataset_name=dataset_name)
    labels = get_labels(exp_dir=exp_dir, dataset_name=dataset_name)
    init = get_init(data=data, exp_dir=exp_dir)

    rng = np.random.default_rng(seed=RANDOM_SEED)

    initial_perplexity = 300
    sample_rate = 0.1
    # perplexities = [100, 50, 20, 10, 5]
    # perplexities = [5, 10, 20, 50, 100]
    # perplexities = [20, 30, 40, 50]
    # perplexities = [5, 10]
    # perplexities = [5]
    # perplexities = [50]
    perplexities = [30]

    # t_setup = time()

    full_distances, annoy = distance_matrix_and_annoy(data, initial_perplexity, num_threads=8)
    full_affinities = normal_affinities(data=None, knn_index=annoy, perplexity=initial_perplexity)
    full_P = full_affinities.P

    # print(f"setup: {time() - t_setup}")

    # results = {}
    #
    # plot_labels = [
    #                # "Normal vs Dijkstra on Distances",
    #                # "Normal vs Dijkstra on Affinities",
    #                # "Dijkstra on Distances vs Dijkstra on Affinities"
    #                ]
    #
    # results[plot_labels[0]] = []
    # results[plot_labels[1]] = []
    # results[plot_labels[2]] = []
    for perplexity in perplexities:
        run_dir_name = f"{str(perplexity).replace('.', 'p')}"

        sampling = get_sampling(data, size_prop=sample_rate, exp_dir=exp_dir, rng=rng, run_dir_name=run_dir_name)
        num_points = sampling.size

        sampled_data = data[sampling]
        sampled_labels = labels[sampling]
        sampled_init = init[sampling]

        # t_total = time()
        run_count = 1
        for i in range(run_count):
            # t_affinities = time()

            # aff1 = full_data_to_sample_affinities(sampled_data, perplexity)  # normal dijkstra
            # aff2 = full_distances_to_sample_distances(full_distances, sampling, perplexity)  # dijkstra on distances
            aff3 = full_affinities_to_sample_affinities(full_P, sampling, perplexity)  # dijkstra on affinities

            # print(f"affinities for perplexity {perplexity}: {time() - t_affinities}")

            # results[plot_labels[0]].append(compare_affinities(aff2, aff3))
            # results[plot_labels[0]].append(compare_affinities(aff1, aff3))
            # results[plot_labels[0]].append(compare_affinities(aff2, aff3))

            # t_embeddings = time()

            grid_path = exp_dir.joinpath(f"vis3_{initial_perplexity}_{perplexity}_{i}.png")
            embeddings = create_embedding(perplexity, init=sampled_init, affinities=aff3, exp_dir=exp_dir,
                                       run_dir_name=run_dir_name)
            # embeddings = get_embedding(perplexity, init=sampled_init, affinities=aff2, exp_dir=exp_dir,
            #                            run_dir_name=run_dir_name)
            plot_grid_embeddings(1, 1, embeddings=[embeddings],
                                 titles=[""],
                                 labels=labels[sampling], fn=grid_path)

            # print(f"embeddings for perplexity {perplexity}: {time() - t_embeddings}")
        # print(f"average time for {run_count} runs for perplexity {perplexity}: {(time() - t_total)/run_count}")
    # plot_comparison(results, perplexities)

