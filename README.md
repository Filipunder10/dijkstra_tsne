# Dijkstra t-SNE

## Setup
### Environment
Make sure you have **Conda** (or **Mamba**, or something else with this functionality). To set up the environment run:
```bash
conda env create -f environment.yml
conda activate mg-emb-exps-dijkstra-tsne
```

### Custom Dijkstra
Our algorithm uses a customised version of Dijkstra's algorithm, which can be found at `_shortest_path.pyx`.
To install that version, place the compiled files `_shortest_path.cp39-win_amd64.dll.a` and `_shortest_path.cp39-win_amd64.pyd`
to `<your-conda-env-path>\Lib\site-packages\scipy\sparse\csgraph` and overwrite the old files.

## Running
Our code is in `sampling_with_dijkstra.py` and can be executed by running:
```bash
python sampling_with_dijkstra.py
```