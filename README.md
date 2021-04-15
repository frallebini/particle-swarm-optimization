# Particle Swarm Optimization
### Francesco Ballerini 
#### francesco.ballerini3@studio.unibo.it

<br>

A Python implementation of Particle Swarm Optimization (PSO) — as suggested in 
[[BrattonKennedy2007]](https://ieeexplore.ieee.org/document/4223164) — and its application to the Computer Vision task 
of Object Detection based on Template Matching.

Tested with:
```
matplotlib 3.3.2
numpy 1.19.2
python 3.8.5
```

## Executable files

Run the files listed below: you will be asked whether to plot the animations or output textual 
information only.

* ### [`benchmarks.py`](benchmarks.py)

    PSO applied to non-convex artificial landscapes.

* ### [`detection.py`](detection.py)

    PSO applied to Object Detection by Template Matching.

## Reproducibility

PSO is a nondeterministic algorithm; however, for the sake of reproducibility of the results, the random `seed` in 
[`swarm.py`](swarm.py) is set to a fixed value — `42`, for 
[obvious reasons](https://en.wikipedia.org/wiki/42_(number)).

Conversely, if you want to get a different outcome at each run, replace
```python
self.rng = np.random.default_rng(seed=42)
```
with
```python
self.rng = np.random.default_rng()
```