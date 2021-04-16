# Particle Swarm Optimization

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

Run the files listed below to get the shown results. You will be asked whether to plot the animations or output textual 
information only.

* ### [`benchmarks.py`](benchmarks.py)

    PSO applied to non-convex artificial landscapes.
    
    ![rastrigin](images/rastrigin.gif "Rastrigin function")
    ![rosenbrock](images/rosenbrock.gif "Rosenbrock function")

* ### [`detection.py`](detection.py)

    PSO applied to Object Detection by Template Matching.
    
    ![quarto_stato](images/quarto_stato/quarto_stato.gif)
    ![arduino](images/arduino/arduino.gif)

## Reproducibility

PSO is a nondeterministic algorithm; however, for the sake of reproducibility of the results, the random `seed` in 
[`swarm.py`](https://github.com/frallebini/particle-swarm-optimization/blob/8cca3d152fcb7e8d17fcdb7f532ebb8b220d8c53/swarm.py#L52) is set to a fixed value — `42`, for 
[obvious reasons](https://en.wikipedia.org/wiki/42_(number)).

Conversely, if you want to get a different outcome at each run, replace
```python
self.rng = np.random.default_rng(seed=42)
```
with
```python
self.rng = np.random.default_rng()
```
