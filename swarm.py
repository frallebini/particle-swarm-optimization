import numpy as np


class Swarm:
    """
    A swarm is a collection of particles, where each particle is just a point in a space of given dimensionality. The
    goal of the swarm is to reach convergence of all (or, at least, most of) its particles to the global minimum point
    of an objective function. In order to do so, at each iteration every particle is updated taking into account both
    the swarm global best position and its own best position found until that point in time.

    In this implementation, each particle is only aware of its two closest neighbours, meaning that there is no actual
    global information stored in a particle, rather a local one related to the best position found among its
    neighbours. The slower convergence rate caused by this so called "ring" topology, although it may appear as a
    disadvantage at first glance, can actually allow the search process to avoid "premature" results and escape
    suboptimal solutions (see Bratton, Kennedy: Defining a Standard for Particle Swarm Optimization (2007)).
    """
    def __init__(self, bounds, f, tol, n_particles=50, c1=2.05, c2=2.05):
        """
        The default parameter values provided are the ones suggested in the cited paper.

        :param bounds: list of tuples [(x1_min, x1_max),...,(x_dim_min, x_dim_max)] such that,
                       for each particle position [x1..x_dim], xi_min <= xi <= xi_max for i=1..dim.
        :param f: objective function to minimize, assumed to take in input dim arguments and return a single value.
        :param tol: tolerance for the convergence condition (see method has_not_converged for details).
        :param n_particles: number of particles in the swarm.
        :param c1: multiplicative constant of the local best term in the velocity update rule
                   (see method update_swarm for details).
        :param c2: multiplicative constant of the personal best term in the velocity update rule
                   (see method update_swarm for details).
        """
        self.dim = len(bounds)
        self.bounds = list(zip(*bounds))  # [(x1_min,...,x_dim_min),...,(x1_max,...,x_dim_max)]
        self.f = lambda x: f(*x)  # vectorized version of the objective function
        self.tol = tol

        self.n_particles = n_particles
        self.c1 = c1
        self.c2 = c2
        c = c1 + c2
        self.constriction_factor = 2 / abs(2 - c - np.sqrt(c ** 2 - 4 * c))  # see update_swarm (and cited paper)

        # i = 1..n_particles
        # j = 1..dim
        #
        # positions[i, j]:               j-th coordinate of position of particle i
        # velocities[i, j]:              j-th coordinate of velocity of particle i
        # personal_best_positions[i, j]: j-th coordinate of the position of particle i
        #                                with lowest f-value found thus far
        # local_best_positions[i, j]:    j-th coordinate of the position of particle i
        #                                with lowest local f-value found thus far,
        #                                where "local" refers to the 2-neighbourhood of i
        self.rng = np.random.default_rng(seed=42)  # fix the seed to make experiments reproducible
        self.positions, self.velocities = self.rng.uniform(self.bounds[0], self.bounds[1], (2, n_particles, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.local_best_positions = np.zeros((n_particles, self.dim))
        self.update_local_bests()

        self.epoch_count = 0  # an epoch is an update of all the particles

    def minimize(self):
        """
        Update the swarm state until convergence is reached.

        Once the updating stops, the current particle position that minimizes the objective function should be
        interpreted as the minimum point found by the swarm. Call ``get_min_point`` to get such position.
        """
        while self.has_not_converged():
            self.update_swarm()

    def has_not_converged(self, max_n_epochs=2000):
        """
        Check termination conditions of the swarm minimization process.

        The swarm is considered to have reached convergence in two cases:

        1. The vector of the variances of the particle positions along each dimension has a small enough Euclidean
        norm, where "small enough" is determined by a chosen tolerance ``tol``. The idea is that, if the particles are
        all sufficiently close to each other, the global minimum has been found.

        2. The number of epochs reached ``max_n_epochs``.
        """
        condition1 = np.linalg.norm(np.var(self.positions, axis=0)) > self.tol
        condition2 = self.epoch_count < max_n_epochs

        if not condition1:
            print("The swarm converged successfully")
            self.print_info()
        if not condition2:
            print("Reached maximum number of epochs")
            self.print_info()

        return condition1 and condition2

    def update_swarm(self):
        """
        Update the information stored in each particle of the swarm.

        In particular, add a velocity vector to each particle position (also interpreted as a vector), such velocity
        being a linear combination of:
        1. The previous velocity.
        2. The vector from the particle to its local best.
        3. The vector from the particle to its personal best.

        The constriction factor is meant to prevent velocities from growing indefinitely over time.
        """
        r1, r2 = self.rng.uniform(size=(2, self.n_particles, self.dim))

        self.velocities = self.constriction_factor * (
                self.velocities +
                self.c1 * r1 * (self.local_best_positions - self.positions) +
                self.c2 * r2 * (self.personal_best_positions - self.positions))
        self.positions += self.velocities

        self.keep_within_bounds()

        self.update_personal_bests()
        self.update_local_bests()

        self.epoch_count += 1

    def keep_within_bounds(self):
        """If a particle position coordinate exceeds a bound, set it to the bound itself."""
        for j in range(self.dim):
            jth_coords = self.positions[:, j]
            min_pos = self.bounds[0][j]
            max_pos = self.bounds[1][j]

            jth_coords[jth_coords < min_pos] = min_pos
            jth_coords[jth_coords > max_pos] = max_pos

    def update_personal_bests(self):
        """
        Set each particle best personal position to the current one if the latter has smaller f-value than the
        previously recorded personal best.
        """
        for i in range(self.n_particles):
            self.personal_best_positions[i] = min(self.positions[i], self.personal_best_positions[i], key=self.f)

    def update_local_bests(self):
        """
        Set the local best position of each particle to the one with minimum f-value among its neighbours.
        The neighbourhood of each particle contains the particle itself and its two adjacent particles.
        """
        left_neighbours = np.roll(self.positions, -1, axis=0)
        right_neighbours = np.roll(self.positions, 1, axis=0)

        for i in range(self.n_particles):
            self.local_best_positions[i] = min(np.vstack((left_neighbours[i],
                                                          self.positions[i],
                                                          right_neighbours[i])), key=self.f)

    def get_f_values(self):
        """Return the values of the objective function in the current particle positions."""
        return [self.f(pos) for pos in self.positions]

    def get_min_point(self):
        """Return the current particle position that minimizes the objective function."""
        return self.positions[np.argmin(self.get_f_values())]

    def print_info(self):
        """Print current epoch count and minimum point found by the swarm."""
        print("Number of epochs:", self.epoch_count)
        print("Minimum: {} at {}".format(self.f(self.get_min_point()), self.get_min_point()))
