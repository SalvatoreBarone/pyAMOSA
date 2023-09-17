# pyAMOSA 
pyAMOSA is a python implementation of the Archived Multi-Objective Simulated Annealing optimization heuristic [1].

## Installing the module
You can install this module using ```pip```.

```bash
pip install pyamosa
```

If you want to install it manually, you can do it by simply running the provided ```setup.py``` script as follows:
```bash
$ python3 setup.py install
```

# Defining and solving a problem

In pyAMOSA a problem is defined by an object that contains some metadata, for instance the number of decision variables, 
their data type, the number of objectives, the number of constraints, lower and upper bounds for decision variables.
These attributes are supposed to be defined in the constructor. 

Objects defining problems can be defined by inheriting from ```AMOSA.Problem```, and thus overriding the ```__init__``` 
method to define the above-mentioned attributes. 

The actual objective-functions evaluation takes place in the ```evaluate ``` method, which aims to fill the ```out``` 
dictionary with approriate data. The objective-function values are supposed to be written into ```out["f"]```, while 
the constraints into ```out["g"]```, if the ```num_of_constraints``` attribute is greater than zero.
The ```evaluate``` will be called for each solution, allowing easy parallelization using processes, and regardless of 
the number of solutions being asked to be evaluated, it retrieves a vector ```x``` of values for decision variables of
the problem.

How the objective-functions and constraint values are calculated is irrelevant from pyAMOSA's point of view. 
Whether it is a simple mathematical equation or a discrete-event simulation, you only have to ensure that for each input
the corresponding values have been set.

Suppose you want to solve the ZDT1 problem [2], i.e. 

<img src="https://render.githubusercontent.com/render/math?math=min\begin{cases}f_1(x)=x_1\\f_2(x)=g(x)\cdot h(f_1(x),g(x))\end{cases}">

where

<img src="https://render.githubusercontent.com/render/math?math=g(x)=1+\frac{9}{29}\left(\sum_{i=2}^n x_i\right)">

and

<img src="https://render.githubusercontent.com/render/math?math=h(f(x),g(x))=1-\sqrt{\frac{f(x)}{g(x)}}">

with

<img src="https://render.githubusercontent.com/render/math?math=0\le x_i\le1 i=1 ... 30">



```python
import pyamosa, numpy as np

class ZDT1(pyamosa.Problem):
    n_var = 30

    def __init__(self):

        pyamosa.Problem.__init__(self, ZDT1.n_var, [pyamosa.Type.REAL] * ZDT1.n_var, [0.0]*ZDT1.n_var, [1.0] * ZDT1.n_var, 2, 0)

    def evaluate(self, x, out):
        f = x[0]
        g = 1 + 9 * sum(x[1:]) / (self.num_of_variables - 1)
        h = 1 - np.sqrt(f / g)
        out["f"] = [f, g * h ]
```

Now, you have to build a proper problem object and also an optimization-engine, as follows.
```python
if __name__ == "__main__":
    problem = ZDT1()

    optimizer = pyamosa.Optimizer(...)
```

The ```pyamosa.Optimizer``` class allows setting a vast plethora of configuration parameters governing the behavior of the 
heuristic. You must do it by creating an ```pyamosa.Config``` object, as follows

```python
    config = pyamosa.Config()
    config.archive_hard_limit = 100
    config.archive_soft_limit = 200
    config.archive_gamma = 2
    config.clustering_max_iterations = 300
    config.hill_climbing_iterations = 500
    config.initial_temperature = 500
    config.cooling_factor = 0.9
    config.annealing_iterations = 1000
    config.annealing_strength = 1
    config.multiprocess_enabled = True
```

 - the ```archive_hard_limit``` attribute allows setting the HL parameter of the heuristic, i.e., the hard limit on the archive size;
 - the ```archive_soft_limit``` attribute allows setting the SL parameter of the heuristic, i.e., the soft limit on the archive size;
 - the ```hill_climbing_iterations``` is the number of refinement iterations performed during the initial hill-climbing refinement;
 - the ```archive_gamma``` attribute allows governing the amount of initial candidate solutions that are generated duting the archive initialization; 
 - the ```clustering_max_iterations``` allows foverning the maximum iterations performed during k-means clustering;
 - the ```annealing_iterations``` allows governing the amount of refinement iterations performed during the main-loop of the heuristic;
 - the ```initial_temperature``` is the initial temperature of the matter;
 - the ```cooling_factor``` governs how quickly the temperature of the matter decreases during the annealing process.
 - the ```annealing_strength``` governs the strength of random perturbations during the annealing phase; specifically, the number of variables whose value is affected by perturbation.
 - the ```multiprocess_enabled``` parameter allows enabling/disabling process-based parallelism.
 
Now you can proceed solving the problem.
```
    optimizer.run(problem, termination_criterion)
```

Kindle note the ```termination_criterion``` paramerer. You can chose one of the following three:
  1. ```pyamosa.StopMinTemperature```: this is the classic termination criterion for simulated annealing: when the temperature of the matter is lower than the threshold, the algorithm is terminated. For instance, to run until the temperature goes below ```1e-7```, the termination criterion can be defined as follows:
  ```python
  pyamosa.StopMinTemperature(1e-7)
  ```
  2. ```pyamosa.StopMaxTime```: the termination can also be based on the time of the algorithm to be executed. For instance, to run an algorithm for 3 hours, 30 minutes, the termination can be defined as it follows (***note the initial hill-climbing is taken into account!***):
```python
termination = pyamosa.StopMaxTime("3:30")
```
  3. ```pyamosa.StopPhyWindow```: the most interesting stopping criterion is to use objective space change to decide whether to terminate the algorithm. Here, we resort to a simple and efficient procedure to determine whether to stop or not, described in [3]: it is based on the inter-generational distance (IGD), and it allows to stop the algorithm in case it cannot improve the Pareto front in a sequence of iterations. Say you want to stop if the algorithm is unable to improve in 10 iterations (meaning complete algorithm iterations, each of which consists of the number of annealing iterations as defined by the corresponding configuration parameter); then, the termination criterion can be defined as it follows
  ```python
  termination = pyamosa.StopPhyWindow(10)
  ```
  

At the end of execution, you can access the Pareto-front and the Pareto-set through the ```pareto_front()``` and 
```pareto_set()``` methods of the ```pyamosa.Optimizer``` class. You can also save the archive on CSV or JSON files, using the 
```archive_to_csv()``` or the ```archive_to_json()``` methods. The class also provides the ```plot_pareto()```, that plots
the Pareto-front resulting from the run.

## Constraint handling

Constraint handling is often neglected in frameworks but is indeed an essential aspect of optimization. Indeed, the 
returned optimum is always required to be feasible. 

In pyAMOSA, inequality constraints are always defined as 
<img src="https://render.githubusercontent.com/render/math?math=c(x)\le0"> constraints. Thus, constraint violation is
defined as follows: a solution is considered as feasible if all constraint violations are less than zero, while a 
solution is considered as infeasible if at least one constraint violation is larger than zero.

Suppose you whant to impose <img src="https://render.githubusercontent.com/render/math?math=-x^4\ge-2">. This has to be
converted in the less-or-equal form, thus <img src="https://render.githubusercontent.com/render/math?math=-x^4-2\le0">.

As objective-functions, constraints evaluation also takes place in the ```evaluate ``` method. You can fill the 
```out``` dictionary as follows.
```
class Problem(pyamosa.Optimizer.Problem):
    ...    
    def evaluate(self, x, out):
        ...
        out["g"] = [ ..., x**4 - 2, ... ]
```
For instance, consider the BNH test problem

*min*: 

<image src="https://latex.codecogs.com/svg.image?f_1(\textbf{x})=4x_1^2+4x_2^2"/>

<image src="https://latex.codecogs.com/svg.image?f_2(\textbf{x})=(x_1-5)^2+(x_2-5)^2"/>

*s.t.*

<image src="https://latex.codecogs.com/svg.image?(x_1-5)^2+x_2^2\le25"/>


<image src="https://latex.codecogs.com/svg.image?(x_1-8)^2+(x_2+3)^2\ge7.7"/>

<image src="https://latex.codecogs.com/svg.image?0 \le x_1 \le 5"/>

<image src="https://latex.codecogs.com/svg.image?0 \le x_2 \le 3"/>


It can be defined as it follows:
```python
import pyamosa, numpy as np


class BNH(pyamosa.Problem):
    n_var = 2

    def __init__(self):
        pyamosa.Problem.__init__(self, BNH.n_var, [pyamosa.Type.REAL] * BNH.n_var, [0.0] * BNH.n_var, [5.0, 3.0], 2, 2)

    def evaluate(self, x, out):
        f1 = 4 * x[0] ** 2 + 4 * x[1] ** 2
        f2 = (x[0] - 5) ** 2 + (x[1] - 5) ** 2
        g1 = (x[0] - 5) ** 2 + x[1] ** 2 - 25
        g2 = 7.7 - (x[0] - 5) ** 2 - (x[1] + 3) ** 2
        out["f"] = [f1, f2 ]
        out["g"] = [g1, g2]
```

## Improving a previous run

When calling the ```run()``` method, you can specify the path of a JSON file containing the archive resulting from a
previous run. Solutions from this archive will be considered as a starting point for a new run of the heuristic, 
possibly resulting in even better solutions.

# Understanding the log prints
When calling the ```run()``` method, during the annealing procedure, the optimizers will print several statistical information in a table format.
These are pretty useful for evaluating the effectiveness of the optimization process -- in specific whether it's going toward either convergence or diversity in the population, etc. -- so it's worth discoursing them.

  - *temp*.: it is the current temperature of the matter; refer to [1] for further details on its impact on the optimization process;
  - *eval*: it is the number of fitness-function evaluations;
  - *nds*: it is the number of non-dominated solutions the algorithm found until then;
  - *feas*: it is the number of **feasible** non-dominated solutions (i.e., those satisfying constraints) the algorithm found until then;
  - *cv min* and *cv avg*: minimum and average constraint violation, computed on unfeasible non-dominated solutions the algorithm found until then;
  - *D\** and *Dnad*: movement of the *ideal* and *nadir* idealized extreme points in the object-space; whether the algotithm is going toward convergence, they tend to be higher (the Pareto front is moving a lot!); see [3] for further details;
  - *phi*: the *intergenerational distance index*, computed on candidate solutions from the previous annealing iteration *P'* and candidate solutions resulting from the very last annealing iteration *P*; this allows monitoring; if the Pareto front is stationary, and can be improved neither by convergence nor by diversity, this value is close to zero; this metric is taken into consideration to determine the early termination condition; see [3] for further details;
  - *C(P', P)* and *C(P, P')*: the *coverage index* as defined in [4], computed on candidate solutions from the previous annealing iteration *P'* and candidate solutions resulting from the very last annealing iteration *P* and vice-versa, respectively; in general, *C(A,B)* is percentage the solutions in *B* that are dominated by at least one solution in *A*, where *A* and *B* are two Pareto fronts; therefore, *C(P, P')* should be alway greater than *C(P', P)* through the optimization process.

# References
1. Bandyopadhyay, S., Saha, S., Maulik, U., & Deb, K. (2008). A simulated annealing-based multiobjective optimization algorithm: AMOSA. IEEE transactions on evolutionary computation, 12(3), 269-283.
2. Deb, K. (2001). Multiobjective Optimization Using Evolutionary Algorithms. New York: Wiley, 2001
3. Blank, Julian, and Kalyanmoy Deb. "A running performance metric and termination criterion for evaluating evolutionary multi-and many-objective optimization algorithms." 2020 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2020.
4. Zitzler, E., e L. Thiele. "Multiobjective evolutionary algorithms: a comparative case study and the strength Pareto approach". IEEE Transactions on Evolutionary Computation 3, fasc. 4 (novembre 1999).
