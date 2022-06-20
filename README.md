# pyAMOSA 
pyAMOSA is a python implementation of the Archived Multi-Objective Simulated Annealing optimization heuristic [1].

## Defining and solving a problem

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
from AMOSA import *

class ZDT1(AMOSA.Problem):
    def __init__(self):
        n_var = 30
        AMOSA.Problem.__init__(self, 30, [AMOSA.Type.REAL] * 30, [0] * 30, [1] * 30, 2, 0)

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

    optimizer = AMOSA(...)
```

The ```AMOSA``` class allows setting a vast plethora of configuration parameters governing the behavior of the 
heuristic. You must do it by creating an ```AMOSAConfig``` object, as follows

```python
    config = AMOSAConfig()
    config.archive_hard_limit = 100
    config.archive_soft_limit = 200
    config.archive_gamma = 2
    config.clustering_max_iterations = 300
    config.hill_climbing_iterations = 500
    config.initial_temperature = 500
    config.final_temperature = 1e-7
    config.cooling_factor = 0.9
    config.annealing_iterations = 1000
    config.annealing_strength = 1
    config.early_terminator_window = 30
    config.multiprocess_enabled = True
```

 - the ```archive_hard_limit``` attribute allows setting the HL parameter of the heuristic, i.e., the hard limit on the archive size;
 - the ```archive_soft_limit``` attribute allows setting the SL parameter of the heuristic, i.e., the soft limit on the archive size;
 - the ```hill_climbing_iterations``` is the number of refinement iterations performed during the initial hill-climbing refinement;
 - the ```archive_gamma``` attribute allows governing the amount of initial candidate solutions that are generated duting the archive initialization; 
 - the ```clustering_max_iterations``` allows foverning the maximum iterations performed during k-means clustering;
 - the ```annealing_iterations``` allows governing the amount of refinement iterations performed during the main-loop of the heuristic;
 - the ```initial_temperature``` is the initial temperature of the matter;
 - the ```final_temperature``` is the final temperature of the matter;
 - the ```cooling_factor``` governs how quickly the temperature of the matter decreases during the annealing process.
 - the ```annealing_strength``` governs the strength of random perturbations during the annealing phase; specifically, the number of variables whose value is affected by perturbation.
 - the ```early_termination_window``` parameter allows the early-termination of the algorithm in case the Pareto-front does not improve through the specified amount of iterations. See [3] for more.
 - the ```multiprocess_enabled``` parameter allows enabling/disabling process-based parallelism.
 
Now you can proceed solving the problem.
```
    optimizer.run(problem)
```
At the end of execution, you can access the Pareto-front and the Pareto-set through the ```pareto_front()``` and 
```pareto_set()``` methods of the ```AMOSA``` class. You can also save the archive on CSV or JSON files, using the 
```archive_to_csv()``` or the ```archive_to_json``` method. The class also provides the ```plot_pareto()```, that plots
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
class Problem(AMOSA.Problem):
    ...    
    def evaluate(self, x, out):
        ...
        out["g"] = [ ..., x**4 - 2, ... ]
```

## Improving a previous run

When calling the ```run()``` method, you can specify the path of a JSON file containing the archive resulting from a
previous run. Solutions from this archive will be considered as a starting point for a new run of the heuristic, 
possibly resulting in even better solutions.

## A note on processing-based multitasking

The multitasking approach adopted in this implementation of the AMOSA algorithm is called *synchronous with intensive
solution exchanges*: threads starts from a random initial solution and they run independently, for a number of 
refinement iterations  being equal to the maximum number of annealing iterations, as specified by the related
configuration parameter. Once they have finished, before decreasing the temperature, threads report their final archive, 
and a reduce operation to obtain the non-dominated solutions is performed.

Nevertheless, in case your fitness-functions are already using multitasking, you can disable the latter in AMOSA by
simply setting ```config.multiprocess_enabled = False```.

## References
1. Bandyopadhyay, S., Saha, S., Maulik, U., & Deb, K. (2008). A simulated annealing-based multiobjective optimization algorithm: AMOSA. IEEE transactions on evolutionary computation, 12(3), 269-283.
2. Deb, K. (2001). Multiobjective Optimization Using Evolutionary Algorithms. New York: Wiley, 2001
3. Blank, Julian, and Kalyanmoy Deb. "A running performance metric and termination criterion for evaluating evolutionary multi-and many-objective optimization algorithms." 2020 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2020.
