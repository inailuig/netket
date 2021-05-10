
# Defining custom gradient preconditioners

NetKet calls _gradient preconditioner_ that class of techniques that transform the
gradient of the cost function in such a way to improve convergence properties before
passing it to an optimiser like :ref:`Sgd` or :ref:`Adam`.

Examples of _gradient preconditioners_ are the [Stochastic Reconfiguration](https://www.attaccalite.com/PhDThesis/html/node15.html) method  (also known as [Natural Gradient](https://wiseodd.github.io/techblog/2018/03/14/natural-gradient/), the Linear Method or second order Hessian-based optimisation.

We call those methods _gradient preconditioners_ because they take as input the gradient of the cost function (usually the energy) and some additional information, and output a transformed gradient.

NetKet comes out of the box with Stocastic Reconfiguration (:ref:`netket.optimiser.SR`), but it is possible to define your own method. 
If you follow the API outlined in this document, you will be able to use your own preconditioner within netket optimisation drivers without issues.

Keep in mind that writing your own optimisation loop only requires writing about 10  lines of code, and you are not forced to use NetKet's drivers!
We believe our design to be fairly modular and flexible, and it should be able to accomodate many use-cases, but there will be algorithms that are hard to experess within the boundaries of our API. 
Do not turn away from netket in those cases, but take all the pieces that you need and write your own optimisation loop!

## The preconditioner interface

Preconditioners are split into two objects for ease of use: 
 - an `object` which holds all informations necessary to apply the preconditioner, and is constructed from a variational state; 
 - a `function` taking as input the object above and the gradient of the cost function, and computing the resulting gradient.

To give a clear example: in the case of the Stochastic Reconfiguration (SR) method, if we call $ \vb{F} $ the gradient of the energy and $ S $ the Quantum Geometric Tensor (also known as SR matrix), we need to solve the system of equation $ S d\vb{w} = F $ to compute the resulting gradient.
The $ S $ matrix in this case is the `object` of the preconditioner, while the function is any linear solver such as `cholesky`, `jnp.linalg.solve` or iterative solvers such as `scipy.sparse.linalg.cg`.

As there are different ways to compute the $ S $ matrix, all with their different computational performance characteristics, and there are different solvers, we believe that this design makes the code more modular and easier to reason about.


### The preconditioner object API

When defining a preconditioner object you have two options: you can implement the bare API, which gives you maximum freedom but makes you responsible for all optimisations, or you can implement the `LinearOperator` interface, which constraints you a bit but will take care of a few performance optimisations.

#### Bare interface

The bare-minimum API a preconditioner `object` must implement:
 
	- It must be a class

	- There must be a function to build it from a variational state. This function will be called with the variational state as the first positional argument.  This function must not necessarily be a method of the class. 

	- This class must have a `solve(self, function, gradient, *, x0=None)` method taking as argument the gradient to be preconditioned and must not error if a keyword argument `x0` is passed to it. `x0` is the output of `solve` the last time it has been called, and might be ignored if not needed. `function` is the function computing the preconditioner.

You can subclass the abstract base class :ref`nk.optimizer.PreconditionerObject` to be sure that you are
implementing the correct interface, but you are not obliged to subclass it.

When you implement such an interface you are left with maximum flexibility, however you will be responsible for `jax.jit`ing all computational intensive methods (most likely `solve`).

```python

def MyObject(variational_state):
	stuff_a, stuff_b = compute_stuff(variational_state)
	return MyObjectT(stuff_a, stuff_b)

class MyObjectT:
	def __init__(self, a,b):
		# setup this object

	def solve(self, preconditioner_function, y, *, x0=None):
		# prepare
		...
		# compute
		return solve_fun(self, y, x0=x0)
```

Be warned that if you want to :ref:`jax.jit` compile the solve method, as it is usually computationally intensive, you must either specify how to flatten and unflatten to a PyTree your `MyObjectT`, or you should mark it as a `flax.struct.dataclass`, which is a frozen dataclass which does that automatically.
Since you cannot write the `__init__` method for a frozen dataclas, we usually define a constructor function as shown above. 

You might be asking why we have each object have a `solve` method, and pass it the preconditioner function isntead of the over way around. The reason for this is to invert the control: `preconditioner_function`s must obey a certain API, but even if they do, different objects might need to perform some different initialization to compute the precondition in a more efficient way. 
This architecture allows every object to run arbitrary logic before actually executing the preconditioner of choice.
Particular examples of this approach can be seen by looking at the implementation of :ref:`netket.optimizer.qgt.QGTJacobianDense` and :ref:`netket.optimizer.qgt.QGTJacobianPyTree`. 

#### LinearOperator interface

You can also subclass :ref:`netket.optimizer.LinearOperator`. 
A LinearOperator must be a [`flax` dataclass](https://flax.readthedocs.io/en/latest/flax.struct.html), which is an immutable
object (therefore after construction you cannot modify its attributes).

LinearOperators have several convenience methods, and they will act as matrices: you can right-multiply them by a PyTree 
vector or a dense vector. You can obtain their dense representation, and it will automatically jit-compile all computationally
intensive operations.

To implement the LinearOperator interface you should implement the following methods:

```python

def MyLinearOperator(variational_state):
	stuff_a, stuff_b = compute_stuff(variational_state)
	return MyLinearOperatorT(stuff_a, stuff_b)

@flax.struct.dataclass
class MyLinearOperatorT(nk.optimizer.LinearOperator):

	@jax.jit
	def __matmul__(self, y):
		# prepare
		...
		# compute
		return result 

	@jax.jit
	def to_dense(self):
		...
		return dense_matrix

	#optional
    @jax.jit
    def _solve(self, solve_fun, y: PyTree, *, x0: Optional[PyTree] = None) -> PyTree:
    	#...
    	return solution, solution_info
```

The bare minimum thing to implement is `__matmul__`, specifying how to multiply the linear operator by a pytree.
You can also define a custom `_solve` method if you have some computationally intensive setup code you wish to 
run before executing a solve function (that will call matmul repeatedly).
The `_solve` takes as first input the solve function, which is passed as a closure so it does not need to be marked
as static (even though it is).  The x0 is an optional argument which must be accepted but can be ignored, and it is the last previous solution to the linear system.
Optionally, one can also define the `to_dense` method.


### The preconditioner function API

The preconditioner function must have the following signature:

```python
def preconditioner_function(object, gradient, *, x0=None):
	#...
	return preconditioned_gradient, x0
```

The object that will be passed is the selected preconditioner object, previously constructed. 
The gradient is the gradient of the loss function to precondition.
x0 is an optional initial condition that might be ignored.

The gradient might be a PyTree version or a dense ravelling of the PyTree. The result of the function should be a preconditioned gradient with the same format.
Additional keyword argument can be present, and will in general be set through a closure or `functools.partial`, because this function will be called with the signature above.
If you have a peculiar preconditioner, you can assume that `preconditioner_function` will be called only from your `preconditioner object`, but in general it is good practice respecting the interface above so that different functions can work with different objects.




