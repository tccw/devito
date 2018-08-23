import sympy
import numpy as np

from devito.symbolics.search import retrieve_functions
from devito.symbolics.extended_sympy import FrozenExpr

__all__ = ['Differentiable']


class Differentiable(FrozenExpr):
    """
    This class represents Devito differentiable objects such as
    sum of functions, product of function or FD approximation and
    provides FD shortcuts for such expressions
    """
    _op_priority = 100.0
    is_Function = True
    is_Indexed = False

    def __new__(cls, *args, **kwargs):
        return sympy.Expr.__new__(cls, *args)

    def __init__(self, expr, **kwargs):
        from devito.finite_differences.finite_difference import generate_fd_functions
        self.expr = expr.expr if isinstance(expr, Differentiable) else expr
        self.indices = self._indices()
        # Generate FD shortcuts for expression or copy from input
        if isinstance(expr, Differentiable):
            self.fd = expr.fd
            # Recover the list of possible FD shortcuts
            self.dtype = expr.dtype
            self.space_order = expr.space_order
            self.time_order = expr.time_order
            self.staggered = expr.staggered
        else:
            # Recover the list of possible FD shortcuts
            self.dtype = self._dtype()
            self.space_order = self._space_order()
            self.time_order = self._space_order()
            self.staggered = self._staggered()
            self.fd = kwargs.get('fd', generate_fd_functions(self))

        for d in self.fd:
            setattr(self.__class__, d[1], property(d[0], d[1]))
        self.derivatives = tuple(d[1] for d in self.fd)

    def __add__(self, other):
        if isinstance(other, Differentiable):
            return Differentiable(sympy.Add(*[self.expr, other.expr], evaluate=False),
                                  fd=self.fd)
        else:
            return Differentiable(sympy.Add(*[self.expr, other], evaluate=False),
                                  fd=self.fd)

    __radd__ = __add__

    def __iadd__(self, other):
        if isinstance(other, Differentiable):
            self._expr = sympy.Add(*[self.expr, other.expr], evaluate=False)
        else:
            self._expr = sympy.Add(*[self.expr, other], evaluate=False)
        return self

    def __sub__(self, other):
        if isinstance(other, Differentiable):
            return Differentiable(sympy.Add(*[self.expr, -other.expr], evaluate=False),
                                  fd=self.fd)
        else:
            return Differentiable(sympy.Add(*[self.expr, -other]), fd=self.fd)

    def __rsub__(self, other):
        if isinstance(other, Differentiable):
            return Differentiable(sympy.Add(*[-self.expr, other.expr], evaluate=False),
                                  fd=self.fd)
        else:
            return Differentiable(sympy.Add(*[-self.expr, other], evaluate=False),
                                  fd=self.fd)

    def __isub__(self, other):
        if isinstance(other, Differentiable):
            self._expr = sympy.Add(*[self.expr, -other.expr], evaluate=False)
        else:
            self._expr = sympy.Add(*[self.expr, -other], evaluate=False)
        return self

    def __mul__(self, other):
        if isinstance(other, Differentiable):
            return Differentiable(sympy.Mul(*[self.expr, other.expr], evaluate=False),
                                  fd=self.fd)
        else:
            return Differentiable(sympy.Mul(*[self.expr, other], evaluate=False),
                                  fd=self.fd)

    __rmul__ = __mul__

    def __imul__(self, other):
        if isinstance(other, Differentiable):
            self._expr = sympy.Mul(*[self.expr, other.expr], evaluate=False)
        else:
            self._expr = sympy.Mul(*[self.expr, other], evaluate=False)
        return self

    def __truediv__(self, other):
        if isinstance(other, Differentiable):
            return Differentiable(self.expr * sympy.Pow(other.expr, -1, evaluate=False),
                                  fd=self.fd)
        else:
            return Differentiable(self.expr * sympy.Pow(other, -1, evaluate=False),
                                  fd=self.fd)

    def __rtruediv__(self, other):
        if isinstance(other, Differentiable):
            return Differentiable(other.expr * sympy.Pow(self.expr, -1, evaluate=False),
                                  fd=self.fd)
        else:
            return Differentiable(other * sympy.Pow(self.expr, -1, evaluate=False),
                                  fd=self.fd)

    def __neg__(self):
        return -1.0 * self

    def __pow__(self, exponent):
        return Differentiable(sympy.Pow(self, exponent, evaluate=False), fd=self.fd)

    def __eq__(self, other):
        if isinstance(other, Differentiable):
            return self.expr.args == other.expr.args
        elif isinstance(other, sympy.Expr):
            return self.expr.args == other.args
        else:
            return sympy.simplify(self.expr) == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def func(self, *args, **kwargs):
        kwargs.pop("evaluate", None)
        if hasattr(self.expr, 'base'):
            return self
        else:
            return Differentiable(self.expr.func(*args, **kwargs), fd=self.fd)

    def __str__(self):
        return self.expr.__str__()

    __repr__ = __str__

    @property
    def args(self):
        if hasattr(self.expr, 'base'):
            return (self.expr,)
        return self.expr.args

    @property
    def is_TimeFunction(self):
        """
        Check wether or not time is an index
        """
        return any(i.is_Time for i in self.indices)

    def _space_order(self):
        """
        Infer space_order from expression
        """
        func = list(retrieve_functions(self.expr))
        order = 100
        for i in func:
            order = min(order, getattr(i, 'space_order', order))

        return order

    def _time_order(self):
        """
        Infer space_order from expression
        """
        func = list(retrieve_functions(self.expr))
        order = 100
        for i in func:
            order = min(order, getattr(i, 'time_order', order))

        return order

    def _dtype(self):
        """
        Infer dtype for expression
        """
        func = list(retrieve_functions(self.expr))
        is_double = False
        for i in func:
            dtype_i = getattr(i, 'dtype', np.float32)
            is_double = dtype_i == np.float64 or is_double

        return np.float64 if is_double else np.float32

    def _indices(self):
        """
        Indices of the expression setup
        """
        func = list(retrieve_functions(self.expr))
        return tuple(set([d for i in func for d in getattr(i, 'indices', ())]))

    def _staggered(self):
        """
        Staggered grid setup
        """
        func = list(retrieve_functions(self.expr))
        func = [f for f in func if hasattr(f, 'staggered')]
        if any(s is None for f in func for s in f.staggered):
            staggered = tuple([None] * len(self.indices))
        else:
            staggered = tuple([0] * len(self.indices))
        return staggered

    def evalf(self, N=None):
        N = N or sympy.N(sympy.Float(1.0))
        if self.is_Number:
            return self.args[0]
        else:
            return self.func(*[i.evalf(N) for i in self.args], evaluate=False)

    def subs(self, subs):
        for k, v in subs.items():
            if isinstance(v, Differentiable):
                subs[k] = v.expr
        return self.expr.subs(subs)

    def __hash__(self):
        return hash(self.expr)


    @property
    def expr(self):
        return self._expr

    @expr.setter
    def expr(self, expr):
        self._expr = expr
