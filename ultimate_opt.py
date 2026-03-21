import copy
from typing import Dict, List, Any, Union

# Try importing the underlying solver facade
try:
    from ultimate_solvers_unified import quick_solve_lp
    SOLVER_AVAILABLE = True
except ImportError:
    SOLVER_AVAILABLE = False


class Expression:
    """
    Base class representing a linear algebraic expression.
    Internally it acts like a computation graph node that stores coefficients.
    """
    def __init__(self, coeffs: Dict[str, float] = None, constant: float = 0.0):
        self.coeffs = coeffs if coeffs is not None else {}
        self.constant = float(constant)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Expression(copy.copy(self.coeffs), self.constant + other)
        elif isinstance(other, Expression):
            new_coeffs = copy.copy(self.coeffs)
            for k, v in other.coeffs.items():
                new_coeffs[k] = new_coeffs.get(k, 0.0) + v
            return Expression(new_coeffs, self.constant + other.constant)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Expression(copy.copy(self.coeffs), self.constant - other)
        elif isinstance(other, Expression):
            new_coeffs = copy.copy(self.coeffs)
            for k, v in other.coeffs.items():
                new_coeffs[k] = new_coeffs.get(k, 0.0) - v
            return Expression(new_coeffs, self.constant - other.constant)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            new_coeffs = {k: -v for k, v in self.coeffs.items()}
            return Expression(new_coeffs, other - self.constant)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new_coeffs = {k: v * other for k, v in self.coeffs.items()}
            return Expression(new_coeffs, self.constant * other)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            new_coeffs = {k: v / other for k, v in self.coeffs.items()}
            return Expression(new_coeffs, self.constant / other)
        return NotImplemented

    def __neg__(self):
        return self.__mul__(-1)

    # Comparison operators generate Constraint objects directly
    def __le__(self, other):
        return Constraint(self - other, '<=', 0)

    def __ge__(self, other):
        return Constraint(self - other, '>=', 0)

    def __eq__(self, other):
        return Constraint(self - other, '==', 0)

    def __repr__(self):
        terms = []
        for v, c in self.coeffs.items():
            if c != 0:
                terms.append(f"{c} * {v}")
        if self.constant != 0 or not terms:
            terms.append(str(self.constant))
        return " + ".join(terms)


class Variable(Expression):
    """
    Continuous Decision Variable. 
    Can be instantiated via standard variables: x = Variable("x", lower_bound=0)
    Follows PyTorch style node generation.
    """
    def __init__(self, name: str = None, lower_bound: float = 0.0, upper_bound: float = float('inf')):
        if name is not None:
            super().__init__({name: 1.0}, 0.0)
        else:
            super().__init__({}, 0.0)
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def set_name(self, name: str):
        self.name = name
        self.coeffs = {name: 1.0}

    def __repr__(self):
        return f"Variable({self.name}, bounds=[{self.lower_bound}, {self.upper_bound}])"


class Constraint:
    """
    Constraint class produced by expressions, mathematically: LHS (operator) RHS.
    """
    def __init__(self, expr: Expression, type_: str, rhs: float):
        # We process constraints into: sum(coeffs*x) <= rhs_val
        self.coeffs = expr.coeffs
        self.type = type_
        self.rhs = rhs - expr.constant
        
    def __repr__(self):
        terms = [f"{v}*{k}" for k, v in self.coeffs.items()]
        return f"Constraint({' + '.join(terms)} {self.type} {self.rhs})"


class Model:
    """
    Equivalent to torch.nn.Module, it provides a structured container to hold 
    parameters (Variable) and construct the computation graph (Constraints/Objective).
    """
    def __init__(self):
        self._variables: Dict[str, Variable] = {}
        self._constraints: List[Constraint] = []
        self._objective: Expression = None
        self._objective_type: str = 'maximize'

    def __setattr__(self, name, value):
        # Automatically register Variables like PyTorch registers parameters
        if isinstance(value, Variable):
            if value.name is None:
                value.set_name(name)
            self.__dict__['_variables'][value.name] = value
        super().__setattr__(name, value)

    def add_var(self, name: str, lower_bound: float = 0.0, upper_bound: float = float('inf')) -> Variable:
        """Helper to create and add a variable dynamically (scripting mode)."""
        v = Variable(name, lower_bound, upper_bound)
        self._variables[name] = v
        return v

    def maximize(self, expr: Expression):
        """DECLARATIVE LINGO STYLE: Set objective to Maximize form"""
        self._objective = expr if isinstance(expr, Expression) else Expression(constant=expr)
        self._objective_type = 'maximize'
        return self._objective

    def minimize(self, expr: Expression):
        """DECLARATIVE LINGO STYLE: Set objective to Minimize form"""
        self._objective = expr if isinstance(expr, Expression) else Expression(constant=expr)
        self._objective_type = 'minimize'
        return self._objective

    def subject_to(self, *constraints: Constraint):
        """DECLARATIVE LINGO STYLE: Add multiple constraints."""
        for c in constraints:
            if isinstance(c, Constraint):
                self._constraints.append(c)
            elif isinstance(c, list) or isinstance(c, tuple):
                for sub_c in c:
                    self._constraints.append(sub_c)

    def add(self, constraint: Constraint):
        self.subject_to(constraint)

    def forward(self):
        """
        PyTorch style lifecycle hook for models strictly inheriting `Model`.
        Users override this to systematically define their computation graph.
        """
        pass

    def compile(self) -> Dict[str, Any]:
        """
        Converts the dynamic object-oriented structure into the standard dict form 
        acceptable by ultimate_solvers_unified.
        """
        # Ensure model is constructed correctly if forward is presented
        self.forward()

        # Phase 1: Collect all known variables from the model attributes
        # Also discover any implicitly created variables from expressions
        all_var_names = set(self._variables.keys())

        if self._objective is not None:
            all_var_names.update(self._objective.coeffs.keys())
        
        for c in self._constraints:
            all_var_names.update(c.coeffs.keys())

        # Build variable mapping
        variables_list = []
        var_name_to_idx = {}
        for idx, name in enumerate(all_var_names):
            var_name_to_idx[name] = idx
            if name in self._variables:
                var = self._variables[name]
                if var.lower_bound != 0.0 or var.upper_bound != float('inf'):
                    variables_list.append({'name': name, 'type': 'continuous', 'bounds': [float(var.lower_bound), float(var.upper_bound)]})
                else:
                    variables_list.append({'name': name, 'type': 'nonneg'})
            else:
                # Implicitly discovered anonymous variable
                variables_list.append({'name': name, 'type': 'free'})

        n_vars = len(variables_list)

        # Build Objective
        obj_coeffs = [0.0] * n_vars
        if self._objective is not None:
            for v_name, coef in self._objective.coeffs.items():
                obj_coeffs[var_name_to_idx[v_name]] = coef

        standard_objective = {
            'type': self._objective_type,
            'coeffs': obj_coeffs,
            'constant': float(self._objective.constant) if self._objective else 0.0
        }

        # Build Constraints
        standard_constraints = []
        for c in self._constraints:
            c_coeffs = [0.0] * n_vars
            for v_name, coef in c.coeffs.items():
                c_coeffs[var_name_to_idx[v_name]] = coef

            # Map operator
            ctype = c.type
            if ctype == '==': ctype = '='
            
            standard_constraints.append({
                'type': ctype,
                'coeffs': c_coeffs,
                'rhs': float(c.rhs)
            })

        return {
            'objective': standard_objective,
            'constraints': standard_constraints,
            'variables': variables_list
        }


class Solver:
    """
    Execution Engine analogous to `Trainer` or wrapping `ultimate_solvers_unified.py`
    """
    def __init__(self, method: str = 'auto', use_gpu: bool = None, use_npu: bool = False, npu_cores: int = 2):
        self.method = method
        self.use_gpu = use_gpu
        self.use_npu = use_npu
        self.npu_cores = npu_cores

    def solve(self, model: Model) -> Dict[str, Any]:
        if not SOLVER_AVAILABLE:
            raise ImportError("Cannot find ultimate_solvers_unified module! Make sure it is in the PYTHONPATH.")
            
        compiled_data = model.compile()
        
        # Dispatch to the underlying engine
        result = quick_solve_lp(
            compiled_data['objective'],
            compiled_data['constraints'],
            compiled_data['variables'],
            use_gpu=self.use_gpu,
            use_npu=self.use_npu,
            npu_cores=self.npu_cores,
            method=self.method
        )
        return result
