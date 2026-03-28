import re
import os
from typing import Dict, List, Any, Tuple

class LPReader:
    """
    Reader for the standard LP strictly formatted mathematical programming files (similar to CPLEX LP format).
    """
    
    def __init__(self):
        self.objective_type = "minimize"
        self.objective_coeffs: Dict[str, float] = {}
        self.constraints: List[Dict[str, Any]] = []
        self.bounds: Dict[str, Tuple[float, float]] = {}
        self.binary_vars: set = set()
        self.integer_vars: set = set()
        self.free_vars: set = set()
        self.all_vars: set = set()

    def read(self, filepath: str) -> Dict[str, Any]:
        """Reads an LP file and converts it into the internal dictionary format."""
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"LP file not found: {filepath}")

        with open(filepath, 'r') as f:
            lines = f.readlines()

        self._parse_lines(lines)
        return self._build_problem_dict()

    def _parse_lines(self, lines: List[str]):
        """Parses lines into internal components."""
        section = None
        current_constraint_name = ""
        current_expression = ""

        # Regular expressions
        obj_sense_re = re.compile(r'^(maximize|minimize|max|min)\s*$', re.IGNORECASE)
        subj_to_re = re.compile(r'^(subject to|such that|st|s\.t\.)\s*$', re.IGNORECASE)
        bounds_re = re.compile(r'^(bounds)\s*$', re.IGNORECASE)
        binaries_re = re.compile(r'^(binaries|binary|bin)\s*$', re.IGNORECASE)
        generals_re = re.compile(r'^(generals|general|gen|integer)\s*$', re.IGNORECASE)
        free_re = re.compile(r'^(free)\s*$', re.IGNORECASE)
        end_re = re.compile(r'^(end)\s*$', re.IGNORECASE)
        
        # Constraint match: e.g., "c1: 2 x + 3 y <= 5"
        constraint_sense_re = re.compile(r'(<=|>=|=|<|>)')

        def process_accumulated_expr(expr: str, is_objective: bool = False, constr_name: str = ""):
            if not expr.strip(): return
            
            # Split into LHS and RHS based on sense
            if not is_objective:
                match = constraint_sense_re.search(expr)
                if not match:
                    # Ignore invalid expression
                    return
                sense = match.group(1)
                lhs_str = expr[:match.start()].strip()
                rhs_str = expr[match.end():].strip()
                
                # Standardize sense
                if sense == '<': sense = '<='
                if sense == '>': sense = '>='
                
                parsed_lhs = self._parse_expression(lhs_str)
                rhs_value = float(rhs_str) if rhs_str else 0.0
                
                self.constraints.append({
                    'name': constr_name if constr_name else f"c{len(self.constraints)+1}",
                    'coeffs': parsed_lhs,
                    'type': sense,
                    'rhs': rhs_value
                })
            else:
                self.objective_coeffs = self._parse_expression(expr)

        for line in lines:
            line = line.strip()
            if not line or line.startswith('\\'): # Ignore comments
                continue
                
            # Check for section changes
            if obj_sense_re.match(line):
                process_accumulated_expr(current_expression, is_objective=(section in ['obj_max', 'obj_min']), constr_name=current_constraint_name)
                current_expression = ""
                sense = line.lower()
                if sense.startswith('max'):
                    section = 'obj_max'
                    self.objective_type = "maximize"
                else:
                    section = 'obj_min'
                    self.objective_type = "minimize"
                continue
            elif subj_to_re.match(line):
                process_accumulated_expr(current_expression, is_objective=(section in ['obj_max', 'obj_min']), constr_name=current_constraint_name)
                current_expression = ""
                section = 'constraints'
                continue
            elif bounds_re.match(line):
                process_accumulated_expr(current_expression, is_objective=False, constr_name=current_constraint_name)
                current_expression = ""
                section = 'bounds'
                continue
            elif binaries_re.match(line):
                section = 'binaries'
                continue
            elif generals_re.match(line):
                section = 'generals'
                continue
            elif free_re.match(line):
                section = 'free'
                continue
            elif end_re.match(line):
                break

            # Parse content based on current section
            if section in ['obj_max', 'obj_min']:
                # The objective might have a name like "obj: 2x + 3y"
                if ':' in line and not current_expression:
                    parts = line.split(':', 1)
                    current_expression += " " + parts[1].strip()
                else:
                    current_expression += " " + line
                    
            elif section == 'constraints':
                # Check if new constraint starts (has ':')
                # But handle edge cases where ':' is part of variable name (rare) - usually it's "name: "
                if ':' in line and not constraint_sense_re.search(line.split(':')[0]):
                    # Process previous constraint
                    process_accumulated_expr(current_expression, is_objective=False, constr_name=current_constraint_name)
                    
                    parts = line.split(':', 1)
                    current_constraint_name = parts[0].strip()
                    current_expression = parts[1].strip()
                else:
                    # Append to current constraint
                    # Check if this line actually contains a sense, if so and we don't have a name, 
                    # it might be a single-line constraint without name
                    current_expression += " " + line

            elif section == 'bounds':
                self._parse_bound_line(line)
            
            elif section == 'binaries':
                vars_in_line = line.split()
                for v in vars_in_line:
                    self.binary_vars.add(v)
                    self.all_vars.add(v)
                    
            elif section == 'generals':
                vars_in_line = line.split()
                for v in vars_in_line:
                    self.integer_vars.add(v)
                    self.all_vars.add(v)

            elif section == 'free':
                vars_in_line = line.split()
                for v in vars_in_line:
                    self.free_vars.add(v)
                    self.all_vars.add(v)

        # Process the last dangling expression if any
        if current_expression:
            process_accumulated_expr(current_expression, is_objective=(section in ['obj_max', 'obj_min']), constr_name=current_constraint_name)

    def _parse_expression(self, expr_str: str) -> Dict[str, float]:
        """Parses a linear expression string into a dictionary of {var_name: coefficient}."""
        # Replace '- ' with ' -', '+ ' with ' +' to normalize signs
        expr_str = expr_str.replace('- ', ' -').replace('+ ', ' +')
        if not expr_str.startswith('+') and not expr_str.startswith('-'):
            expr_str = '+' + expr_str
            
        # Add space before signs to ensure proper splitting
        expr_str = expr_str.replace('-', ' -').replace('+', ' +')
        
        terms = expr_str.split()
        parsed_coeffs = {}
        
        current_sign = 1.0
        current_coeff = None
        
        i = 0
        while i < len(terms):
            token = terms[i]
            
            # Extract sign and optional coefficient stuck to variable (e.g., "-2x" or "+x" or "- 2 x" or "-2 x")
            if token == '+':
                current_sign = 1.0
                i += 1
                continue
            elif token == '-':
                current_sign = -1.0
                i += 1
                continue
                
            # Match token to see if it's a number, a variable, or both (e.g., "2x")
            match = re.match(r'^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)(.*)$', token)
            
            if match and match.group(1) != token:
                # It has both number and string (e.g. 2.5x)
                coeff_str = match.group(1)
                var_name = match.group(2)
                
                coeff = float(coeff_str)
                parsed_coeffs[var_name] = parsed_coeffs.get(var_name, 0.0) + (current_sign * coeff)
                self.all_vars.add(var_name)
                current_sign = 1.0 # reset Default
                
            elif match and match.group(1) == token:
                # It's just a number. Next token must be the variable
                current_coeff = current_sign * float(token)
                if i + 1 < len(terms) and not terms[i+1] in ['+', '-']:
                    var_name = terms[i+1]
                    parsed_coeffs[var_name] = parsed_coeffs.get(var_name, 0.0) + current_coeff
                    self.all_vars.add(var_name)
                    i += 1 # skip variable
                current_sign = 1.0
                current_coeff = None
            else:
                # It's just a variable (implicit coefficient 1.0)
                # Handle cases like "-x" or "+x"
                if token.startswith('-'):
                    var_name = token[1:]
                    coeff = -1.0
                elif token.startswith('+'):
                    var_name = token[1:]
                    coeff = 1.0
                else:
                    var_name = token
                    coeff = 1.0
                    
                coeff *= current_sign
                parsed_coeffs[var_name] = parsed_coeffs.get(var_name, 0.0) + coeff
                self.all_vars.add(var_name)
                current_sign = 1.0
                
            i += 1
            
        return parsed_coeffs

    def _parse_bound_line(self, line: str):
        """Parses a bounds line like: 0 <= x1 <= 100 or x2 free."""
        line = line.lower()
        
        # Check for 'free'
        if ' free' in line:
            var_name = line.replace(' free', '').strip()
            self.free_vars.add(var_name)
            self.all_vars.add(var_name)
            return

        # Parse inequalities using regex to extract numbers and variables
        # Match cases: "num <= var", "var <= num", "num <= var <= num"
        tokens = line.replace('<=', ' <= ').replace('>=', ' >= ').replace('=', ' = ').split()
        
        if len(tokens) == 3:
            # e.g.: x <= 10 or 10 >= x
            if tokens[1] in ['<=', '=', '>=']:
                try:
                    val = float(tokens[2])
                    var = tokens[0]
                    op = tokens[1]
                except ValueError:
                    val = float(tokens[0])
                    var = tokens[2]
                    op = '<=' if tokens[1] == '>=' else ('>=' if tokens[1] == '<=' else '=')
                
                self.all_vars.add(var)
                current_low, current_high = self.bounds.get(var, (0.0, float('inf')))
                
                if op == '<=':
                    self.bounds[var] = (current_low, val)
                elif op == '>=':
                    self.bounds[var] = (val, current_high)
                elif op == '=':
                    self.bounds[var] = (val, val)
                    
        elif len(tokens) == 5:
            # e.g.: 0 <= x <= 10
            if tokens[1] == '<=' and tokens[3] == '<=':
                try:
                    low = float(tokens[0])
                    var = tokens[2]
                    high = float(tokens[4])
                    self.all_vars.add(var)
                    self.bounds[var] = (low, high)
                except ValueError:
                    pass

    def _build_problem_dict(self) -> Dict[str, Any]:
        """Converts internal representations into the solver's standard dictionary format."""
        
        # Sort variables for deterministic ordering
        var_names = sorted(list(self.all_vars))
        var_idx = {name: i for i, name in enumerate(var_names)}
        
        n_vars = len(var_names)
        
        # Build Objective
        obj_coeffs = [0.0] * n_vars
        for var, coeff in self.objective_coeffs.items():
            if var in var_idx:
                obj_coeffs[var_idx[var]] = coeff
                
        objective = {
            'type': self.objective_type,
            'coeffs': obj_coeffs
        }
        
        # Build Constraints
        formatted_constraints = []
        for c in self.constraints:
            c_coeffs = [0.0] * n_vars
            for var, coeff in c['coeffs'].items():
                if var in var_idx:
                    c_coeffs[var_idx[var]] = coeff
                    
            formatted_constraints.append({
                'name': c['name'],
                'type': c['type'],
                'coeffs': c_coeffs,
                'rhs': c['rhs']
            })
            
        # Build Variables
        variables = []
        for var in var_names:
            var_dict = {'name': var}
            
            # Determine type
            if var in self.binary_vars:
                var_dict['type'] = 'binary'
                var_dict['bounds'] = [0, 1]
            elif var in self.free_vars:
                var_dict['type'] = 'free'
                if var in self.bounds:
                    var_dict['bounds'] = [self.bounds[var][0], self.bounds[var][1]]
            else:
                var_dict['type'] = 'continuous'
                if var in self.integer_vars:
                    var_dict['type'] = 'integer' # Note: solver handles integer similar to continuous in base format unless passed to milp
                
                # Default lower bound is 0 in LP format unless specified
                low, high = self.bounds.get(var, (0.0, float('inf')))
                var_dict['bounds'] = [low, high]
                
            variables.append(var_dict)
            
        return {
            'objective': objective,
            'constraints': formatted_constraints,
            'variables': variables
        }

def read_lp(filepath: str) -> Dict[str, Any]:
    """Convenience function to read LP file."""
    reader = LPReader()
    return reader.read(filepath)
