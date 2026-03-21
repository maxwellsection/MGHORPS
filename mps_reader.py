import re
import os
from typing import Dict, List, Any, Tuple

class MPSReader:
    """
    Reader for the standard MPS mathematically programming format.
    """
    
    def __init__(self):
        self.objective_name = None
        self.objective_coeffs: Dict[str, float] = {}
        self.constraints: Dict[str, Dict[str, Any]] = {}
        self.bounds: Dict[str, Tuple[float, float]] = {}
        self.integer_vars: set = set()
        self.all_vars: set = set()
        self.ranges: Dict[str, float] = {}
        
    def read(self, filepath: str) -> Dict[str, Any]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"MPS file not found: {filepath}")
            
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        self._parse_lines(lines)
        return self._build_problem_dict()
        
    def _parse_lines(self, lines: List[str]):
        """Parses lines of an MPS file by sections."""
        section = None
        
        # MPS uses fixed column formatting originally, but modern parsers allow free format
        # as long as keywords are recognized. We'll implement a robust split-based parser.
        
        is_integer_marker = False
        
        for line in lines:
            line = line.rstrip() # keep leading spaces
            if not line or line.startswith('*'): 
                continue # comment or blank
                
            # Check for section headers (start at column 0)
            if not line.startswith(' ') and not line.startswith('\t'):
                header = line.split()[0].upper()
                if header == 'NAME':
                    section = 'NAME'
                elif header == 'ROWS':
                    section = 'ROWS'
                elif header == 'COLUMNS':
                    section = 'COLUMNS'
                elif header == 'RHS':
                    section = 'RHS'
                elif header == 'BOUNDS':
                    section = 'BOUNDS'
                elif header == 'RANGES':
                    section = 'RANGES'
                elif header == 'ENDATA':
                    break
                continue
                
            # Parse section content
            tokens = line.split()
            if not tokens: continue
            
            if section == 'ROWS':
                # Type and Name
                # N (Objective/Free), L (<=), G (>=), E (=)
                row_type = tokens[0].upper()
                row_name = tokens[1]
                
                if row_type == 'N':
                    if self.objective_name is None:
                        self.objective_name = row_name
                else:
                    sense_map = {'L': '<=', 'G': '>=', 'E': '='}
                    if row_type in sense_map:
                        self.constraints[row_name] = {
                            'name': row_name,
                            'type': sense_map[row_type],
                            'coeffs': {},
                            'rhs': 0.0 # Default RHS
                        }
                        
            elif section == 'COLUMNS':
                # Can be: ColName RowName1 Value1 [RowName2 Value2]
                if 'MARKER' in line and "'INTORG'" in line:
                    is_integer_marker = True
                    continue
                elif 'MARKER' in line and "'INTEND'" in line:
                    is_integer_marker = False
                    continue
                    
                col_name = tokens[0]
                self.all_vars.add(col_name)
                
                if is_integer_marker:
                    self.integer_vars.add(col_name)
                
                # Parse pairs of RowName and Value
                for i in range(1, len(tokens), 2):
                    if i + 1 < len(tokens):
                        row_name = tokens[i]
                        try:
                            val = float(tokens[i+1])
                        except ValueError:
                            continue
                            
                        if row_name == self.objective_name:
                            self.objective_coeffs[col_name] = self.objective_coeffs.get(col_name, 0.0) + val
                        elif row_name in self.constraints:
                            self.constraints[row_name]['coeffs'][col_name] = val
                            
            elif section == 'RHS':
                # RhsName RowName1 Value1 [RowName2 Value2]
                # Usually we ignore RhsName as there's only one RHS vector used
                for i in range(1, len(tokens), 2):
                    if i + 1 < len(tokens):
                        row_name = tokens[i]
                        try:
                            val = float(tokens[i+1])
                        except ValueError:
                            continue
                            
                        if row_name in self.constraints:
                            self.constraints[row_name]['rhs'] = val
                            
            elif section == 'BOUNDS':
                # BoundType BoundName ColName [Value]
                bound_type = tokens[0].upper()
                # Tokens: [1] is BoundName
                col_name = tokens[2]
                self.all_vars.add(col_name)
                
                val = 0.0
                if len(tokens) > 3:
                    try:
                        val = float(tokens[3])
                    except ValueError:
                        pass
                        
                current_low, current_high = self.bounds.get(col_name, (0.0, float('inf'))) # default non-negative
                
                if bound_type == 'LO': # Lower bound
                    self.bounds[col_name] = (val, current_high)
                elif bound_type == 'UP': # Upper bound
                    self.bounds[col_name] = (min(val, current_low) if current_low > 0 else 0.0, val) 
                    # Note: standard MPS behavior sets low=0 unless overridden
                elif bound_type == 'FX': # Fixed
                    self.bounds[col_name] = (val, val)
                elif bound_type == 'FR': # Free
                    self.bounds[col_name] = (-float('inf'), float('inf'))
                elif bound_type == 'MI': # Minus infinity bound (upper is 0.0 unless overriden)
                    self.bounds[col_name] = (-float('inf'), current_high)
                elif bound_type == 'PL': # Plus infinity 
                    self.bounds[col_name] = (current_low, float('inf'))
                elif bound_type == 'BV': # Binary
                    self.bounds[col_name] = (0.0, 1.0)
                    self.integer_vars.add(col_name)
                elif bound_type == 'LI': # Integer lower bound
                    self.bounds[col_name] = (val, current_high)
                    self.integer_vars.add(col_name)
                elif bound_type == 'UI': # Integer upper bound
                    self.bounds[col_name] = (current_low, val)
                    self.integer_vars.add(col_name)
                    
            elif section == 'RANGES':
                # RngName RowName1 Value1 [RowName2 Value2]
                for i in range(1, len(tokens), 2):
                    if i + 1 < len(tokens):
                        row_name = tokens[i]
                        try:
                            val = float(tokens[i+1])
                            self.ranges[row_name] = val
                        except ValueError:
                            continue

    def _build_problem_dict(self) -> Dict[str, Any]:
        """Convert internal structures into the solver dictionary format."""
        # Process ranges if any
        # A range constraint R bounds the row activity ax:
        # If row is <= rhs, ax <= rhs AND ax >= rhs - |r|
        # In our dict, we would need to split this into two constraints.
        formatted_constraints = []
        var_names = sorted(list(self.all_vars))
        var_idx = {name: i for i, name in enumerate(var_names)}
        n_vars = len(var_names)
        
        for c_name, c_data in self.constraints.items():
            coeffs = [0.0] * n_vars
            for var, val in c_data['coeffs'].items():
                if var in var_idx:
                    coeffs[var_idx[var]] = val
            
            sense = c_data['type']
            rhs = c_data['rhs']
            
            if c_name in self.ranges:
                r_val = self.ranges[c_name]
                if sense == '<=':
                    # ax <= rhs and ax >= rhs - |r|
                    formatted_constraints.append({'name': c_name + "_upper", 'type': '<=', 'coeffs': coeffs.copy(), 'rhs': rhs})
                    formatted_constraints.append({'name': c_name + "_lower", 'type': '>=', 'coeffs': coeffs.copy(), 'rhs': rhs - abs(r_val)})
                elif sense == '>=':
                    # ax >= rhs and ax <= rhs + |r|
                    formatted_constraints.append({'name': c_name + "_lower", 'type': '>=', 'coeffs': coeffs.copy(), 'rhs': rhs})
                    formatted_constraints.append({'name': c_name + "_upper", 'type': '<=', 'coeffs': coeffs.copy(), 'rhs': rhs + abs(r_val)})
                elif sense == '=':
                    # ax = rhs to ax between [rhs, rhs+|r|] if r > 0, else [rhs-|r|, rhs]
                    if r_val > 0:
                        formatted_constraints.append({'name': c_name + "_lower", 'type': '>=', 'coeffs': coeffs.copy(), 'rhs': rhs})
                        formatted_constraints.append({'name': c_name + "_upper", 'type': '<=', 'coeffs': coeffs.copy(), 'rhs': rhs + r_val})
                    else:
                        formatted_constraints.append({'name': c_name + "_lower", 'type': '>=', 'coeffs': coeffs.copy(), 'rhs': rhs + r_val})
                        formatted_constraints.append({'name': c_name + "_upper", 'type': '<=', 'coeffs': coeffs.copy(), 'rhs': rhs})
            else:
                formatted_constraints.append({'name': c_name, 'type': sense, 'coeffs': coeffs, 'rhs': rhs})
                
        # Objective
        obj_coeffs = [0.0] * n_vars
        for var, val in self.objective_coeffs.items():
            if var in var_idx:
                obj_coeffs[var_idx[var]] = val
                
        objective = {
            'type': 'minimize', # MPS format traditionally defaults to minimize
            'coeffs': obj_coeffs
        }
        
        # Variables
        variables = []
        for var in var_names:
            var_dict = {'name': var}
            low, high = self.bounds.get(var, (0.0, float('inf')))
            
            if var in self.integer_vars:
                if low == 0.0 and high == 1.0:
                    var_dict['type'] = 'binary'
                else:
                    var_dict['type'] = 'integer'
            else:
                # Determine generic type
                if low == -float('inf') and high == float('inf'):
                    var_dict['type'] = 'free'
                elif low == 0.0 and high == float('inf'):
                    var_dict['type'] = 'nonneg'
                else:
                    var_dict['type'] = 'continuous'
                    
            var_dict['bounds'] = [low, high]
            variables.append(var_dict)
            
        return {
            'objective': objective,
            'constraints': formatted_constraints,
            'variables': variables
        }

def read_mps(filepath: str) -> Dict[str, Any]:
    """Convenience function to read MPS file."""
    reader = MPSReader()
    return reader.read(filepath)
