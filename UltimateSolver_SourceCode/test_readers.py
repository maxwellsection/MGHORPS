from ultimate_solvers_unified import solve_from_file
import tempfile
import os

def test_mps():
    print("\n--- Testing MPS Reader & Presolve ---")
    mps_data = """NAME          TESTPROB
ROWS
 N  OBJ
 L  C1
 L  C2
COLUMNS
    X1        OBJ       1.0
    X1        C1        1.0
    X1        C2        2.0
    X2        OBJ       2.0
    X2        C1        1.0
    X2        C2        1.0
RHS
    RHS1      C1        4.0
    RHS1      C2        5.0
BOUNDS
 LO BND1      X1        0.0
 UP BND1      X1        2.0
 LO BND1      X2        0.0
ENDATA
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mps', delete=False) as f:
        f.write(mps_data)
        filepath = f.name
        
    try:
        res = solve_from_file(filepath, method='revised_simplex')
        print(f"Status: {res['status']}")
        print(f"Objective: {res['objective_value']}")
        print(f"Solution: {res['solution']}")
    finally:
        os.remove(filepath)

def test_lp():
    print("\n--- Testing LP Reader & Presolve ---")
    # A single variable constraint x3 <= 1 acts as a presolve test
    lp_data = """Maximize
 obj: 3 x1 + 2 x2 + x3
Subject To
 c1: x1 + x2 <= 4
 c2: 2 x1 + x2 <= 5
 c3: x3 <= 1
Bounds
 0 <= x1 <= 10
 0 <= x2
 free x3
End
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False) as f:
        f.write(lp_data)
        filepath = f.name
        
    try:
        res = solve_from_file(filepath, method='revised_simplex')
        print(f"Status: {res['status']}")
        print(f"Objective: {res['objective_value']}")
        print(f"Solution: {res['solution']}")
    finally:
        os.remove(filepath)

if __name__ == "__main__":
    test_mps()
    test_lp()
