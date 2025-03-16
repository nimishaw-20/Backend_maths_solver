from flask import Flask, request, jsonify
from sympy import symbols, Eq, solve, diff, integrate, pi, exp, sin, cos, tan, log, sqrt
from sympy.solvers.inequalities import solve_univariate_inequality
from sympy.parsing.sympy_parser import parse_expr
import re
from flask_cors import CORS
import json
import os
import logging
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Environment configuration
DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
PORT = int(os.environ.get("PORT", 5000))
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-key-change-in-production")

# Configure CORS with more restrictive settings
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

# Response caching
def cache_response(timeout=300):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            response = f(*args, **kwargs)
            if isinstance(response, tuple):
                response_obj, status_code = response
                if status_code == 200:
                    response_obj.headers['Cache-Control'] = f'public, max-age={timeout}'
                return response_obj, status_code
            else:
                response.headers['Cache-Control'] = f'public, max-age={timeout}'
                return response
        return decorated_function
    return decorator

# Define symbols
x, y, z, t, a, b, c, n = symbols('x y z t a b c n')

def safe_eval(expr_str, local_dict=None):
    """Safely evaluate a mathematical expression."""
    try:
        # Try to parse the expression with sympy's parser first (safer)
        return parse_expr(expr_str)
    except Exception as e:
        logger.warning(f"Parse_expr failed for '{expr_str}': {str(e)}")
        # If sympy parsing fails, try with restricted locals and globals
        if local_dict is None:
            local_dict = {
                "x": x, "y": y, "z": z, "t": t, "a": a, "b": b, "c": c, "n": n,
                "sin": sin, "cos": cos, "tan": tan, "exp": exp, 
                "log": log, "sqrt": sqrt, "pi": pi
            }
        try:
            # Use eval with empty __builtins__ to restrict operations
            return eval(expr_str, {"__builtins__": {}}, local_dict)
        except Exception as e:
            logger.error(f"Safe eval failed for '{expr_str}': {str(e)}")
            raise ValueError(f"Could not safely evaluate expression: {expr_str}")

# Health check endpoint
@app.route('/health')
def health():
    return jsonify({"status": "ok", "service": "Math Solver API"})

@app.route('/')
def home():
    return jsonify({"message": "Math Solver API is running", "version": "1.1"})

@app.route('/problem_types', methods=['GET'])
@cache_response(timeout=3600)  # Cache for an hour since this rarely changes
def get_problem_types():
    """Returns available problem types for the Flutter UI to display."""
    problem_types = {
        "types": [
            {"id": "linear", "name": "Linear Equation", "hasSubtypes": False},
            {"id": "quadratic", "name": "Quadratic Equation", "hasSubtypes": False},
            {"id": "system", "name": "System of Equations", "hasSubtypes": False},
            {"id": "inequality", "name": "Inequality", "hasSubtypes": False},
            {"id": "polynomial", "name": "Polynomial Equation", "hasSubtypes": False},
            {"id": "geometry", "name": "Geometry", "hasSubtypes": True},
            {"id": "differentiation", "name": "Differentiation", "hasSubtypes": False},
            {"id": "integration", "name": "Integration", "hasSubtypes": False},
            {"id": "trigonometry", "name": "Trigonometric Equations", "hasSubtypes": False},
            {"id": "limit", "name": "Limits", "hasSubtypes": False},
            {"id": "statistics", "name": "Statistics", "hasSubtypes": True}
        ],
        "subtypes": {
            "geometry": [
                {"id": "circle_area", "name": "Circle Area"},
                {"id": "circle_circumference", "name": "Circle Circumference"},
                {"id": "triangle_area", "name": "Triangle Area"},
                {"id": "rectangle_area", "name": "Rectangle Area"},
                {"id": "sphere_volume", "name": "Sphere Volume"}
            ],
            "statistics": [
                {"id": "mean", "name": "Mean (Average)"},
                {"id": "median", "name": "Median"},
                {"id": "mode", "name": "Mode"},
                {"id": "standard_deviation", "name": "Standard Deviation"},
                {"id": "variance", "name": "Variance"},
                {"id": "range", "name": "Range"}
            ]
        },
        "examples": {
            "linear": "2*x + 3 = 7",
            "quadratic": "x**2 + 5*x + 6 = 0",
            "system": "x + y = 10; 2*x - y = 5",
            "inequality": "x**2 - 4 < 0",
            "polynomial": "x**3 - 6*x**2 + 11*x - 6 = 0",
            "differentiation": "x**2 + 3*x + 2",
            "integration": "2*x + 3",
            "trigonometry": "sin(x) = 0.5",
            "limit": "limit(x, 0, (sin(x)/x))",
            "statistics": "data = [10, 20, 30, 40, 50]",
            "geometry": {
                "circle_area": "radius = 5",
                "circle_circumference": "radius = 5",
                "triangle_area": "base = 5; height = 8",
                "rectangle_area": "length = 5; width = 10",
                "sphere_volume": "radius = 3"
            }
        }
    }
    return jsonify(problem_types)

@app.route('/solve', methods=['POST'])
def solve_problem():
    try:
        # Validate request has JSON
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.json
        
        # Validate required fields
        if 'type' not in data:
            return jsonify({"error": "Problem type is required"}), 400
        if 'expression' not in data:
            return jsonify({"error": "Expression is required"}), 400
            
        problem_type = data.get('type')
        expression = data.get('expression')
        sub_type = data.get('subType', '')

        steps = []
        solution = None
        graph_data = None

        if problem_type == "linear":
            lhs, rhs = expression.split('=')
            lhs_expr = safe_eval(lhs.strip())
            rhs_expr = safe_eval(rhs.strip())
            eq = Eq(lhs_expr, rhs_expr)
            
            steps.append(f"Formulate the equation: {lhs} = {rhs}")
            steps.append(f"Move all terms to the left side: {lhs} - ({rhs}) = 0")
            simplified = lhs_expr - rhs_expr
            steps.append(f"Simplified equation: {simplified} = 0")
            
            # Check which variable to solve for
            var_to_solve = None
            for var in [x, y, z]:
                if var in simplified.free_symbols:
                    var_to_solve = var
                    break
            
            if var_to_solve:
                steps.append(f"Solve for {var_to_solve}")
                solution = solve(eq, var_to_solve)
                if solution:
                    steps.append(f"Solution: {var_to_solve} = {solution[0]}")
                    solution = f"{var_to_solve} = {solution[0]}"
                else:
                    steps.append("No solution found")
                    solution = "No solution"
            else:
                steps.append("No variable found to solve for")
                solution = "No variable found"

        elif problem_type == "quadratic":
            lhs, rhs = expression.split('=')
            lhs_expr = safe_eval(lhs.strip())
            rhs_expr = safe_eval(rhs.strip())
            eq = Eq(lhs_expr, rhs_expr)
            
            steps.append(f"Write the equation: {lhs} = {rhs}")
            steps.append(f"Move all terms to the left side: {lhs} - ({rhs}) = 0")
            
            # Move everything to the left side
            expr = lhs_expr - rhs_expr
            steps.append(f"Standard form: {expr} = 0")
            
            # Try to identify the variable
            var_to_solve = None
            for var in [x, y, z]:
                if var in expr.free_symbols:
                    var_to_solve = var
                    break
                    
            if not var_to_solve:
                return jsonify({"error": "No variable found in equation"}), 400
                
            steps.append(f"Identify this as a quadratic equation in {var_to_solve}")
            steps.append("Use the quadratic formula: x = [-b ± √(b² - 4ac)] / 2a")
            
            solutions = solve(eq, var_to_solve)
            
            if len(solutions) == 2:
                steps.append(f"Calculate the discriminant and find two roots")
                solution = f"{var_to_solve} = {solutions[0]} or {var_to_solve} = {solutions[1]}"
            elif len(solutions) == 1:
                steps.append(f"The discriminant is zero, giving a repeated root")
                solution = f"{var_to_solve} = {solutions[0]}"
            else:
                steps.append(f"No real solutions found")
                solution = "No real solutions"
                
            # Prepare graph data for Flutter
            coeffs = Eq(expr, 0).as_poly().all_coeffs()
            x_vals = list(range(-10, 11))
            y_vals = [sum(c * (x_val ** i) for i, c in enumerate(reversed(coeffs))) for x_val in x_vals]
            graph_data = {
                "type": "polynomial",
                "points": [{"x": x_val, "y": float(y_val)} for x_val, y_val in zip(x_vals, y_vals)],
                "roots": [float(sol) for sol in solutions if sol.is_real]
            }

        elif problem_type == "system":
            equations = expression.split(';')
            system_eqs = []
            
            steps.append("Write the system of equations:")
            
            # Parse each equation and convert to SymPy equation objects
            for i, eq_str in enumerate(equations):
                if '=' not in eq_str:
                    return jsonify({"error": f"Equation {i+1} does not contain an equals sign"}), 400
                    
                lhs, rhs = eq_str.split('=')
                lhs_expr = safe_eval(lhs.strip())
                rhs_expr = safe_eval(rhs.strip())
                eq = Eq(lhs_expr, rhs_expr)
                system_eqs.append(eq)
                steps.append(f"Equation {i+1}: {lhs} = {rhs}")
            
            # Collect variables in the system
            variables = set()
            for eq in system_eqs:
                variables.update(eq.free_symbols)
            
            variables = list(variables)
            
            if len(variables) != len(system_eqs):
                steps.append(f"Note: The system has {len(variables)} variables and {len(system_eqs)} equations.")
            
            steps.append("Solving the system using substitution method.")
            
            try:
                solution_dict = solve(system_eqs, variables)
                
                if solution_dict:
                    solution_parts = []
                    for var, val in solution_dict.items():
                        solution_parts.append(f"{var} = {val}")
                    solution = ", ".join(solution_parts)
                    steps.append("Found solution: " + solution)
                else:
                    steps.append("No solution found for the system.")
                    solution = "No solution"
            except Exception as e:
                logger.warning(f"Error solving system: {str(e)}")
                steps.append(f"Error solving system: {str(e)}")
                solution = "Could not solve system"

        elif problem_type == "inequality":
            try:
                # Try to parse directly as an inequality
                ineq_expr = eval(expression, {"__builtins__": {}}, 
                                {"x": x, "y": y, "z": z, "<": lambda a, b: a < b, 
                                 ">": lambda a, b: a > b, "<=": lambda a, b: a <= b, 
                                 ">=": lambda a, b: a >= b})
                
                steps.append(f"Inequality: {expression}")
                
                # Find the variable
                variables = ineq_expr.free_symbols
                if not variables:
                    return jsonify({"error": "No variable found in inequality"}), 400
                
                var = list(variables)[0]
                steps.append(f"Solve for {var}")
                
                solution = solve_univariate_inequality(ineq_expr, var)
                steps.append(f"Solution: {solution}")
                
            except Exception:
                # If direct parsing fails, try to split by inequality operators
                for op in ['<=', '>=', '<', '>']:
                    if op in expression:
                        parts = expression.split(op)
                        if len(parts) == 2:
                            lhs = safe_eval(parts[0].strip())
                            rhs = safe_eval(parts[1].strip())
                            
                            steps.append(f"Inequality: {parts[0]} {op} {parts[1]}")
                            
                            # Find the variable
                            variables = set()
                            variables.update(lhs.free_symbols)
                            variables.update(rhs.free_symbols)
                            
                            if not variables:
                                return jsonify({"error": "No variable found in inequality"}), 400
                            
                            var = list(variables)[0]
                            steps.append(f"Solve for {var}")
                            
                            # Create the inequality expression
                            if op == '<':
                                ineq_expr = lhs < rhs
                            elif op == '>':
                                ineq_expr = lhs > rhs
                            elif op == '<=':
                                ineq_expr = lhs <= rhs
                            elif op == '>=':
                                ineq_expr = lhs >= rhs
                            
                            solution = solve_univariate_inequality(ineq_expr, var)
                            steps.append(f"Solution: {solution}")
                            break
                else:
                    return jsonify({"error": "Invalid inequality format"}), 400

        elif problem_type == "polynomial":
            if '=' not in expression:
                return jsonify({"error": "Equation must contain an equals sign"}), 400
                
            lhs, rhs = expression.split('=')
            lhs_expr = safe_eval(lhs.strip())
            rhs_expr = safe_eval(rhs.strip())
            eq = Eq(lhs_expr, rhs_expr)
            
            steps.append(f"Write the polynomial equation: {lhs} = {rhs}")
            steps.append(f"Move all terms to the left side: {lhs} - ({rhs}) = 0")
            
            # Move everything to the left side
            expr = lhs_expr - rhs_expr
            steps.append(f"Standard form: {expr} = 0")
            
            # Try to identify the variable
            var_to_solve = None
            for var in [x, y, z]:
                if var in expr.free_symbols:
                    var_to_solve = var
                    break
                    
            if not var_to_solve:
                return jsonify({"error": "No variable found in equation"}), 400
                
            steps.append(f"Find the roots of the polynomial in {var_to_solve}")
            
            solutions = solve(eq, var_to_solve)
            
            if solutions:
                solution_strs = [f"{var_to_solve} = {sol}" for sol in solutions]
                steps.append(f"Found {len(solutions)} solution(s)")
                solution = " or ".join(solution_strs)
            else:
                steps.append(f"No real solutions found")
                solution = "No real solutions"
                
            # Prepare graph data for Flutter
            try:
                coeffs = Eq(expr, 0).as_poly().all_coeffs()
                x_vals = list(range(-10, 11))
                y_vals = [sum(c * (x_val ** i) for i, c in enumerate(reversed(coeffs))) for x_val in x_vals]
                graph_data = {
                    "type": "polynomial",
                    "points": [{"x": x_val, "y": float(y_val)} for x_val, y_val in zip(x_vals, y_vals)],
                    "roots": [float(sol) for sol in solutions if sol.is_real]
                }
            except Exception as e:
                logger.warning(f"Error creating graph data: {str(e)}")
                # If graphing fails, continue without it
                pass

        elif problem_type == "geometry":
            steps.append(f"Geometry problem type: {sub_type}")
            
            try:
                # Parse the expression to extract values
                params = {}
                for part in expression.split(';'):
                    if '=' in part:
                        key, value = part.split('=')
                        params[key.strip()] = float(value.strip())
                
                if sub_type == "circle_area":
                    if 'radius' in params:
                        r = params['radius']
                        steps.append(f"Circle with radius = {r}")
                        steps.append(f"Area of a circle: A = π × r²")
                        area = pi * r**2
                        steps.append(f"A = π × {r}² = {area}")
                        solution = f"Area = {area}"
                        graph_data = {
                            "type": "circle",
                            "radius": float(r)
                        }
                        
                elif sub_type == "circle_circumference":
                    if 'radius' in params:
                        r = params['radius']
                        steps.append(f"Circle with radius = {r}")
                        steps.append(f"Circumference of a circle: C = 2π × r")
                        circumference = 2 * pi * r
                        steps.append(f"C = 2π × {r} = {circumference}")
                        solution = f"Circumference = {circumference}"
                        graph_data = {
                            "type": "circle",
                            "radius": float(r)
                        }
                        
                elif sub_type == "triangle_area":
                    if 'base' in params and 'height' in params:
                        b = params['base']
                        h = params['height']
                        steps.append(f"Triangle with base = {b} and height = {h}")
                        steps.append(f"Area of a triangle: A = (b × h) / 2")
                        area = (b * h) / 2
                        steps.append(f"A = ({b} × {h}) / 2 = {area}")
                        solution = f"Area = {area}"
                        graph_data = {
                            "type": "triangle",
                            "base": float(b),
                            "height": float(h)
                        }
                        
                elif sub_type == "rectangle_area":
                    if 'length' in params and 'width' in params:
                        l = params['length']
                        w = params['width']
                        steps.append(f"Rectangle with length = {l} and width = {w}")
                        steps.append(f"Area of a rectangle: A = l × w")
                        area = l * w
                        steps.append(f"A = {l} × {w} = {area}")
                        solution = f"Area = {area}"
                        graph_data = {
                            "type": "rectangle",
                            "length": float(l),
                            "width": float(w)
                        }
                        
                elif sub_type == "sphere_volume":
                    if 'radius' in params:
                        r = params['radius']
                        steps.append(f"Sphere with radius = {r}")
                        steps.append(f"Volume of a sphere: V = (4/3) × π × r³")
                        volume = (4/3) * pi * r**3
                        steps.append(f"V = (4/3) × π × {r}³ = {volume}")
                        solution = f"Volume = {volume}"
                        graph_data = {
                            "type": "sphere",
                            "radius": float(r)
                        }
                        
                else:
                    return jsonify({"error": "Unsupported geometry sub-type"}), 400
                    
            except Exception as e:
                logger.error(f"Error in geometry calculation: {str(e)}")
                return jsonify({"error": f"Error in geometry calculation: {str(e)}"}), 400

        elif problem_type == "differentiation":
            steps.append(f"Expression to differentiate: {expression}")
            steps.append("Find the derivative with respect to x")
            
            try:
                expr = safe_eval(expression)
                derivative = diff(expr, x)
                steps.append(f"Apply the rules of differentiation")
                steps.append(f"The derivative is: {derivative}")
                solution = f"f'(x) = {derivative}"
                
                # Prepare graph data for Flutter
                try:
                    x_vals = list(range(-5, 6))
                    y_vals_orig = [float(expr.subs(x, x_val)) for x_val in x_vals]
                    y_vals_deriv = [float(derivative.subs(x, x_val)) for x_val in x_vals]
                    
                    graph_data = {
                        "type": "function_comparison",
                        "function": [{"x": x_val, "y": y_val} for x_val, y_val in zip(x_vals, y_vals_orig)],
                        "derivative": [{"x": x_val, "y": y_val} for x_val, y_val in zip(x_vals, y_vals_deriv)]
                    }
                except Exception as e:
                    logger.warning(f"Error generating graph data: {str(e)}")
                    # If graphing fails, continue without it
                    pass
                    
            except Exception as e:
                logger.error(f"Error in differentiation: {str(e)}")
                return jsonify({"error": f"Error in differentiation: {str(e)}"}), 400

        elif problem_type == "integration":
            steps.append(f"Expression to integrate: {expression}")
            steps.append("Find the indefinite integral with respect to x")
            
            try:
                expr = safe_eval(expression)
                integral = integrate(expr, x)
                steps.append(f"Apply the rules of integration")
                steps.append(f"The indefinite integral is: {integral} + C")
                solution = f"∫{expression} dx = {integral} + C"
                
                # Prepare graph data for Flutter
                try:
                    x_vals = list(range(-5, 6))
                    y_vals_orig = [float(expr.subs(x, x_val)) for x_val in x_vals]
                    y_vals_integ = [float(integral.subs(x, x_val)) for x_val in x_vals]
                    
                    graph_data = {
                        "type": "function_comparison",
                        "function": [{"x": x_val, "y": y_val} for x_val, y_val in zip(x_vals, y_vals_orig)],
                        "integral": [{"x": x_val, "y": y_val} for x_val, y_val in zip(x_vals, y_vals_integ)]
                    }
                except Exception as e:
                    logger.warning(f"Error generating graph data: {str(e)}")
                    # If graphing fails, continue without it
                    pass
                    
            except Exception as e:
                logger.error(f"Error in integration: {str(e)}")
                return jsonify({"error": f"Error in integration: {str(e)}"}), 400

        elif problem_type == "trigonometry":
            if '=' not in expression:
                return jsonify({"error": "Trigonometric equation must contain an equals sign"}), 400
                
            lhs, rhs = expression.split('=')
            lhs_expr = safe_eval(lhs.strip())
            rhs_expr = safe_eval(rhs.strip())
            eq = Eq(lhs_expr, rhs_expr)
            
            steps.append(f"Trigonometric equation: {lhs} = {rhs}")
            
            # Try to identify the variable
            var_to_solve = None
            for var in [x, y, z]:
                if var in lhs_expr.free_symbols or var in rhs_expr.free_symbols:
                    var_to_solve = var
                    break
                    
            if not var_to_solve:
                return jsonify({"error": "No variable found in equation"}), 400
                
            steps.append(f"Solve for {var_to_solve}")
            
            try:
                solutions = solve(eq, var_to_solve)
                
                if solutions:
                    # Filter out complex solutions
                    real_solutions = [sol for sol in solutions if sol.is_real]
                    
                    if real_solutions:
                        steps.append(f"Found {len(real_solutions)} real solution(s)")
                        # Get one period of solutions
                        solutions_in_period = []
                        for sol in real_solutions:
                            solutions_in_period.append(f"{var_to_solve} = {sol}")
                            steps.append(f"General solution: {var_to_solve} = {sol} + 2πn, where n is an integer")
                        
                        solution = " or ".join(solutions_in_period)
                        
                        # Prepare graph data for Flutter
                        try:
                            x_vals = [i/10 for i in range(-31, 32)]  # -π to π
                            expr = lhs_expr - rhs_expr
                            y_vals = [float(expr.subs(var_to_solve, x_val)) for x_val in x_vals]
                            
                            graph_data = {
                                "type": "trigonometric",
                                "points": [{"x": x_val, "y": y_val} for x_val, y_val in zip(x_vals, y_vals)],
                                "solutions": [float(sol) for sol in real_solutions]
                            }
                        except Exception as e:
                            logger.warning(f"Error generating graph data: {str(e)}")
                            # If graphing fails, continue without it
                            pass
                    else:
                        steps.append("No real solutions found")
                        solution = "No real solutions"
                else:
                    steps.append("No solutions found")
                    solution = "No solution"
            except Exception as e:
                logger.error(f"Error solving trigonometric equation: {str(e)}")
                return jsonify({"error": f"Error solving trigonometric equation: {str(e)}"}), 400

        elif problem_type == "limit":
            steps.append(f"Limit problem: {expression}")
            
            # Parse limit expression
            if expression.startswith("limit"):
                # Extract variable, point, and expression from the limit notation
                match = re.match(r"limit\(\s*(\w+)\s*,\s*([^,]+)\s*,\s*(.+)\s*\)", expression)
                if match:
                    var_str, point_str, expr_str = match.groups()
                    
                    # Determine the variable
                    if var_str == 'x':
                        var = x
                    elif var_str == 'y':
                        var = y
                    elif var_str == 'z':
                        var = z
                    else:
                        return jsonify({"error": f"Unsupported variable: {var_str}"}), 400
                    
                    point = safe_eval(point_str)
                    expr = safe_eval(expr_str)
                    
                    steps.append(f"Computing the limit of {expr} as {var} approaches {point}")
                    
                    try:
                        from sympy import limit as sympy_limit
                        result = sympy_limit(expr, var, point)
                        steps.append(f"Apply limit rules and evaluate")
                        steps.append(f"The limit equals {result}")
                        solution = f"lim({var_str}→{point}) {expr_str} = {result}"
                        
                        # Prepare graph data for Flutter
                        try:
                            # Generate points around the limit point for graphing
                            epsilon = 0.1
                            x_vals = [point - epsilon * (10-i)/10 for i in range(10)]
                            x_vals.extend([point + epsilon * (i+1)/10 for i in range(10)])
                            
                            # Calculate function values
                            y_vals = []
                            for x_val in x_vals:
                                try:
                                    y_val = float(expr.subs(var, x_val))
                                    y_vals.append(y_val)
                                except:
                                    y_vals.append(None)  # Handle undefined points
                            
                            graph_data = {
                                "type": "limit",
                                "points": [{"x": float(x_val), "y": y_val} for x_val, y_val in zip(x_vals, y_vals) if y_val is not None],
                                "limitPoint": float(point),
                                "limitValue": float(result) if result.is_real else None
                            }
                        except Exception as e:
                            logger.warning(f"Error generating graph data: {str(e)}")
                            # If graphing fails, continue without it
                            pass
                    except Exception as e:
                        logger.error(f"Error computing limit: {str(e)}")
                        return jsonify({"error": f"Error computing limit: {str(e)}"}), 400
                else:
                    return jsonify({"error": "Invalid limit syntax. Use format: limit(x, a, f(x))"}), 400
            else:
                return jsonify({"error": "Invalid limit syntax. Use format: limit(x, a, f(x))"}), 400
            
        elif problem_type == "statistics":
            try:
                if sub_type == "mean":
                    # Extract data from the expression
                    data_match = re.search(r'data\s*=\s*\[(.*?)\]', expression)
                    if not data_match:
                        return jsonify({"error": "Data format incorrect. Use: data = [values]"}), 400
                        
                    data_str = data_match.group(1)
                    data_list = [float(x.strip()) for x in data_str.split(',')]
                    
                    steps.append(f"Calculate the mean of the data: {data_list}")
                    steps.append("Mean = (sum of all values) / (number of values)")
                    
                    mean_value = sum(data_list) / len(data_list)
                    steps.append(f"Mean = {sum(data_list)} / {len(data_list)} = {mean_value}")
                    solution = f"Mean = {mean_value}"
                    
                    # Create graph data for visualization
                    graph_data = {
                        "type": "statistics",
                        "subType": "mean",
                        "data": data_list,
                        "mean": mean_value
                    }
                    
                elif sub_type == "median":
                    # Extract data from the expression
                    data_match = re.search(r'data\s*=\s*\[(.*?)\]', expression)
                    if not data_match:
                        return jsonify({"error": "Data format incorrect. Use: data = [values]"}), 400
                        
                    data_str = data_match.group(1)
                    data_list = [float(x.strip()) for x in data_str.split(',')]
                    sorted_data = sorted(data_list)
                    
                    steps.append(f"Calculate the median of the data: {data_list}")
                    steps.append(f"First, sort the data: {sorted_data}")
                    steps.append("Find the middle value of the sorted data")
                    
                    n = len(sorted_data)
                    if n % 2 == 0:  # Even number of items
                        median_value = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
                        steps.append(f"Since there are {n} values (even), take the average of the two middle values")
                        steps.append(f"Median = ({sorted_data[n//2 - 1]} + {sorted_data[n//2]}) / 2 = {median_value}")
                    else:  # Odd number of items
                        median_value = sorted_data[n//2]
                        steps.append(f"Since there are {n} values (odd), take the middle value")
                        steps.append(f"Median = {median_value}")
                    
                    solution = f"Median = {median_value}"
                    
                    # Create graph data for visualization
                    graph_data = {
                        "type": "statistics",
                        "subType": "median",
                        "data": data_list,
                        "median": median_value
                    }
                    
                elif sub_type == "mode":
                    # Extract data from the expression
                    data_match = re.search(r'data\s*=\s*\[(.*?)\]', expression)
                    if not data_match:
                        return jsonify({"error": "Data format incorrect. Use: data = [values]"}), 400
                        
                    data_str = data_match.group(1)
                    data_list = [float(x.strip()) for x in data_str.split(',')]
                    
                    steps.append(f"Calculate the mode of the data: {data_list}")
                    steps.append("Count the frequency of each value")
                    
                    # Count frequencies
                    from collections import Counter
                    freq = Counter(data_list)
                    max_freq = max(freq.values())
                    
                    # Find mode(s)
                    modes = [k for k, v in freq.items() if v == max_freq]
                    
                    if len(modes) == len(data_list):
                        steps.append("No mode exists as all values appear exactly once")
                        solution = "No mode exists"
                    else:
                        steps.append(f"The most frequent value(s) appear(s) {max_freq} times")
                        if len(modes) == 1:
                            steps.append(f"Mode = {modes[0]}")
                            solution = f"Mode = {modes[0]}"
                        else:
                            steps.append(f"Multiple modes exist: {modes}")
                            solution = f"Modes = {modes}"
                    
                    # Create graph data for visualization
                    graph_data = {
                        "type": "statistics",
                        "subType": "mode",
                        "data": data_list,
                        "mode": modes
                    }
                    
                elif sub_type == "standard_deviation":
                    # Extract data from the expression
                    data_match = re.search(r'data\s*=\s*\[(.*?)\]', expression)
                    if not data_match:
                        return jsonify({"error": "Data format incorrect. Use: data = [values]"}), 400
                        
                    data_str = data_match.group(1)
                    data_list = [float(x.strip()) for x in data_str.split(',')]
                    
                    steps.append(f"Calculate the standard deviation of the data: {data_list}")
                    
                    # Calculate mean
                    mean_value = sum(data_list) / len(data_list)
                    steps.append(f"Step 1: Calculate the mean = {mean_value}")
                    
                    # Calculate squared differences
                    squared_diffs = [(x - mean_value)**2 for x in data_list]
                    steps.append(f"Step 2: Calculate the squared differences from the mean")
                    steps.append(f"Squared differences: {squared_diffs}")
                    
                    # Calculate variance
                    variance = sum(squared_diffs) / len(data_list)
                    steps.append(f"Step 3: Calculate the variance (average of squared differences) = {variance}")
                    
                    # Calculate standard deviation
                    std_dev = variance**0.5
                    steps.append(f"Step 4: Calculate the standard deviation (square root of variance) = {std_dev}")
                    
                    solution = f"Standard Deviation = {std_dev}"
                    
                    # Create graph data for visualization
                    graph_data = {
                        "type": "statistics",
                        "subType": "standard_deviation",
                        "data": data_list,
                        "mean": mean_value,
                        "standardDeviation": std_dev
                    }
                    
                elif sub_type == "variance":
                    # Extract data from the expression
                    data_match = re.search(r'data\s*=\s*\[(.*?)\]', expression)
                    if not data_match:
                        return jsonify({"error": "Data format incorrect. Use: data = [values]"}), 400
                        
                    data_str = data_match.group(1)
                    data_list = [float(x.strip()) for x in data_str.split(',')]
                    
                    steps.append(f"Calculate the variance of the data: {data_list}")
                    
                    # Calculate mean
                    mean_value = sum(data_list) / len(data_list)
                    steps.append(f"Step 1: Calculate the mean = {mean_value}")
                    
                    # Calculate squared differences
                    squared_diffs = [(x - mean_value)**2 for x in data_list]
                    steps.append(f"Step 2: Calculate the squared differences from the mean")
                    steps.append(f"Squared differences: {squared_diffs}")
                    
                    # Calculate variance
                    variance = sum(squared_diffs) / len(data_list)
                    steps.append(f"Step 3: Calculate the variance (average of squared differences) = {variance}")
                    
                    solution = f"Variance = {variance}"
                    
                    # Create graph data for visualization
                    graph_data = {
                        "type": "statistics",
                        "subType": "variance",
                        "data": data_list,
                        "mean": mean_value,
                        "variance": variance
                    }
                    
                elif sub_type == "range":
                    # Extract data from the expression
                    data_match = re.search(r'data\s*=\s*\[(.*?)\]', expression)
                    if not data_match:
                        return jsonify({"error": "Data format incorrect. Use: data = [values]"}), 400
                        
                    data_str = data_match.group(1)
                    data_list = [float(x.strip()) for x in data_str.split(',')]
                    
                    steps.append(f"Calculate the range of the data: {data_list}")
                    steps.append("Range = Maximum value - Minimum value")
                    
                    min_val = min(data_list)
                    max_val = max(data_list)
                    range_val = max_val - min_val
                    
                    steps.append(f"Range = {max_val} - {min_val} = {range_val}")
                    solution = f"Range = {range_val}"
                    
                    # Create graph data for visualization
                    graph_data = {
                        "type": "statistics",
                        "subType": "range",
                        "data": data_list,
                        "min": min_val,
                        "max": max_val,
                        "range": range_val
                    }
                
                else:
                    return jsonify({"error": f"Unsupported statistics sub-type: {sub_type}"}), 400
                    
            except Exception as e:
                logger.error(f"Error in statistics calculation: {str(e)}")
                return jsonify({"error": f"Error in statistics calculation: {str(e)}"}), 400
        
        else:
            return jsonify({"error": f"Unsupported problem type: {problem_type}"}), 400
            
        # Format the solution with LaTeX notation if needed
        solution_latex = solution
        
        # Prepare the response
        result = {
            "solution": solution,
            "solution_latex": solution_latex,
            "steps": steps
        }
        
        # Add graph data if available
        if graph_data:
            result["graph_data"] = graph_data
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Generate example expressions for a given problem type
@app.route('/examples/<problem_type>', methods=['GET'])
@cache_response(timeout=3600)  # Cache for an hour
def get_examples(problem_type):
    """Returns example expressions for the requested problem type."""
    with open('examples.json', 'r') as f:
        examples = json.load(f)
        
    if problem_type in examples:
        return jsonify(examples[problem_type])
    else:
        return jsonify({"error": "Problem type not found"}), 404

# Add error handling for various HTTP error codes
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404
    
@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405
    
@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info(f"Starting Math Solver API on port {PORT}")
    logger.info(f"Debug mode: {DEBUG}")
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG)                  