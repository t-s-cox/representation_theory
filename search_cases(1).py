import sympy
from itertools import product
import time
from multiprocessing import Pool, cpu_count

def principal_polynomial(k, t1, t2, t3, s1=0, s2=0, s3=0):
    k0, k1, k2, k3 = k
    term_k4 = t1 * t2 * t3 * (k0**4 + k1**4 + k2**4 + k3**4)
    
    s3_minus_s2_sq = (s3 - s2)**2
    s3_minus_s1_sq = (s3 - s1)**2
    s1_minus_s2_sq = (s1 - s2)**2
    
    term_k2k2_1 = t1 * (t2**2 + t3**2 + s3_minus_s2_sq) * (k2**2 * k3**2 - k0**2 * k1**2)
    term_k2k2_2 = t2 * (t1**2 + t3**2 + s3_minus_s1_sq) * (k1**2 * k3**2 - k0**2 * k2**2)
    term_k2k2_3 = t3 * (t1**2 + t2**2 + s1_minus_s2_sq) * (k1**2 * k2**2 - k0**2 * k3**2)
    
    term_k0k1k2k3 = 2 * k0 * k1 * k2 * k3 * (
        t1**2 * (s3 - s1) +
        t2**2 * (s1 - s3) -
        (t3**2 + (s3 - s2) * (s3 - s1)) * (s1 - s2)
    )
    
    return sympy.simplify(term_k4 + term_k2k2_1 + term_k2k2_2 + term_k2k2_3 + term_k0k1k2k3)

def calculate_determinant(k, s0, s1, s2, s3):
    k0, k1, k2, k3 = k
    matrix_sum = k0 * s0 + k1 * s1 + k2 * s2 + k3 * s3
    return sympy.simplify(matrix_sum.det())

def check_combination(args):
    s0, s1, s2, s3, target_poly, k_vec = args
    current_det = calculate_determinant(k_vec, s0, s1, s2, s3)
    if sympy.simplify(current_det - target_poly) == 0:
        return (s0, s1, s2, s3)
    return None

def print_solution(matrices):
    for i, matrix in enumerate(matrices):
        print(f"sigma^{i}:")
        sympy.pprint(matrix)
        print()

def create_anti_diagonal(vec):
    if len(vec) != 4:
        raise ValueError("Input vector must have 4 elements for a 4x4 anti-diagonal matrix.")
    mat = sympy.zeros(4)
    for i in range(4):
        mat[i, 3 - i] = vec[i]
    return mat

def search_for_solution(case_name, target_poly, det_constraint, mag_pool, k_vec):
    print(f"\nSearching for solution to: {case_name}")
    print("=" * 50)
    print(f"Target Polynomial: {target_poly}")
    
    I = sympy.I
    
    s_diag_candidates = []
    s_anti_diag_candidates = []

    signs_real = [1, -1]
    signs_imag = [I, -I]
    sign_groups = list(product(signs_real, signs_real)) + list(product(signs_imag, signs_imag))
    
    for mag_pair in [mag_pool, mag_pool[::-1]]:
        a_mag, b_mag = mag_pair
        for a_sign, b_sign in sign_groups:
            a = a_sign * a_mag
            b = b_sign * b_mag
            
            vec_sym = [a, b, b, a]
            s_diag_sym = sympy.diag(*vec_sym)
            if sympy.simplify(s_diag_sym.det() - det_constraint) == 0:
                s_diag_candidates.append(s_diag_sym)
                s_anti_diag_candidates.append(create_anti_diagonal(vec_sym))

            vec_asym = [a, b, -b, -a]
            s_diag_asym = sympy.diag(*vec_asym)
            if sympy.simplify(s_diag_asym.det() - det_constraint) == 0:
                s_diag_candidates.append(s_diag_asym)
                s_anti_diag_candidates.append(create_anti_diagonal(vec_asym))
    
    unique_diag_candidates = []
    for mat in s_diag_candidates:
        if mat not in unique_diag_candidates:
            unique_diag_candidates.append(mat)
    s_diag_candidates = unique_diag_candidates
    
    unique_anti_diag_candidates = []
    for mat in s_anti_diag_candidates:
        if mat not in unique_anti_diag_candidates:
            unique_anti_diag_candidates.append(mat)
    s_anti_diag_candidates = unique_anti_diag_candidates

    print(f"Generated {len(s_diag_candidates)} diagonal candidates.")
    print(f"Generated {len(s_anti_diag_candidates)} anti-diagonal candidates.")

    search_space = list(product(s_diag_candidates, s_anti_diag_candidates, s_anti_diag_candidates, s_diag_candidates))
    total_combinations = len(search_space)
    
    tasks = [(s0, s1, s2, s3, target_poly, k_vec) for s0, s1, s2, s3 in search_space]
    
    num_cores = cpu_count()
    print(f"Starting parallel search over {total_combinations} combinations using {num_cores} cores...")
    start_time = time.time()
    
    solution_found = None
    
    with Pool(processes=num_cores) as pool:
        results_iterator = pool.imap_unordered(check_combination, tasks)
        
        for i, result in enumerate(results_iterator):
            if i > 0 and i % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"Checked {i}/{total_combinations}... ({elapsed:.2f}s elapsed)", end='\r')

            if result is not None:
                solution_found = result
                print("\n" + "!" * 20)
                print("!!!         SOLUTION FOUND         !!!")
                print("!" * 20)
                pool.terminate()
                break

    end_time = time.time()
    
    if solution_found:
        print(f"Found in {end_time - start_time:.2f} seconds.")
        print_solution(solution_found)
    else:
        print(f"\nSearch complete. No solution found. Total time: {end_time - start_time:.2f}s")


def main():
    k0, k1, k2, k3 = sympy.symbols('k0 k1 k2 k3')
    k_vec = (k0, k1, k2, k3)
    t = sympy.symbols('tau', positive=True)

    # --- Case 0: t1 = t2 = t3 = t ---
    # p_k_c0_expected = t**3 * (k0**2 - k1**2 - k2**2 - k3**2)**2
    # det_c0 = t**3
    # mags_c0 = [t, sympy.sqrt(t)]
    # search_for_solution("Case 0", p_k_c0_expected, det_c0, mags_c0, k_vec)

    
    # --- Case 1: t1=t, t2=t3=1 ---
    #t1_sym = sympy.symbols('tau1', positive=True)
    #p_k_c1_expected = principal_polynomial(k_vec, t1_sym, 1, 1)
    #det_c1 = t1_sym
    #mags_c1 = [sympy.sqrt(t1_sym), 1]
    #search_for_solution("Case 1", p_k_c1_expected, det_c1, mags_c1, k_vec)

    # --- Case 2: t1 = t, t2 = c * t3 ---
    t, c = sympy.symbols('tau c', positive=True)
    t1_sym = t
    t3_sym = sympy.symbols('tau3', positive=True)
    t2_sym = c * t3_sym
    p_k_c2_expected = principal_polynomial(k_vec, t1_sym, t2_sym, t3_sym)
    det_c2 = t1_sym * t2_sym * t3_sym
    mags_c2 = [sympy.sqrt(t1_sym), sympy.sqrt(t3_sym)]
    search_for_solution("Case 2", p_k_c2_expected, det_c2, mags_c2, k_vec)
    
    # --- Case 3: t1 = t, t2 = c2 * t, t3 = c3 * t ---
    t, c2, c3 = sympy.symbols('tau c2 c3', positive=True)
    t1_sym = t
    t2_sym = c2 * t
    t3_sym = c3 * t
    p_k_c3_expected = principal_polynomial(k_vec, t1_sym, t2_sym, t3_sym)
    det_c3 = t1_sym * t2_sym * t3_sym
    mags_c3 = [sympy.sqrt(t1_sym), sympy.sqrt(t2_sym)]
    search_for_solution("Case 3", p_k_c3_expected, det_c3, mags_c3, k_vec)

if __name__ == "__main__":
    main()

