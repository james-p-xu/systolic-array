"""
Systolic Array Implementation: C = A @ B using a weight-stationary systolic array

- Weights A[i, k] are preloaded into PE(i, k) and stay stationary.
- Activations B[k, j] stream North-to-South down column k.
- Partial sums stream West-to-East along row i.
- Final results C[i, j] emerge from the East edge of the PE array.

(This does not exactly match the example animation in the README!)
"""

import numpy as np  # Verify against standard matmul


class ProcessingElement:
    def __init__(self):
        """A single Processing Element (PE) in the systolic array."""
        self.a = 0  # Stationary weight (from matrix A)
        self.b = 0  # Activation (from matrix B), latched at start of cycle
        self.sum = 0  # Partial sum (for matrix C), latched from West at start of cycle

    def compute_sum_out(self):
        """Performs MAC and returns the sum to propagate East."""
        return self.sum + (
            self.a * self.b
        )  # nb: self.sum holds the value from the West PE (or 0)


class SystolicArray:
    def __init__(self, M, N, K, A, B):
        """
        An M x K grid of PEs performing C = A @ B.

        M: Number of rows in A and C (PE array rows).
        N: Number of columns in B and C (Output dimension).
        K: Inner dimension of matmul (columns of A, rows of B, PE array columns).
        A: Weight matrix (M x K).
        B: Activation matrix (K x N).
        """
        assert len(A) == M and len(A[0]) == K, f"A must be {M}x{K}"
        assert len(B) == K and len(B[0]) == N, f"B must be {K}x{N}"

        self.M, self.N, self.K = M, N, K
        self.A, self.B = A, B

        # M x K PE grid
        self.pes = [[ProcessingElement() for _ in range(K)] for _ in range(M)]

        # Preload weights (weight-stationary): Each PE's 'a' register holds one weight element.
        for i in range(M):
            for k in range(K):  # k is the PE column index
                self.pes[i][k].a = A[i][k]

    def run(self):
        """
        Multiply A (M x K) by B (K x N) using a systolic array.
        Returns C (M x N) and the number of cycles executed.
        """
        M, N, K = self.M, self.N, self.K
        C = [[0] * N for _ in range(M)]

        # Total cycles (0‑indexed): (M-1) + (N-1) + (K-1) = M+N+K-3
        #   -> total # of cycles = (M+N+K-3) + 1 = M+N+K-2
        total_cycles_index = M + N + K - 3

        # Initialize PE states (registers)
        for i in range(M):
            for k in range(K):
                pe = self.pes[i][k]
                pe.b = 0  # Activation register
                pe.sum = 0  # Partial sum (from West) register

        # Buffers for inputs arriving DURING cycle t (like physical wires)
        # Values in these buffers will be latched into PE registers at the END of cycle t
        next_b_inputs = [
            [0] * K for _ in range(M)
        ]  # For activations arriving from North
        next_sum_inputs = [
            [0] * K for _ in range(M)
        ]  # For partial sums arriving from West

        # Handle initial activation B[0][0] for cycle t=0
        if 0 < K and 0 < N:
            self.pes[0][0].b = self.B[0][0]

        for t in range(total_cycles_index + 1):  # Execute cycles 0 to M+N+K-3
            # Step 1: Compute & Determine Next State
            # Based on values latched at the START of cycle t
            for i in range(M):
                for k in range(K):  # k = PE column index
                    pe = self.pes[i][k]

                    # Each PE performs one MAC operation this cycle
                    sum_out = pe.compute_sum_out()

                    # Collect final result C[i, j] if emerging from East edge
                    if k == K - 1:  # Check if this PE is on the eastern boundary
                        # Sum emerges during cycle t = i + j + K - 1
                        j_out = (
                            t - i - (K - 1)
                        )  # Calculate the corresponding output column j
                        if 0 <= j_out < N:
                            C[i][j_out] = sum_out  # Store the final dot product

                    # Determine Activation arriving for NEXT latch (from North/Injection)
                    # This populates the 'next_b_inputs' buffer during cycle t
                    if i == 0:  # Top row PEs receive injected activations
                        # Activation B[k, j] needed at PE(0,k) for start of t+1 if t+1 = k+j
                        j_inject = t + 1 - k
                        if 0 <= j_inject < N and 0 <= k < K:
                            next_b_inputs[i][k] = self.B[k][j_inject]
                        # else: buffer remains 0 (no injection this cycle for this PE)
                    else:  # Other rows receive activation from PE above
                        # Forward activation south: Use the value latched by PE above at start of t
                        next_b_inputs[i][k] = self.pes[i - 1][k].b

                    # Determine Partial Sum arriving for NEXT latch (from West)
                    # This populates the 'next_sum_inputs' buffer during cycle t
                    if k == 0:  # West edge PEs receive 0 sum input
                        next_sum_inputs[i][k] = 0
                    else:  # Other PEs receive sum computed THIS cycle by PE to the West
                        pe_west = self.pes[i][k - 1]
                        sum_out_west = (
                            pe_west.compute_sum_out()
                        )  # Use sum computed by west PE
                        next_sum_inputs[i][k] = sum_out_west

            # Step 2: Latch Phase
            # ALL PEs simultaneously update registers for the START of the next cycle (t+1)
            for i in range(M):
                for k in range(K):
                    pe = self.pes[i][k]
                    pe.b = next_b_inputs[i][k]  # Latch activation
                    pe.sum = next_sum_inputs[i][k]  # Latch partial sum from West

        # Return the final C matrix and the index of the last cycle executed
        return C, total_cycles_index


# Helper function to run a single test case
def run_test(A, B, test_name="Test Case"):
    print(f"\n--- {test_name} ---")
    if not A or not isinstance(A[0], list) or not B or not isinstance(B[0], list):
        print("Invalid input matrices (must be list of lists).")
        return

    M, K_A = len(A), len(A[0])
    K_B, N = len(B), len(B[0])

    if K_A != K_B:
        print(f"Dimension mismatch: K_A={K_A}, K_B={K_B}. Cannot multiply.")
        return
    K = K_A

    print(f"Running Systolic Array M={M}, N={N}, K={K}")
    print("Matrix A (Weights):")
    for row in A:
        print(f"  {row}")
    print("Matrix B (Activations):")
    for row in B:
        print(f"  {row}")

    sa = SystolicArray(M, N, K, A, B)
    C, total_cycles_idx = sa.run()
    num_cycles_run = total_cycles_idx + 1
    expected_cycles = M + N + K - 2

    print(f"Total cycles executed: {num_cycles_run} (Expected: {expected_cycles})")
    print("Result C = A @ B using Systolic Array:")

    try:
        correct_C = np.dot(np.array(A), np.array(B)).tolist()
        if np.allclose(C, correct_C):
            print("Result is CORRECT.")
            print("Computed C:")
            for row in C:
                print(f"  {row}")
        else:
            print("Result is INCORRECT.")
            print("Computed C:")
            for row in C:
                print(f"  {row}")
            print("Expected C (using numpy.dot):")
            for row in correct_C:
                print(f"  {row}")
    except Exception as e:
        print(f"Error during verification: {e}")
        print("Computed C:")
        for row in C:
            print(f"  {row}")


if __name__ == "__main__":
    # Test Case 1: Original (3 x 3) x (3 x 3)
    A1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    B1 = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
    run_test(A1, B1, "Original 3x3x3 Test")

    # Test Case 2: Non-Square; M=2, N=4, K=3
    A2 = [[1, 0, 2], [0, 3, 1]]  # 2x3
    B2 = [[1, 2, 0, 1], [3, 0, 1, 0], [2, 1, 0, 4]]  # 3x4
    run_test(A2, B2, "Non-Square Test (M=2, N=4, K=3)")

    # Test Case 3: Non-Square; M=4, N=2, K=2
    A3 = [[1, 2], [3, 4], [5, 6], [7, 8]]  # 4x2
    B3 = [[10, 20], [30, 40]]  # 2x2
    run_test(A3, B3, "Non-Square Test (M=4, N=2, K=2)")

    # Test Case 4: Inner dimension K=1
    A4 = [[1], [2], [3]]  # 3x1
    B4 = [[10, 20, 30, 40]]  # 1x4
    run_test(A4, B4, "Inner Dimension K=1 Test")

    # Test Case 5: Identity and another matrix
    A5 = [[1, 0], [0, 1]]  # 2x2 (Identity)
    B5 = [[5, 6], [7, 8]]  # 2x2
    run_test(A5, B5, "Identity Matrix Test")

    # Test Case 6: Matrix with zeros
    A6 = [[1, 0, 1], [0, 2, 0]]  # 2x3
    B6 = [[0, 1], [2, 0], [0, 3]]  # 3x2
    run_test(A6, B6, "Zeros Test")
