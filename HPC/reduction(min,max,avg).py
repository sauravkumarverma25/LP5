import numpy as np
import multiprocessing as mp
import time

def minval(arr):
    return min(arr)

def maxval(arr):
    return max(arr)

def sum(arr):
    return np.sum(arr)

def average(arr):
    return np.mean(arr)

if __name__ == "__main__":
    n = int(input("Enter the size of the array: "))

    # Generate random array elements
    arr = np.random.randint(0, 200, size=n)

    print(f"\nGenerated random array of length {n} with elements between 0 to 200\n")
    print("Given array is =>\n")
    print(", ".join(map(str, arr)))
    print("\n")

    start_time = time.time()
    with mp.Pool() as pool:
        min_val = min(pool.map(minval, [arr]))
    end_time = time.time()
    print("Sequential Min:", min_val, f"({(end_time - start_time) * 1000:.0f}ms)")

    start_time = time.time()
    with mp.Pool() as pool:
        min_val = min(pool.map(minval, [arr]))
    end_time = time.time()
    print("Parallel (16) Min:", min_val, f"({(end_time - start_time) * 1000:.0f}ms)")

    start_time = time.time()
    with mp.Pool() as pool:
        max_val = max(pool.map(maxval, [arr]))
    end_time = time.time()
    print("Sequential Max:", max_val, f"({(end_time - start_time) * 1000:.0f}ms)")

    start_time = time.time()
    with mp.Pool() as pool:
        max_val = max(pool.map(maxval, [arr]))
    end_time = time.time()
    print("Parallel (16) Max:", max_val, f"({(end_time - start_time) * 1000:.0f}ms)")

    start_time = time.time()
    with mp.Pool() as pool:
        total_sum = sum(pool.map(sum, [arr]))
    end_time = time.time()
    print("Sequential Sum:", total_sum, f"({(end_time - start_time) * 1000:.0f}ms)")

    start_time = time.time()
    with mp.Pool() as pool:
        total_sum = sum(pool.map(sum, [arr]))
    end_time = time.time()
    print("Parallel (16) Sum:", total_sum, f"({(end_time - start_time) * 1000:.0f}ms)")

    start_time = time.time()
    with mp.Pool() as pool:
        avg = average(pool.map(sum, [arr])) / n
    end_time = time.time()
    print("Sequential Average:", avg, f"({(end_time - start_time) * 1000:.0f}ms)")

    start_time = time.time()
    with mp.Pool() as pool:
        avg = average(pool.map(sum, [arr])) / n
    end_time = time.time()
    print("Parallel (16) Average:", avg, f"({(end_time - start_time) * 1000:.0f}ms)")
