import time
import random

def bubble(array):
    n = len(array)
    for i in range(n - 1):
        for j in range(n - i - 1):
            if array[j] > array[j + 1]:
                array[j], array[j + 1] = array[j + 1], array[j]

def pBubble(array):
    n = len(array)
    for i in range(n):
        for j in range(1, n, 2):
            if array[j] < array[j - 1]:
                array[j], array[j - 1] = array[j - 1], array[j]

        # Synchronize
        # No need for barrier in Python as the GIL (Global Interpreter Lock) ensures only one thread executes Python code at a time

        for j in range(2, n, 2):
            if array[j] < array[j - 1]:
                array[j], array[j - 1] = array[j - 1], array[j]

def merge(arr, low, mid, high):
    n1 = mid - low + 1
    n2 = high - mid

    left = arr[low:low + n1]
    right = arr[mid + 1:mid + 1 + n2]

    i = j = 0
    k = low

    while i < n1 and j < n2:
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1

    while i < n1:
        arr[k] = left[i]
        i += 1
        k += 1

    while j < n2:
        arr[k] = right[j]
        j += 1
        k += 1

def parallelMergeSort(arr, low, high):
    if low < high:
        mid = (low + high) // 2

        parallelMergeSort(arr, low, mid)
        parallelMergeSort(arr, mid + 1, high)

        merge(arr, low, mid, high)

def mergeSort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]

        mergeSort(left)
        mergeSort(right)

        i = j = k = 0

        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1

        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1

if __name__ == "__main__":
    n = int(input("Enter the size of the array: "))
    arr = [random.randint(1, 1000) for _ in range(n)]

    start_time = time.time()
    bubble(arr)
    end_time = time.time()
    print("Sequential Bubble Sort took:", end_time - start_time, "seconds.")
    print("Sorted Array (Bubble Sort):", arr)

    arr_copy = arr.copy()

    start_time = time.time()
    pBubble(arr_copy)
    end_time = time.time()
    print("Parallel Bubble Sort took:", end_time - start_time, "seconds.")
    print("Sorted Array (Parallel Bubble Sort):", arr_copy)

    arr_copy = arr.copy()

    start_time = time.time()
    mergeSort(arr_copy)
    end_time = time.time()
    print("Sequential Merge Sort took:", end_time - start_time, "seconds.")
    print("Sorted Array (Merge Sort):", arr_copy)

    arr_copy = arr.copy()

    start_time = time.time()
    parallelMergeSort(arr_copy, 0, len(arr_copy) - 1)
    end_time = time.time()
    print("Parallel Merge Sort took:", end_time - start_time, "seconds.")
    print("Sorted Array (Parallel Merge Sort):", arr_copy)

'''
import time

def bubble(array):
    n = len(array)
    for i in range(n - 1):
        for j in range(n - i - 1):
            if array[j] > array[j + 1]:
                array[j], array[j + 1] = array[j + 1], array[j]

def pBubble(array):
    n = len(array)
    for i in range(n):
        for j in range(1, n, 2):
            if array[j] < array[j - 1]:
                array[j], array[j - 1] = array[j - 1], array[j]

        # Synchronize
        # No need for barrier in Python as the GIL (Global Interpreter Lock) ensures only one thread executes Python code at a time

        for j in range(2, n, 2):
            if array[j] < array[j - 1]:
                array[j], array[j - 1] = array[j - 1], array[j]

def merge(arr, low, mid, high):
    n1 = mid - low + 1
    n2 = high - mid

    left = arr[low:low + n1]
    right = arr[mid + 1:mid + 1 + n2]

    i = j = 0
    k = low

    while i < n1 and j < n2:
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1

    while i < n1:
        arr[k] = left[i]
        i += 1
        k += 1

    while j < n2:
        arr[k] = right[j]
        j += 1
        k += 1

def parallelMergeSort(arr, low, high):
    if low < high:
        mid = (low + high) // 2

        parallelMergeSort(arr, low, mid)
        parallelMergeSort(arr, mid + 1, high)

        merge(arr, low, mid, high)

def mergeSort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]

        mergeSort(left)
        mergeSort(right)

        i = j = k = 0

        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1

        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1

if __name__ == "__main__":
    n = int(input("Enter the size of the array: "))
    arr = []

    print("Enter the elements of the array:")
    for i in range(n):
        element = int(input())
        arr.append(element)

    start_time = time.time()
    bubble(arr)
    end_time = time.time()
    print("Sequential Bubble Sort took:", end_time - start_time, "seconds.")
    print("Sorted Array (Bubble Sort):", arr)

    arr_copy = arr.copy()

    start_time = time.time()
    pBubble(arr_copy)
    end_time = time.time()
    print("Parallel Bubble Sort took:", end_time - start_time, "seconds.")
    print("Sorted Array (Parallel Bubble Sort):", arr_copy)

    arr_copy = arr.copy()

    start_time = time.time()
    mergeSort(arr_copy)
    end_time = time.time()
    print("Sequential Merge Sort took:", end_time - start_time, "seconds.")
    print("Sorted Array (Merge Sort):", arr_copy)

    arr_copy = arr.copy()

    start_time = time.time()
    parallelMergeSort(arr_copy, 0, len(arr_copy) - 1)
    end_time = time.time()
    print("Parallel Merge Sort took:", end_time - start_time, "seconds.")
    print("Sorted Array (Parallel Merge Sort):", arr_copy)
'''