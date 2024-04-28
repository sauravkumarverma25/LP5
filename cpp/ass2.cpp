#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>

using namespace std;

void bubble(vector<int> &array) {
    int n = array.size();
    #pragma omp parallel for
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (array[j] > array[j + 1]) {
                swap(array[j], array[j + 1]);
            }
        }
    }
}

void pBubble(vector<int> &array) {
    int n = array.size();
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = (i % 2) + 1; j < n; j += 2) {
            if (array[j] < array[j - 1]) {
                swap(array[j], array[j - 1]);
            }
        }
    }
}

void merge(vector<int> &arr, int low, int mid, int high) {
    int n1 = mid - low + 1;
    int n2 = high - mid;

    vector<int> left(arr.begin() + low, arr.begin() + low + n1);
    vector<int> right(arr.begin() + mid + 1, arr.begin() + mid + 1 + n2);

    int i = 0, j = 0, k = low;

    while (i < n1 && j < n2) {
        if (left[i] <= right[j]) {
            arr[k] = left[i++];
        } else {
            arr[k] = right[j++];
        }
        ++k;
    }

    while (i < n1) {
        arr[k++] = left[i++];
    }

    while (j < n2) {
        arr[k++] = right[j++];
    }
}

void parallelMergeSort(vector<int> &arr, int low, int high) {
    if (low < high) {
        int mid = (low + high) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            parallelMergeSort(arr, low, mid);
            #pragma omp section
            parallelMergeSort(arr, mid + 1, high);
        }

        merge(arr, low, mid, high);
    }
}

void mergeSort(vector<int> &arr) {
    if (arr.size() <= 1) {
        return;
    }

    int mid = arr.size() / 2;
    vector<int> left(arr.begin(), arr.begin() + mid);
    vector<int> right(arr.begin() + mid, arr.end());

    mergeSort(left);
    mergeSort(right);

    int i = 0, j = 0, k = 0;
    while (i < left.size() && j < right.size()) {
        if (left[i] < right[j]) {
            arr[k++] = left[i++];
        } else {
            arr[k++] = right[j++];
        }
    }

    while (i < left.size()) {
        arr[k++] = left[i++];
    }

    while (j < right.size()) {
        arr[k++] = right[j++];
    }
}

int main() {
    int n;
    cout << "Enter the size of the array: ";
    cin >> n;

    vector<int> arr(n);
    default_random_engine generator;
    uniform_int_distribution<int> distribution(1, 1000);

    cout << "Generated random array: ";
    for (int i = 0; i < n; ++i) {
        arr[i] = distribution(generator);
        cout << arr[i] << " ";
    }
    cout << endl;

    auto start_time = chrono::steady_clock::now();
    bubble(arr);
    auto end_time = chrono::steady_clock::now();
    cout << "Sequential Bubble Sort took: " << chrono::duration_cast
        <chrono::duration<double>>(end_time - start_time).count() << " seconds." << endl;
    cout << "Sorted Array (Bubble Sort): ";
    for (int num : arr) {
        cout << num << " ";
    }
    cout << endl;

    vector<int> arr_copy = arr;

    start_time = chrono::steady_clock::now();
    pBubble(arr_copy);
    end_time = chrono::steady_clock::now();
    cout << "Parallel Bubble Sort took: " << chrono::duration_cast
        <chrono::duration<double>>(end_time - start_time).count() << " seconds." << endl;
    cout << "Sorted Array (Parallel Bubble Sort): ";
    for (int num : arr_copy) {
        cout << num << " ";
    }
    cout << endl;

    arr_copy = arr;

    start_time = chrono::steady_clock::now();
    mergeSort(arr_copy);
    end_time = chrono::steady_clock::now();
    cout << "Sequential Merge Sort took: " << chrono::duration_cast
        <chrono::duration<double>>(end_time - start_time).count() << " seconds." << endl;
    cout << "Sorted Array (Merge Sort): ";
    for (int num : arr_copy) {
        cout << num << " ";
    }
    cout << endl;

    arr_copy = arr;

    start_time = chrono::steady_clock::now();
    parallelMergeSort(arr_copy, 0, arr_copy.size() - 1);
    end_time = chrono::steady_clock::now();
    cout << "Parallel Merge Sort took: " << chrono::duration_cast
        <chrono::duration<double>>(end_time - start_time).count() << " seconds." << endl;
    cout << "Sorted Array (Parallel Merge Sort): ";
    for (int num : arr_copy) {
        cout << num << " ";
    }
    cout << endl;

    return 0;
}


