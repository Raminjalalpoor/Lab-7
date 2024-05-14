import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter
import time
import random


def generate_int_list(size):
    return [random.randint(1, 101) for _ in range(size)]

def generate_int_array(low, high, size):
    return np.random.randint(low, high + 1, size=size)

def time_for_numpy_operations(arr1, arr2):
    start = time.perf_counter()
    np.multiply(arr1, arr2)
    end = time.perf_counter()
    return end - start

def time_for_standard_list_operations(list1, list2):
    start = time.perf_counter()
    results = [x * y for x, y in zip(list1, list2)]
    end = time.perf_counter()
    return end - start

def performance_comparison():
    random_list_a = generate_int_list(1000000)
    random_list_b = generate_int_list(1000000)

    list_timing = time_for_standard_list_operations(random_list_a, random_list_b)
    print('Timing with standard lists:', list_timing)

    numpy_array_a = generate_int_array(0, 1000, 1000000)
    numpy_array_b = generate_int_array(0, 1000, 1000000)

    numpy_timing = time_for_numpy_operations(numpy_array_a, numpy_array_b)
    print('Timing with NumPy arrays:', numpy_timing)

    speed_ratio = list_timing / numpy_timing
    print('NumPy is', speed_ratio, 'times faster than list')

def visualize_data_histogram():
    data = np.genfromtxt('data2.csv', delimiter=',', skip_header=1)
    valid_data = np.array(data[:, 0], dtype=float)
    valid_data = valid_data[~np.isnan(valid_data)]

    plt.figure(figsize=(10, 8))
    plt.hist(valid_data, bins=40, color='green', alpha=0.75, edgecolor='darkgreen')
    plt.grid(True, linestyle='--')
    plt.title('Data Distribution')
    plt.xlabel(f'Values\nStandard Deviation: {np.std(valid_data):.2f}')
    plt.ylabel('Frequency Count')
    plt.show()

def render_3d_function_plot():
    x = np.linspace(-2 * np.pi, 2 * np.pi, 40)
    y = np.tan(x)
    z = y ** 2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, linestyle='-', color='purple', marker='^')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Tangent Squared Plot')
    plt.show()

def animate_sine_wave():
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'r-')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-2, 2)

    writer = PillowWriter(fps=25)
    x_points = np.linspace(-10, 10, 300)
    y_points = np.sin(x_points)

    with writer.saving(fig, "sine_wave.gif", 100):
        for x, y in zip(x_points, y_points):
            line.set_data(x_points[x_points <= x], y_points[x_points <= x])
            writer.grab_frame()

if __name__ == '__main__':
    performance_comparison()
    visualize_data_histogram()
    render_3d_function_plot()
    animate_sine_wave()
