import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Дані з таблиці (Варіант 7)
x = np.array([0.4, 0.6, 0.9, 1.4, 2.0])
y = np.array([2.45, 1.63, 0.95, 0.73, 1.95])

# Крок 1: Обчислення кінцевих різниць для інтерполяції Ньютона
n = len(x)
div_diff = np.zeros((n, n))
div_diff[:, 0] = y  # Перша колонка - це значення y

for j in range(1, n):
    for i in range(n - j):
        div_diff[i, j] = (div_diff[i + 1, j - 1] - div_diff[i, j - 1]) / (x[i + j] - x[i])

# Крок 2: Обчислення похідних
def newton_first_derivative(x_val, x_data, div_diff):
    """
    Обчислення першої похідної за допомогою інтерполяції Ньютона.
    """
    n = len(x_data)
    derivative = div_diff[0, 1]
    product_term = 1
    for i in range(1, n - 1):
        product_term *= (x_val - x_data[i - 1])
        derivative += product_term * div_diff[0, i + 1]
    return derivative

def newton_second_derivative(x_val, x_data, div_diff):
    """
    Обчислення другої похідної за допомогою інтерполяції Ньютона.
    """
    n = len(x_data)
    if n < 3:
        return None  # Потрібно як мінімум 3 точки для другої похідної

    second_derivative = div_diff[0, 2]
    product_term = 1
    for i in range(2, min(n - 2, 3)):  # Обмеження по індексах
        product_term *= (x_val - x_data[i - 2])
        if i + 2 < n:
            second_derivative += product_term * div_diff[0, i + 2]
    return second_derivative

# Обчислення похідних для конкретних точок, наприклад x = 0.9
x_val = 0.9
first_derivative = newton_first_derivative(x_val, x, div_diff)
second_derivative = newton_second_derivative(x_val, x, div_diff)

# Виведення таблиці кінцевих різниць
div_diff_table = pd.DataFrame(div_diff, columns=[f'd{i}' for i in range(n)], index=[f'x{i}' for i in range(n)])
print("Таблиця кінцевих різниць:")
print(div_diff_table)

# Крок 3: Побудова графіків функції та її похідних
x_values = np.linspace(min(x), max(x), 100)
y_values = np.interp(x_values, x, y)
y_prime_values = [newton_first_derivative(val, x, div_diff) for val in x_values]
y_double_prime_values = [newton_second_derivative(val, x, div_diff) for val in x_values]

plt.figure(figsize=(10, 6))

# Графік функції
plt.subplot(3, 1, 1)
plt.plot(x_values, y_values, label='f(x)', color='blue')
plt.scatter(x, y, color='red', zorder=5)
plt.title('Графік функції f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

# Графік першої похідної
plt.subplot(3, 1, 2)
plt.plot(x_values, y_prime_values, label="f'(x)", color='green')
plt.title("Перша похідна f'(x)")
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.legend()

# Графік другої похідної
plt.subplot(3, 1, 3)
plt.plot(x_values, y_double_prime_values, label="f''(x)", color='orange')
plt.title("Друга похідна f''(x)")
plt.xlabel('x')
plt.ylabel("f''(x)")
plt.legend()

plt.tight_layout()
plt.show()

# Виведення результатів
print(f"Перша похідна в точці x = {x_val}: f'({x_val}) = {first_derivative}")
print(f"Друга похідна в точці x = {x_val}: f''({x_val}) = {second_derivative}")


