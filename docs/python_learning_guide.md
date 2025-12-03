# Python Learning Guide for C Programmers
## Focus: Machine Learning & AI Applications

---

## Table of Contents
1. [Python vs C: Key Differences](#1-python-vs-c-key-differences)
2. [Python Fundamentals](#2-python-fundamentals)
3. [Data Structures Deep Dive](#3-data-structures-deep-dive)
4. [Object-Oriented Python](#4-object-oriented-python)
5. [Functional Programming Features](#5-functional-programming-features)
6. [Essential Libraries for ML/AI](#6-essential-libraries-for-mlai)
7. [NumPy - The Foundation](#7-numpy---the-foundation)
8. [Pandas - Data Manipulation](#8-pandas---data-manipulation)
9. [Matplotlib & Visualization](#9-matplotlib--visualization)
10. [Scikit-Learn - Classical ML](#10-scikit-learn---classical-ml)
11. [Deep Learning Frameworks](#11-deep-learning-frameworks)
12. [Best Practices & Pythonic Code](#12-best-practices--pythonic-code)
13. [Learning Roadmap](#13-learning-roadmap)

---

## 1. Python vs C: Key Differences

### Memory Management
```c
// C: Manual memory allocation
int *arr = (int*)malloc(10 * sizeof(int));
// ... use array ...
free(arr);  // Must free manually!
```

```python
# Python: Automatic garbage collection
arr = [0] * 10  # List created
# No need to free - Python handles it automatically
```

### Variable Declaration
```c
// C: Explicit typing required
int x = 10;
float y = 3.14;
char name[] = "Hello";
```

```python
# Python: Dynamic typing - no type declaration needed
x = 10          # int
y = 3.14        # float
name = "Hello"  # str

# Variables can change type (though not recommended)
x = "now I'm a string"
```

### Indentation Matters!
```c
// C: Braces define blocks
if (x > 0) {
    printf("Positive");
    x++;
}
```

```python
# Python: Indentation defines blocks (typically 4 spaces)
if x > 0:
    print("Positive")
    x += 1
# No braces, no semicolons!
```

### No Pointers (Mostly)
In Python, everything is an object reference. You don't manipulate memory addresses directly.

```python
# Python passes objects by reference (but immutables act like pass-by-value)
def modify_list(lst):
    lst.append(4)  # This modifies the original!

my_list = [1, 2, 3]
modify_list(my_list)
print(my_list)  # [1, 2, 3, 4]
```

### Comparison Table

| Feature | C | Python |
|---------|---|--------|
| Typing | Static | Dynamic |
| Compilation | Compiled to machine code | Interpreted (bytecode) |
| Memory | Manual (malloc/free) | Automatic (garbage collection) |
| Speed | Very fast | Slower (but libraries use C under the hood) |
| Syntax | Verbose, explicit | Concise, readable |
| Arrays | Fixed size, homogeneous | Dynamic, heterogeneous (lists) |
| Strings | Char arrays, null-terminated | Immutable objects |
| Entry point | main() function | Top-level code or `if __name__ == "__main__":` |

---

## 2. Python Fundamentals

### 2.1 Basic Data Types

```python
# Numbers
integer_num = 42
float_num = 3.14159
complex_num = 3 + 4j  # Python has built-in complex numbers!

# Strings (immutable)
single = 'Hello'
double = "World"
multiline = """This is a
multiline string"""
f_string = f"The answer is {integer_num}"  # Formatted string (Python 3.6+)

# Boolean
is_valid = True
is_empty = False

# None (like NULL in C)
result = None
```

### 2.2 Operators

```python
# Arithmetic
a = 10
b = 3
print(a + b)   # 13 - Addition
print(a - b)   # 7  - Subtraction
print(a * b)   # 30 - Multiplication
print(a / b)   # 3.333... - True division (always float!)
print(a // b)  # 3  - Floor division (like C integer division)
print(a % b)   # 1  - Modulo
print(a ** b)  # 1000 - Exponentiation (no ^ like in C)

# Comparison
print(a == b)  # False
print(a != b)  # True
print(a > b)   # True

# Logical (and, or, not - NOT &&, ||, !)
print(True and False)  # False
print(True or False)   # True
print(not True)        # False

# Identity
x = [1, 2, 3]
y = x
z = [1, 2, 3]
print(x is y)      # True (same object)
print(x is z)      # False (different objects, same value)
print(x == z)      # True (same value)

# Membership
print(2 in [1, 2, 3])      # True
print('a' in 'abc')        # True
print('x' not in 'abc')    # True
```

### 2.3 Control Flow

```python
# If-elif-else
score = 85

if score >= 90:
    grade = 'A'
elif score >= 80:
    grade = 'B'
elif score >= 70:
    grade = 'C'
else:
    grade = 'F'

# Ternary operator (conditional expression)
status = "pass" if score >= 60 else "fail"

# For loops - iterate over sequences
for i in range(5):      # 0, 1, 2, 3, 4
    print(i)

for i in range(2, 10, 2):  # start, stop, step: 2, 4, 6, 8
    print(i)

# Iterate over collections directly (Pythonic!)
fruits = ['apple', 'banana', 'cherry']
for fruit in fruits:
    print(fruit)

# With index using enumerate
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")

# While loop
count = 0
while count < 5:
    print(count)
    count += 1  # Note: no ++ operator in Python!

# Break and continue work like in C
for i in range(10):
    if i == 3:
        continue  # Skip 3
    if i == 7:
        break     # Stop at 7
    print(i)
```

### 2.4 Functions

```python
# Basic function
def greet(name):
    """This is a docstring - documents the function."""
    return f"Hello, {name}!"

# Default arguments
def power(base, exponent=2):
    return base ** exponent

print(power(3))      # 9 (uses default exponent=2)
print(power(3, 3))   # 27

# Keyword arguments
def create_user(name, age, city="Unknown"):
    return {"name": name, "age": age, "city": city}

user = create_user(age=25, name="Alice")  # Order doesn't matter with keywords

# *args - variable positional arguments (like ... in C)
def sum_all(*numbers):
    return sum(numbers)

print(sum_all(1, 2, 3, 4, 5))  # 15

# **kwargs - variable keyword arguments
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=25, city="NYC")

# Lambda functions (anonymous functions)
square = lambda x: x ** 2
print(square(5))  # 25

# Type hints (Python 3.5+) - optional but recommended for ML projects
def calculate_mean(numbers: list[float]) -> float:
    return sum(numbers) / len(numbers)
```

### 2.5 Exception Handling

```python
# Unlike C (error codes), Python uses exceptions
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
except Exception as e:
    print(f"An error occurred: {e}")
else:
    print("Success!")  # Runs if no exception
finally:
    print("This always runs")  # Cleanup code

# Raising exceptions
def validate_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    return age

# Common exceptions you'll encounter in ML:
# - ValueError: Invalid value
# - TypeError: Wrong type
# - IndexError: List index out of range
# - KeyError: Dictionary key not found
# - FileNotFoundError: File doesn't exist
# - ImportError: Module not found
```

---

## 3. Data Structures Deep Dive

### 3.1 Lists (Dynamic Arrays)

```python
# Creating lists
empty_list = []
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]  # Can mix types (but usually don't)

# Indexing (0-based like C)
print(numbers[0])    # 1 (first element)
print(numbers[-1])   # 5 (last element - negative indexing!)
print(numbers[-2])   # 4 (second to last)

# Slicing [start:stop:step] - extremely powerful!
print(numbers[1:4])    # [2, 3, 4] - elements 1, 2, 3
print(numbers[:3])     # [1, 2, 3] - first 3 elements
print(numbers[2:])     # [3, 4, 5] - from index 2 to end
print(numbers[::2])    # [1, 3, 5] - every 2nd element
print(numbers[::-1])   # [5, 4, 3, 2, 1] - reversed!

# List methods
numbers.append(6)           # Add to end: [1, 2, 3, 4, 5, 6]
numbers.insert(0, 0)        # Insert at index: [0, 1, 2, 3, 4, 5, 6]
numbers.extend([7, 8, 9])   # Add multiple elements
popped = numbers.pop()      # Remove and return last element
numbers.remove(3)           # Remove first occurrence of value
numbers.sort()              # Sort in place
numbers.reverse()           # Reverse in place

# List comprehensions - concise way to create lists (VERY Pythonic!)
squares = [x**2 for x in range(10)]  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With condition
evens = [x for x in range(20) if x % 2 == 0]  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

# Nested comprehension (like nested loops)
matrix = [[i*j for j in range(3)] for i in range(3)]
# [[0, 0, 0], [0, 1, 2], [0, 2, 4]]
```

### 3.2 Tuples (Immutable Lists)

```python
# Tuples cannot be modified after creation
point = (3, 4)
coordinates = (10.5, 20.3, 30.1)

# Often used for:
# 1. Returning multiple values from functions
def get_stats(numbers):
    return min(numbers), max(numbers), sum(numbers)/len(numbers)

minimum, maximum, average = get_stats([1, 2, 3, 4, 5])  # Tuple unpacking

# 2. Dictionary keys (lists can't be keys)
# 3. Data that shouldn't change

# Named tuples - like C structs
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
p = Point(3, 4)
print(p.x, p.y)  # 3 4
```

### 3.3 Dictionaries (Hash Maps)

```python
# Key-value pairs - like hash tables
empty_dict = {}
person = {
    "name": "Alice",
    "age": 30,
    "city": "NYC"
}

# Access
print(person["name"])        # Alice
print(person.get("email"))   # None (safe access, no error)
print(person.get("email", "N/A"))  # "N/A" (with default)

# Modify
person["age"] = 31           # Update value
person["email"] = "alice@example.com"  # Add new key

# Check existence
if "name" in person:
    print("Name exists")

# Iteration
for key in person:
    print(key, person[key])

for key, value in person.items():
    print(f"{key}: {value}")

# Dictionary comprehension
squares = {x: x**2 for x in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Common in ML for storing hyperparameters:
hyperparams = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "optimizer": "adam"
}
```

### 3.4 Sets (Unique Collections)

```python
# Unordered, unique elements
unique_numbers = {1, 2, 3, 2, 1}  # {1, 2, 3}

# Set operations (useful for ML data processing)
set_a = {1, 2, 3, 4}
set_b = {3, 4, 5, 6}

print(set_a | set_b)  # Union: {1, 2, 3, 4, 5, 6}
print(set_a & set_b)  # Intersection: {3, 4}
print(set_a - set_b)  # Difference: {1, 2}

# Remove duplicates from list
original = [1, 2, 2, 3, 3, 3, 4]
unique = list(set(original))  # [1, 2, 3, 4]
```

---

## 4. Object-Oriented Python

### 4.1 Classes Basics

```python
class Dog:
    # Class variable (shared by all instances)
    species = "Canis familiaris"
    
    # Constructor (like __init__ in Python, not Dog() like C++)
    def __init__(self, name, age):
        # Instance variables
        self.name = name
        self.age = age
    
    # Instance method (always has 'self' as first parameter)
    def bark(self):
        return f"{self.name} says Woof!"
    
    # String representation
    def __str__(self):
        return f"{self.name}, {self.age} years old"
    
    # Representation for debugging
    def __repr__(self):
        return f"Dog('{self.name}', {self.age})"

# Creating instances
my_dog = Dog("Buddy", 3)
print(my_dog.bark())        # Buddy says Woof!
print(my_dog)               # Buddy, 3 years old
print(Dog.species)          # Canis familiaris
```

### 4.2 Inheritance

```python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        raise NotImplementedError("Subclass must implement")

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

# Polymorphism
animals = [Dog("Buddy"), Cat("Whiskers")]
for animal in animals:
    print(animal.speak())
```

### 4.3 Special Methods (Dunder Methods)

```python
class Vector:
    """A simple 2D vector class - useful for understanding NumPy later."""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    # Enable: v1 + v2
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    # Enable: v1 * scalar
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    
    # Enable: len(v)
    def __len__(self):
        return 2
    
    # Enable: v[0], v[1]
    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        raise IndexError("Index out of range")
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(1, 2)
v2 = Vector(3, 4)
print(v1 + v2)      # Vector(4, 6)
print(v1 * 3)       # Vector(3, 6)
```

### 4.4 Dataclasses (Python 3.7+) - Great for ML!

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    """Configuration for ML model - cleaner than regular class."""
    model_name: str
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    hidden_layers: List[int] = None
    dropout: Optional[float] = None
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 64]

# Automatic __init__, __repr__, __eq__ generated!
config = ModelConfig("MyNet", learning_rate=0.01)
print(config)
# ModelConfig(model_name='MyNet', learning_rate=0.01, batch_size=32, ...)
```

---

## 5. Functional Programming Features

### 5.1 Map, Filter, Reduce

```python
from functools import reduce

numbers = [1, 2, 3, 4, 5]

# Map - apply function to each element
squared = list(map(lambda x: x**2, numbers))  # [1, 4, 9, 16, 25]
# Pythonic alternative:
squared = [x**2 for x in numbers]

# Filter - keep elements that satisfy condition
evens = list(filter(lambda x: x % 2 == 0, numbers))  # [2, 4]
# Pythonic alternative:
evens = [x for x in numbers if x % 2 == 0]

# Reduce - combine elements into single value
product = reduce(lambda x, y: x * y, numbers)  # 120
```

### 5.2 Generators (Memory Efficient)

```python
# Generator function - yields values one at a time
def countdown(n):
    while n > 0:
        yield n
        n -= 1

for num in countdown(5):
    print(num)  # 5, 4, 3, 2, 1

# Generator expression - like list comprehension but lazy
squares_gen = (x**2 for x in range(1000000))  # No memory allocated yet!

# Important for ML: Processing large datasets
def read_large_file(file_path):
    """Read file line by line without loading entire file into memory."""
    with open(file_path, 'r') as f:
        for line in f:
            yield line.strip()
```

### 5.3 Decorators

```python
import time
from functools import wraps

# Decorator to measure execution time (useful for ML training)
def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer
def train_model(epochs):
    # Simulated training
    time.sleep(2)
    return "Model trained"

train_model(100)  # train_model took 2.0023 seconds
```

---

## 6. Essential Libraries for ML/AI

### Installation

```bash
# Create virtual environment (recommended!)
python -m venv ml_env
source ml_env/bin/activate  # Linux/Mac
# or: ml_env\Scripts\activate  # Windows

# Install essential packages
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# For deep learning (choose one or both)
pip install torch torchvision  # PyTorch
pip install tensorflow          # TensorFlow
```

### Library Overview

| Library | Purpose | Priority |
|---------|---------|----------|
| NumPy | Numerical computing, arrays | Essential |
| Pandas | Data manipulation, analysis | Essential |
| Matplotlib | Basic plotting | Essential |
| Seaborn | Statistical visualization | Important |
| Scikit-learn | Classical ML algorithms | Essential |
| PyTorch | Deep learning | Essential for DL |
| TensorFlow/Keras | Deep learning | Alternative to PyTorch |
| Jupyter | Interactive notebooks | Essential |
| SciPy | Scientific computing | Useful |
| Statsmodels | Statistical modeling | Useful |

---

## 7. NumPy - The Foundation

NumPy is the foundation of all scientific Python. It provides efficient array operations implemented in C.

### 7.1 Arrays Basics

```python
import numpy as np

# Creating arrays (like C arrays, but better!)
arr1d = np.array([1, 2, 3, 4, 5])
arr2d = np.array([[1, 2, 3], [4, 5, 6]])

# Array creation functions
zeros = np.zeros((3, 4))           # 3x4 matrix of zeros
ones = np.ones((2, 3))             # 2x3 matrix of ones
identity = np.eye(3)               # 3x3 identity matrix
random_arr = np.random.rand(3, 3)  # Random values [0, 1)
range_arr = np.arange(0, 10, 2)    # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)    # [0, 0.25, 0.5, 0.75, 1]

# Array properties
print(arr2d.shape)    # (2, 3) - dimensions
print(arr2d.dtype)    # int64 - data type
print(arr2d.ndim)     # 2 - number of dimensions
print(arr2d.size)     # 6 - total elements
```

### 7.2 Array Operations (Vectorized - No Loops!)

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# Element-wise operations (MUCH faster than Python loops)
print(a + b)      # [11, 22, 33, 44]
print(a * b)      # [10, 40, 90, 160]
print(a ** 2)     # [1, 4, 9, 16]
print(np.sqrt(a)) # [1, 1.41, 1.73, 2]
print(np.exp(a))  # [2.72, 7.39, 20.09, 54.60]
print(np.log(a))  # [0, 0.69, 1.10, 1.39]

# Comparison in C vs NumPy:
# C: for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
# NumPy: c = a + b   <- MUCH simpler!
```

### 7.3 Indexing and Slicing

```python
arr = np.arange(10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Basic slicing (same as Python lists)
print(arr[2:5])      # [2, 3, 4]
print(arr[::2])      # [0, 2, 4, 6, 8]

# Boolean indexing (VERY powerful for data filtering!)
print(arr[arr > 5])  # [6, 7, 8, 9]
print(arr[arr % 2 == 0])  # [0, 2, 4, 6, 8]

# 2D array indexing
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(matrix[0, 1])      # 2 (row 0, column 1)
print(matrix[1, :])      # [4, 5, 6] (entire row 1)
print(matrix[:, 2])      # [3, 6, 9] (entire column 2)
print(matrix[:2, 1:])    # [[2, 3], [5, 6]] (submatrix)

# Fancy indexing
indices = [0, 2]
print(matrix[indices])   # Rows 0 and 2
```

### 7.4 Array Manipulation

```python
arr = np.arange(12)

# Reshape (CRITICAL for ML - preparing data shapes)
matrix = arr.reshape(3, 4)   # 3 rows, 4 columns
matrix = arr.reshape(3, -1)  # -1 means "figure it out"

# Transpose
print(matrix.T)  # Swap rows and columns

# Flatten
flat = matrix.flatten()  # Back to 1D

# Concatenate
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

vertical = np.vstack([a, b])    # Stack vertically
horizontal = np.hstack([a, b])  # Stack horizontally

# Adding dimensions (for neural networks)
arr = np.array([1, 2, 3])
print(arr.shape)                # (3,)
print(arr[np.newaxis, :].shape) # (1, 3) - add row dimension
print(arr[:, np.newaxis].shape) # (3, 1) - add column dimension
```

### 7.5 Linear Algebra (Essential for ML!)

```python
# Matrix multiplication (the core of neural networks)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Element-wise multiplication
print(A * B)  # [[5, 12], [21, 32]]

# Matrix multiplication (dot product)
print(A @ B)           # [[19, 22], [43, 50]]
print(np.dot(A, B))    # Same result

# Linear algebra operations
print(np.linalg.det(A))     # Determinant: -2
print(np.linalg.inv(A))     # Inverse
eigenvalues, eigenvectors = np.linalg.eig(A)  # Eigendecomposition

# Solve linear system: Ax = b
b = np.array([1, 2])
x = np.linalg.solve(A, b)
```

### 7.6 Statistical Operations

```python
data = np.random.randn(1000)  # 1000 random numbers, normal distribution

print(np.mean(data))    # Mean
print(np.std(data))     # Standard deviation
print(np.var(data))     # Variance
print(np.median(data))  # Median
print(np.min(data))     # Minimum
print(np.max(data))     # Maximum
print(np.percentile(data, [25, 50, 75]))  # Quartiles

# Aggregation along axes
matrix = np.random.rand(3, 4)
print(np.sum(matrix, axis=0))   # Sum each column
print(np.mean(matrix, axis=1))  # Mean of each row
```

### 7.7 Broadcasting

```python
# NumPy automatically expands arrays for operations
matrix = np.ones((3, 4))
row = np.array([1, 2, 3, 4])

# Broadcasting: row is "stretched" to match matrix shape
result = matrix + row  # Each row gets [1, 2, 3, 4] added

# This is how you normalize data:
data = np.random.rand(100, 5)  # 100 samples, 5 features
mean = data.mean(axis=0)       # Mean of each feature
std = data.std(axis=0)         # Std of each feature
normalized = (data - mean) / std  # Broadcasting handles dimensions!
```

---

## 8. Pandas - Data Manipulation

Pandas is built on NumPy and provides high-level data structures for data analysis.

### 8.1 Series and DataFrame

```python
import pandas as pd

# Series - 1D labeled array
s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
print(s['b'])  # 2

# DataFrame - 2D labeled structure (like a spreadsheet or SQL table)
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 30, 35, 28],
    'salary': [50000, 60000, 70000, 55000],
    'department': ['Engineering', 'Marketing', 'Engineering', 'Sales']
})

print(df)
#       name  age  salary   department
# 0    Alice   25   50000  Engineering
# 1      Bob   30   60000    Marketing
# 2  Charlie   35   70000  Engineering
# 3    Diana   28   55000        Sales
```

### 8.2 Reading/Writing Data

```python
# Read from CSV (most common in ML)
df = pd.read_csv('data.csv')

# Read from Excel
df = pd.read_excel('data.xlsx')

# Read from SQL
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM table', conn)

# Write to CSV
df.to_csv('output.csv', index=False)

# Read from URL
url = 'https://example.com/data.csv'
df = pd.read_csv(url)
```

### 8.3 Data Exploration

```python
# First look at data
print(df.head())        # First 5 rows
print(df.tail())        # Last 5 rows
print(df.shape)         # (rows, columns)
print(df.info())        # Data types, non-null counts
print(df.describe())    # Statistical summary

# Column operations
print(df.columns)       # Column names
print(df.dtypes)        # Data types
print(df['age'])        # Single column (Series)
print(df[['name', 'age']])  # Multiple columns (DataFrame)
```

### 8.4 Selection and Filtering

```python
# Row selection
print(df.iloc[0])        # First row (by integer position)
print(df.iloc[0:3])      # First 3 rows
print(df.loc[0])         # Row with index 0 (by label)

# Filtering (boolean indexing)
seniors = df[df['age'] > 30]           # Age over 30
engineers = df[df['department'] == 'Engineering']

# Multiple conditions
result = df[(df['age'] > 25) & (df['salary'] > 55000)]

# Query method (SQL-like syntax)
result = df.query('age > 25 and salary > 55000')
```

### 8.5 Data Manipulation

```python
# Add new column
df['bonus'] = df['salary'] * 0.1

# Apply function
df['age_group'] = df['age'].apply(lambda x: 'Young' if x < 30 else 'Senior')

# Rename columns
df = df.rename(columns={'name': 'full_name'})

# Drop columns/rows
df = df.drop('bonus', axis=1)  # Drop column
df = df.drop([0, 1], axis=0)   # Drop rows

# Handle missing values
df = df.fillna(0)              # Fill NaN with 0
df = df.fillna(df.mean())      # Fill with column mean
df = df.dropna()               # Drop rows with NaN

# Sort
df = df.sort_values('age', ascending=False)

# Reset index
df = df.reset_index(drop=True)
```

### 8.6 Grouping and Aggregation

```python
# Group by and aggregate
dept_stats = df.groupby('department').agg({
    'salary': ['mean', 'min', 'max'],
    'age': 'mean'
})

# Simple group by
avg_salary = df.groupby('department')['salary'].mean()

# Multiple aggregations
summary = df.groupby('department').agg(
    avg_salary=('salary', 'mean'),
    employee_count=('name', 'count'),
    max_age=('age', 'max')
)

# Pivot tables
pivot = df.pivot_table(
    values='salary',
    index='department',
    columns='age_group',
    aggfunc='mean'
)
```

### 8.7 Merging DataFrames

```python
# Like SQL joins
df1 = pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C']})
df2 = pd.DataFrame({'id': [2, 3, 4], 'score': [85, 90, 88]})

# Inner join (only matching rows)
merged = pd.merge(df1, df2, on='id', how='inner')

# Left join (all rows from left)
merged = pd.merge(df1, df2, on='id', how='left')

# Concatenate DataFrames
combined = pd.concat([df1, df1], ignore_index=True)
```

---

## 9. Matplotlib & Visualization

### 9.1 Basic Plots

```python
import matplotlib.pyplot as plt
import numpy as np

# Line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)', color='blue', linestyle='-', linewidth=2)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Sine Wave')
plt.legend()
plt.grid(True)
plt.savefig('plot.png', dpi=300)
plt.show()
```

### 9.2 Common Plot Types

```python
# Create sample data
np.random.seed(42)
x = np.random.randn(100)
y = 2 * x + np.random.randn(100)
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Scatter plot
axes[0, 0].scatter(x, y, alpha=0.6, c='blue')
axes[0, 0].set_title('Scatter Plot')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('Y')

# Bar plot
axes[0, 1].bar(categories, values, color='green')
axes[0, 1].set_title('Bar Plot')

# Histogram
axes[1, 0].hist(x, bins=20, color='orange', edgecolor='black')
axes[1, 0].set_title('Histogram')

# Box plot
data = [np.random.randn(100) for _ in range(4)]
axes[1, 1].boxplot(data, labels=categories)
axes[1, 1].set_title('Box Plot')

plt.tight_layout()
plt.show()
```

### 9.3 Seaborn for Statistical Visualization

```python
import seaborn as sns
import pandas as pd

# Load sample dataset
tips = sns.load_dataset('tips')

# Distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(data=tips, x='total_bill', hue='time', kde=True)
plt.title('Distribution of Total Bill by Time')
plt.show()

# Correlation heatmap (essential for feature analysis!)
plt.figure(figsize=(8, 6))
correlation = tips[['total_bill', 'tip', 'size']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Pair plot (visualize relationships between features)
sns.pairplot(tips, hue='time')
plt.show()
```

---

## 10. Scikit-Learn - Classical ML

### 10.1 Machine Learning Workflow

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# 1. Load/Create data
from sklearn.datasets import load_iris
data = load_iris()
X, y = data.data, data.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Preprocess (scale features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train data
X_test_scaled = scaler.transform(X_test)        # Apply to test data

# 4. Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 5. Predict
y_pred = model.predict(X_test_scaled)

# 6. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))
```

### 10.2 Common Algorithms

```python
# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Clustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA  # Dimensionality reduction

# Example: Random Forest
rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum depth
    random_state=42
)
rf.fit(X_train, y_train)
importance = rf.feature_importances_  # Feature importance scores
```

### 10.3 Model Selection & Validation

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

# Cross-validation
scores = cross_val_score(model, X, y, cv=5)
print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

# Hyperparameter tuning with Grid Search
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
best_model = grid_search.best_estimator_
```

### 10.4 Pipelines (Best Practice!)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Chain preprocessing and model together
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('classifier', SVC())
])

# Fit entire pipeline
pipeline.fit(X_train, y_train)

# Predict (preprocessing is automatic!)
predictions = pipeline.predict(X_test)
```

---

## 11. Deep Learning Frameworks

### 11.1 PyTorch Basics (Recommended)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Tensors (like NumPy arrays, but can run on GPU)
x = torch.tensor([1, 2, 3], dtype=torch.float32)
y = torch.randn(3, 4)  # Random tensor
z = torch.zeros(2, 3)  # Zeros tensor

# NumPy <-> PyTorch conversion
numpy_arr = x.numpy()
torch_tensor = torch.from_numpy(numpy_arr)

# GPU transfer
x = x.to(device)
```

### 11.2 Building a Neural Network

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Create model
model = NeuralNetwork(input_size=10, hidden_size=64, output_size=3)
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, dataloader, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
```

### 11.3 TensorFlow/Keras Alternative

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Sequential model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

---

## 12. Best Practices & Pythonic Code

### 12.1 Code Style (PEP 8)

```python
# Good Python style
import os  # Standard library first
import sys

import numpy as np  # Third-party libraries second
import pandas as pd

from my_module import my_function  # Local imports last

# Constants in UPPER_CASE
MAX_ITERATIONS = 1000
LEARNING_RATE = 0.001

# Functions in snake_case
def calculate_loss(predictions, targets):
    """Calculate mean squared error loss.
    
    Args:
        predictions: Model predictions
        targets: True values
    
    Returns:
        Mean squared error
    """
    return np.mean((predictions - targets) ** 2)

# Classes in PascalCase
class DataProcessor:
    """Process data for ML pipeline."""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self._cache = None  # Private attribute (convention)
```

### 12.2 Virtual Environments

```bash
# Create virtual environment
python -m venv ml_env

# Activate (Windows)
ml_env\Scripts\activate

# Activate (Linux/Mac)
source ml_env/bin/activate

# Install packages
pip install numpy pandas scikit-learn torch

# Save dependencies
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt

# Deactivate
deactivate
```

### 12.3 Jupyter Notebooks

```python
# Start Jupyter
jupyter notebook

# Or JupyterLab (recommended)
jupyter lab

# Useful magics in notebooks
%matplotlib inline  # Show plots inline
%timeit some_function()  # Time execution
%load_ext autoreload  # Auto-reload modules
%autoreload 2
```

### 12.4 Common Patterns in ML Code

```python
# Configuration dictionary
config = {
    'model_name': 'resnet50',
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Progress bars with tqdm
from tqdm import tqdm

for epoch in tqdm(range(config['epochs']), desc='Training'):
    for batch in tqdm(dataloader, desc='Batches', leave=False):
        # Training code
        pass

# Logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Training started")
logger.warning("Low GPU memory")
logger.error("Training failed")

# Seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

---

## 13. Learning Roadmap

### Phase 1: Python Fundamentals (1-2 weeks)
- [ ] Variables, data types, operators
- [ ] Control flow (if, for, while)
- [ ] Functions and modules
- [ ] Data structures (lists, dicts, sets, tuples)
- [ ] File I/O
- [ ] Exception handling
- [ ] OOP basics

### Phase 2: Scientific Python (2-3 weeks)
- [ ] NumPy: Arrays, operations, broadcasting
- [ ] Pandas: DataFrames, data manipulation
- [ ] Matplotlib: Basic plotting
- [ ] Jupyter notebooks

### Phase 3: Machine Learning (4-6 weeks)
- [ ] ML concepts: supervised vs unsupervised
- [ ] Scikit-learn workflow
- [ ] Classification algorithms
- [ ] Regression algorithms
- [ ] Model evaluation and validation
- [ ] Feature engineering
- [ ] Hyperparameter tuning

### Phase 4: Deep Learning (4-6 weeks)
- [ ] Neural network basics
- [ ] PyTorch or TensorFlow
- [ ] CNNs for computer vision
- [ ] RNNs/Transformers for NLP
- [ ] Transfer learning
- [ ] GPU training

### Practice Projects
1. **Beginner**: Iris flower classification
2. **Intermediate**: House price prediction
3. **Intermediate**: Sentiment analysis
4. **Advanced**: Image classification with CNNs
5. **Advanced**: Text generation with Transformers

---

## Quick Reference

### NumPy Cheat Sheet
```python
np.array([1,2,3])     # Create array
np.zeros((m,n))       # m×n zeros
np.ones((m,n))        # m×n ones
np.eye(n)             # n×n identity
np.random.rand(m,n)   # Random [0,1)
np.random.randn(m,n)  # Random normal
arr.reshape(m,n)      # Reshape
arr.T                 # Transpose
arr @ brr             # Matrix multiply
arr.mean(), .sum()    # Aggregations
arr[arr > 0]          # Boolean indexing
```

### Pandas Cheat Sheet
```python
pd.read_csv('file.csv')    # Load CSV
df.head(), df.tail()       # Preview
df.info(), df.describe()   # Info
df['col']                  # Select column
df[['c1','c2']]            # Select columns
df[df['col'] > x]          # Filter rows
df.groupby('col').mean()   # Group by
df.merge(df2, on='col')    # Join
df.fillna(0)               # Handle NaN
df.to_csv('out.csv')       # Save
```

### Scikit-learn Cheat Sheet
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
model = LogisticRegression().fit(X_train_scaled, y_train)
y_pred = model.predict(scaler.transform(X_test))
accuracy = accuracy_score(y_test, y_pred)
```

---

## Resources

### Official Documentation
- [Python](https://docs.python.org/3/)
- [NumPy](https://numpy.org/doc/)
- [Pandas](https://pandas.pydata.org/docs/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [PyTorch](https://pytorch.org/docs/)
- [TensorFlow](https://www.tensorflow.org/api_docs)

### Free Courses
- [Kaggle Learn](https://www.kaggle.com/learn)
- [fast.ai](https://www.fast.ai/)
- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course)

### Practice
- [Kaggle](https://www.kaggle.com/) - Competitions and datasets
- [LeetCode](https://leetcode.com/) - Python practice problems
- [Project Euler](https://projecteuler.net/) - Math/programming challenges

---

*Happy Learning! Remember: the best way to learn is by doing. Start coding!* 🚀

