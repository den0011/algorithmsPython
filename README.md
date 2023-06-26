## Различные алгоритмы на python

#### Алгоритмы сортировки
- Алгоритм сортировки пузырьком (Bubble Sort):

```python
def bubble_sort(list):
    for i in range(len(list)):
        for j in range(len(list) - 1):
            if list[j] > list[j + 1]:
                list[j], list[j + 1] = list[j + 1], list[j] # swap
    return list
```

- Алгоритм сортировки выбором (Selection Sort):

```python
def selection_sort(list):
    for i in range(len(list)):
        min_index = i
        for j in range(i + 1, len(list)):
            if list[min_index] > list[j]:
                min_index = j
        list[i], list[min_index] = list[min_index], list[i] # swap
    return list
```

- Алгоритм сортировки вставками (Insertion Sort):

```python
def insertion_sort(list):
    for i in range(1, len(list)):
        key = list[i]
        j = i - 1
        while j >=0 and key < list[j] :
            list[j+1] = list[j]
            j -= 1
        list[j+1] = key
    return list
```

- Алгоритм быстрой сортировка (Quick Sort):

```python
def partition(array, low, high):
    i = (low-1)
    pivot = array[high]

    for j in range(low, high):
        if array[j] <= pivot:
            i = i+1
            array[i], array[j] = array[j], array[i]
    array[i+1], array[high] = array[high], array[i+1]
    return (i+1)

def quick_sort(array, low, high):
    if len(array) == 1:
        return array
    if low < high:
        partition_index = partition(array, low, high)
        quick_sort(array, low, partition_index-1)
        quick_sort(array, partition_index+1, high)
```

- Алгоритм сортировки слиянием (Merge Sort):

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    # Разделение массива на две половины
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]
    
    # Рекурсивная сортировка обеих половин
    left_half = merge_sort(left_half)
    right_half = merge_sort(right_half)
    
    # Слияние отсортированных половин
    return merge(left_half, right_half)

def merge(left, right):
    result = []
    i = 0
    j = 0
    
    # Слияние элементов из обеих половин
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Добавление оставшихся элементов
    while i < len(left):
        result.append(left[i])
        i += 1
    while j < len(right):
        result.append(right[j])
        j += 1
    
    return result

```

#### Алгоритмы поиска

- Алгоритм линейного поиска (Linear Search):

```python
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

```

- Алгоритм двоичного поиска (Binary Search) в отсортированном массиве:

```python
def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

```

- Алгоритм поиска в глубину (Depth-First Search) в графе:

```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    
    visited.add(start)
    print(start, end=" ")

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

```

- Алгоритм поиска в ширину (Breadth-First Search) в графе:

```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        vertex = queue.popleft()
        print(vertex, end=" ")

        for neighbor in graph[vertex]:
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)

```

- Алгоритм A* (A-star) для поиска кратчайшего пути в графе:

```python
import heapq

def astar(graph, start, goal):
    open_set = [(0, start)]  # Открытое множество
    came_from = {}  # Сохранение пути от стартовой вершины
    g_score = {vertex: float('inf') for vertex in graph}  # Расстояние от стартовой вершины до текущей вершины
    g_score[start] = 0
    f_score = {vertex: float('inf') for vertex in graph}  # Оценка полного расстояния от стартовой вершины до целевой вершины
    f_score[start] = heuristic(start, goal)  # Пример эвристической функции

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in graph[current]:
            tentative_g_score = g_score[current] + graph[current][neighbor]

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None

def heuristic(vertex, goal):
    # Пример эвристической функции (Евклидово расстояние между вершинами)
    x1, y1 = vertex
    x2, y2 = goal
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

```

#### Алгоритмы сжатия

- Алгоритм Хаффмана (Huffman Coding):

```python
import heapq
from collections import defaultdict

def build_huffman_tree(data):
    # Подсчёт частоты символов
    freq = defaultdict(int)
    for char in data:
        freq[char] += 1

    # Построение очереди с приоритетом
    heap = [[weight, [char, ""]] for char, weight in freq.items()]
    heapq.heapify(heap)

    # Построение дерева Хаффмана
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    return heap[0][1:]

def compress_huffman(data):
    tree = build_huffman_tree(data)
    huffman_code = {char: code for char, code in tree}
    compressed_data = "".join(huffman_code[char] for char in data)
    return compressed_data, huffman_code

def decompress_huffman(compressed_data, huffman_code):
    reversed_code = {code: char for char, code in huffman_code.items()}
    current_code = ""
    decompressed_data = ""
    for bit in compressed_data:
        current_code += bit
        if current_code in reversed_code:
            char = reversed_code[current_code]
            decompressed_data += char
            current_code = ""
    return decompressed_data

```

- Алгоритм Lempel-Ziv-Welch (LZW Compression):

```python
def compress_lzw(data):
    dictionary = {chr(i): i for i in range(256)}
    current_code = 256
    compressed_data = []
    current_sequence = ""
    for char in data:
        current_sequence += char
        if current_sequence not in dictionary:
            compressed_data.append(dictionary[current_sequence[:-1]])
            dictionary[current_sequence] = current_code
            current_code += 1
            current_sequence = char
    compressed_data.append(dictionary[current_sequence])
    return compressed_data

def decompress_lzw(compressed_data):
    dictionary = {i: chr(i) for i in range(256)}
    current_code = 256
    decompressed_data = []
    previous_sequence = chr(compressed_data[0])
    decompressed_data.append(previous_sequence)
    for code in compressed_data[1:]:
        if code in dictionary:
            current_sequence = dictionary[code]
        elif code == current_code:
            current_sequence = previous_sequence + previous_sequence[0]
        else:
            raise ValueError("Invalid compressed data")
        decompressed_data.append(current_sequence)
        dictionary[current_code] = previous_sequence + current_sequence[0]
        current_code += 1
        previous_sequence = current_sequence
    return "".join(decompressed_data)

```

- Алгоритм RLE (Run-Length Encoding):

```python
def compress_rle(data):
    compressed_data = ""
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            compressed_data += str(count) + data[i - 1]
            count = 1
    compressed_data += str(count) + data[-1]
    return compressed_data

def decompress_rle(compressed_data):
    decompressed_data = ""
    for i in range(0, len(compressed_data), 2):
        count = int(compressed_data[i])
        char = compressed_data[i + 1]
        decompressed_data += count * char
    return decompressed_data

```

- Алгоритм BWT (Burrows-Wheeler Transform) с применением алгоритма MTF (Move-to-Front):

```python
def compress_lzss(data, window_size=12, lookahead_buffer_size=4):
    compressed_data = []
    i = 0
    while i < len(data):
        if i >= window_size:
            search_start = i - window_size
        else:
            search_start = 0
        search_end = min(i, len(data) - lookahead_buffer_size)
        best_match_length = 0
        best_match_distance = 0
        for j in range(search_end, search_start - 1, -1):
            match_length = 0
            while i + match_length < len(data) and data[j + match_length] == data[i + match_length] and match_length < lookahead_buffer_size:
                match_length += 1
            if match_length > best_match_length:
                best_match_length = match_length
                best_match_distance = i - j
        if best_match_length > 2:
            compressed_data.append((0, best_match_distance, best_match_length - 3))
            i += best_match_length
        else:
            compressed_data.append((1, data[i]))
            i += 1
    return compressed_data

def decompress_lzss(compressed_data, window_size=12):
    decompressed_data = []
    for token in compressed_data:
        if token[0] == 0:
            _, distance, length = token
            for _ in range(length + 3):
                decompressed_data.append(decompressed_data[-distance - 1])
        else:
            _, char = token
            decompressed_data.append(char)
    return decompressed_data

```

#### Парсинг json файла

Пример создания сложного JSON-файла:

```python
import json

data = {
    "employees": [
        {
            "id": 1,
            "name": "John",
            "position": "Manager",
            "salary": 50000,
            "skills": ["Python", "SQL", "Leadership"]
        },
        {
            "id": 2,
            "name": "Jane",
            "position": "Developer",
            "salary": 40000,
            "skills": ["JavaScript", "HTML", "CSS"]
        },
        {
            "id": 3,
            "name": "Mark",
            "position": "Designer",
            "salary": 45000,
            "skills": ["Photoshop", "Illustrator", "UI/UX"]
        }
    ],
    "company": "XYZ Corporation"
}

with open("data.json", "w") as file:
    json.dump(data, file, indent=4)

```

Пример парсинга сложного JSON-файла:

```python
import json

with open("data.json") as file:
    data = json.load(file)

company = data["company"]
employees = data["employees"]

print("Company:", company)
print("Employees:")
for employee in employees:
    print("ID:", employee["id"])
    print("Name:", employee["name"])
    print("Position:", employee["position"])
    print("Salary:", employee["salary"])
    print("Skills:", ", ".join(employee["skills"]))
    
```
#### Парсинг xml файла

Пример создания сложного XML-файла:

```python
import xml.etree.ElementTree as ET

root = ET.Element("company")

employee1 = ET.SubElement(root, "employee")
employee1.set("id", "1")

name1 = ET.SubElement(employee1, "name")
name1.text = "John"

position1 = ET.SubElement(employee1, "position")
position1.text = "Manager"

salary1 = ET.SubElement(employee1, "salary")
salary1.text = "50000"

skills1 = ET.SubElement(employee1, "skills")
ET.SubElement(skills1, "skill").text = "Python"
ET.SubElement(skills1, "skill").text = "SQL"
ET.SubElement(skills1, "skill").text = "Leadership"

employee2 = ET.SubElement(root, "employee")
employee2.set("id", "2")

name2 = ET.SubElement(employee2, "name")
name2.text = "Jane"

position2 = ET.SubElement(employee2, "position")
position2.text = "Developer"

salary2 = ET.SubElement(employee2, "salary")
salary2.text = "40000"

skills2 = ET.SubElement(employee2, "skills")
ET.SubElement(skills2, "skill").text = "JavaScript"
ET.SubElement(skills2, "skill").text = "HTML"
ET.SubElement(skills2, "skill").text = "CSS"

tree = ET.ElementTree(root)
tree.write("data.xml")

```

Пример парсинга сложного XML-файла:

```python
import xml.etree.ElementTree as ET

tree = ET.parse("data.xml")
root = tree.getroot()

company = root.tag
print("Company:", company)

employees = root.findall("employee")
print("Employees:")
for employee in employees:
    employee_id = employee.get("id")
    name = employee.find("name").text
    position = employee.find("position").text
    salary = employee.find("salary").text

    skills = employee.find("skills")
    skill_list = [skill.text for skill in skills.findall("skill")]

    print("ID:", employee_id)
    print("Name:", name)
    print("Position:", position)
    print("Salary:", salary)
    print("Skills:", ", ".join(skill_list))
    

```