#Section 1.0 Foundations

#Variables & Primitive Types: Integers, floats, strings, and booleans.
#Control Flow: if/else statements for logical branching.
#Loops: for and while loops to iterate through data collections.
#Functions: Definiton, parameters, arguments, and scope rules.


"""
Python Data Structures Roadmap: Section 1 Foundations
This script demonstrates Primitives, Control Flow, Loops, and Functions.
"""

# ==========================================
# CONCEPT 1: VARIABLES & PRIMITIVE TYPES
# ==========================================
print("--- CONCEPT 1: Variables & Primitives ---")

# Numeric primitives
item_count = 5  # Integer (int)
item_price = 19.99  # Floating-point number (float)

# Textual primitive
structure_name = "Stack"  # String (str)

# Logical primitive
is_empty = True  # Boolean (bool)

# Display values and their types dynamically
print(f"Structure: {structure_name} (Type: {type(structure_name)})")
print(f"Item Count: {item_count} (Type: {type(item_count)})")
print(f"Total Value: ${item_count * item_price}")
print(f"Is Container Empty? {is_empty}\n")

# ==========================================
# CONCEPT 2: CONTROL FLOW (IF / ELIF / ELSE)
# ==========================================
print("--- CONCEPT 2: Control Flow ---")

max_capacity = 3
current_size = 3

# Logical branching based on structural capacity constraints
if current_size == 0:
    print("Status Check: Underflow configuration. Container is empty.")
elif current_size >= max_capacity:
    print("Status Check: Overflow configuration. Container is full.")
else:
    print("Status Check: Operational configuration. Space is available.")
print("")

# ==========================================
# CONCEPT 3: LOOPS (FOR & WHILE)
# ==========================================
print("--- CONCEPT 3: Loops ---")
print("Executing a For Loop (Iterating over a range sequence):")
# For loops are ideal when the number of iterations is known
for step in range(1, 4):
    print(f" -> Inserting item node #{step} into our temporary array")

print("\nExecuting a While Loop (Iterating based on a conditional state):")
# While loops continue until a specific condition evaluates to False
current_node_index = 0
while current_node_index < max_capacity:
    print(f" -> Processing structural index reference: {current_node_index}")
    current_node_index += 1  # Crucial increment mutation step to avoid infinite looping
print("")

# ==========================================
# CONCEPT 4: FUNCTIONS & SCOPE RULES
# ==========================================
print("--- CONCEPT 4: Functions & Scope ---")

# Global scope variable
global_capacity_ceiling = 1000


def evaluate_structural_health(elements, threshold):
    """
    Calculates operational health based on element load density.
    Demonstrates parameters, return statements, and local scope.
    """
    # Local scope variables (only accessible inside this function block)
    load_ratio = elements / threshold
    is_safe = load_ratio <= 0.85

    # Utilizing our global variable inside the function read-only
    if threshold > global_capacity_ceiling:
        print("Warning: Threshold exceeds system ceiling limits.")

    return is_safe, load_ratio


# Invoking the function and destructuring its returned tuple values
safety_flag, current_load = evaluate_structural_health(elements=75, threshold=100)

print(f"Function Evaluation Returned -> Is Safe: {safety_flag}, Load Ratio: {current_load:.2%}")

#SECTION 2:Built-in Data Structures

"""
Python Data Structures Roadmap: Section 2 Built-in Data Structures
This script demonstrates Lists, Tuples, Dictionaries, Sets, and Collections.
"""
from collections import deque, namedtuple, Counter

# ==========================================
# CONCEPT 1: LISTS (Dynamic Arrays)
# ==========================================
print("--- CONCEPT 1: Lists (Mutable & Ordered) ---")

# Creating a list representing a simple process queue
data_list = ["Task A", "Task B", "Task C"]

# Indexing and slicing
print(f"First element: {data_list[0]}")
print(f"Last element [-1]: {data_list[-1]}")
print(f"Slice [0:2]: {data_list[0:2]}")  # Gets index 0 and 1

# Modifying elements (Lists are mutable)
data_list.append("Task D")        # Adds to the end: O(1)
data_list.insert(1, "Priority Task") # Inserts at index 1: O(n)
print(f"Updated list: {data_list}")

# Removing elements
removed_item = data_list.pop()    # Removes from the end: O(1)
print(f"Popped item: {removed_item}")
print(f"Final list: {data_list}\n")


# ==========================================
# CONCEPT 2: TUPLES (Fixed Records)
# ==========================================
print("--- CONCEPT 2: Tuples (Immutable & Ordered) ---")

# Tuples represent data that should not change (e.g., coordinates, configurations)
node_location = (4, 12)

print(f"X-Coordinate: {node_location[0]}")
print(f"Y-Coordinate: {node_location[1]}")

# Tuple unpacking
x, y = node_location
print(f"Unpacked variables -> x: {x}, y: {y}")

# Trying to mutate a tuple like: node_location[0] = 5 will raise a TypeError
print("Note: Tuples cannot be modified after creation.\n")


# ==========================================
# CONCEPT 3: DICTIONARIES (Hash Maps)
# ==========================================
print("--- CONCEPT 3: Dictionaries (Key-Value Pairs) ---")

# Dictionaries offer fast O(1) average lookups by key
user_profile = {
    "user_id": 101,
    "username": "coder_99",
    "role": "admin"
}

# Accessing and modifying values
print(f"Username lookup: {user_profile['username']}")
user_profile["role"] = "super_admin"  # Update value
user_profile["is_active"] = True      # Add new key-value pair

# Safe lookup using .get() to prevent KeyError if key doesn't exist
print(f"Login status: {user_profile.get('last_login', 'Never')}")

# Iterating over key-value pairs
for key, value in user_profile.items():
    print(f" -> {key}: {value}")
print("")


# ==========================================
# CONCEPT 4: SETS (Unique Collections)
# ==========================================
print("--- CONCEPT 4: Sets (Unordered & Unique) ---")

# Sets automatically discard duplicates and offer O(1) membership testing
active_session_ids = {1001, 1002, 1003, 1001}
print(f"Unique sessions (duplicates dropped): {active_session_ids}")

# Fast membership testing using 'in' keyword
print(f"Is 1002 active? {1002 in active_session_ids}")

# Set mathematical operations
admin_users = {"Alice", "Bob"}
moderators = {"Bob", "Charlie"}

print(f"Union (All staff): {admin_users.union(moderators)}")
print(f"Intersection (Both roles): {admin_users.intersection(moderators)}")
print("")


# ==========================================
# CONCEPT 5: COLLECTIONS MODULE (Specialized Types)
# ==========================================
print("--- CONCEPT 5: Collections Module ---")

# 1. deque: Optimized double-ended queue for fast O(1) appends and pops from both sides
queue = deque(["User1", "User2"])
queue.append("User3")         # Add to right side
queue.appendleft("VIP_User")  # Add to left side
print(f"Deque contents: {queue}")
queue.popleft()               # Remove from left side: O(1)
print(f"Deque after popleft: {queue}")

# 2. namedtuple: Creates tuple subclasses with named fields for cleaner code readability
Point3D = namedtuple('Point3D', ['x', 'y', 'z'])
pt = Point3D(x=10, y=20, z=30)
print(f"NamedTuple point coordinates: x={pt.x}, y={pt.y}, z={pt.z}")

# 3. Counter: A dictionary subclass designed specifically for counting items
word_list = ["apple", "banana", "apple", "cherry", "banana", "apple"]
word_counts = Counter(word_list)
print(f"Counter results: {word_counts}")
print(f"Most common item: {word_counts.most_common(1)}")

"""
Python Data Structures Roadmap: Section 3 Object-Oriented Programming (OOP)
This script demonstrates Classes, Objects, Attributes, Constructors, and Methods.
"""

# =====================================================================
# CONCEPT 1 & 3: CLASSES, OBJECTS, & THE CONSTRUCTOR (__init__)
# =====================================================================
print("--- CONCEPTS 1 & 3: Classes, Objects, and Constructors ---")


# A Class is a blueprint. Think of it as a custom data container type.
class SimpleNode:
    """Represents a basic single element container used in data structures."""

    # The __init__ method is the Constructor. It runs automatically when
    # we create a new instance object of this class.
    def __init__(self, incoming_value):
        # CONCEPT 2: ATTRIBUTES
        # Variables attached to a class instance are called attributes.
        self.data = incoming_value  # Holds the actual data payload
        self.next = None  # Intended to point to another node later


# Instantiating Objects (creating concrete instances from our blueprint)
node_a = SimpleNode(55)
node_b = SimpleNode(99)

print(f"Node A memory object created. Stored data payload: {node_a.data}")
print(f"Node B memory object created. Stored data payload: {node_b.data}")
print(f"Initial next pointer for Node A is currently empty: {node_a.next}\n")

# =====================================================================
# CONCEPT 2 & 4: METHODS AND THE 'self' KEYWORD
# =====================================================================
print("--- CONCEPTS 2 & 4: Instance Methods and the 'self' Keyword ---")


class SmartContainer:
    """A custom structure tracking a collection of items manually."""

    def __init__(self, label):
        self.label = label
        self.storage = []  # Internal array storage element

    # An Instance Method is a function attached inside a class template.
    # The 'self' argument represents the specific object calling the method.
    def insert_item(self, target_item):
        """Appends an item to this specific container instance."""
        self.storage.append(target_item)
        print(f" -> [{self.label}] Successfully archived item: {target_item}")

    def show_metrics(self):
        """Prints state specifications belonging to 'self' context."""
        item_count = len(self.storage)
        print(f" -> [{self.label}] Contains {item_count} items total: {self.storage}")


# Create two completely distinct data containers from the same blueprint
box_one = SmartContainer(label="Alpha Box")
box_two = SmartContainer(label="Beta Box")

# Invoking methods passes the respective object context implicitly into 'self'
box_one.insert_item("Data Package X")
box_one.insert_item("Data Package Y")

box_two.insert_item("System Log Z")

print("\nReading independent object states:")
box_one.show_metrics()  # 'self' targets box_one values
box_two.show_metrics()  # 'self' targets box_two values
print("")

# =====================================================================
# OOP IN ACTION: LINKING THE OBJECTS TOGETHER
# =====================================================================
print("--- PREVIEW: Linking Objects (The Foundation of Structures) ---")

# Let's use our SimpleNode class from above to build a basic chain link.
head_pointer = SimpleNode(10)  # Create first node
second_node = SimpleNode(20)  # Create second node

# Link the objects by setting the next attribute to point to the other object
head_pointer.next = second_node

print("Traversing our linked setup using pointers:")
print(f"Starting Node value: {head_pointer.data}")
print(f"Following the '.next' link pointer to the next value: {head_pointer.next.data}")

"""
Python Data Structures Roadmap: Section 4 Memory Model & Pointers
This script demonstrates Object References, Mutability, None Pointers, and Garbage Collection.
"""

# Helper class for demonstrations
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

# =====================================================================
# CONCEPT 1: OBJECT REFERENCES (Pointers in Python)
# =====================================================================
print("--- CONCEPT 1: Object References & Aliasing ---")

# In Python, variables are labels (pointers) pointing to memory addresses.
node_alpha = Node(100)
print(f"node_alpha references memory location: {id(node_alpha)}")

# Assignment (=) does NOT copy the object; it copies the reference label.
node_beta = node_alpha  # Both now point to the exact same memory object!
print(f"node_beta references memory location:  {id(node_beta)}")
print(f"Are they the exact same object in memory? {node_alpha is node_beta}")

# Modifying node_beta changes node_alpha because they are aliases of one object!
node_beta.val = 999
print(f"node_alpha.val reflects the change: {node_alpha.val}\n")


# =====================================================================
# CONCEPT 2: MUTABILITY VS IMMUTABILITY
# =====================================================================
print("--- CONCEPT 2: Mutability vs Immutability ---")

# 1. Immutable types (Integers, Strings, Tuples) cannot be altered in-place.
number_label = 10
old_id = id(number_label)
number_label += 1  # This creates a brand new integer object in memory!
print(f"Immutable: Value updated to {number_label}, but its memory location changed: {old_id != id(number_label)}")

# 2. Mutable types (Lists, Dicts, Custom Objects) can change inside the same memory slot.
data_list = [1, 2, 3]
list_id = id(data_list)
data_list.append(4)  # Modifies the same object in-place
print(f"Mutable: Value updated to {data_list}, and its memory location remains identical: {list_id == id(data_list)}\n")


# =====================================================================
# CONCEPT 3: THE 'None' CONSTANT (Null Pointer)
# =====================================================================
print("--- CONCEPT 3: The None Constant ---")

# None represents the absence of a value or an empty reference link.
current_node = Node(50)
print(f"Node initialized with value {current_node.val}.")
print(f"Its '.next' pointer starts as: {current_node.next}")

# We use 'is None' or 'is not None' to check for terminal dead-ends in data structures
if current_node.next is None:
    print(" -> Status: Tail reached. There are no trailing nodes in this sequence.")
print("")


# =====================================================================
# CONCEPT 4: AUTOMATIC GARBAGE COLLECTION
# =====================================================================
print("--- CONCEPT 4: Garbage Collection ---")

# Python deletes objects from memory automatically when their reference count hits 0.
temporary_node = Node(25)  # Reference count = 1
print(f"temporary_node allocated at: {id(temporary_node)}")

# Breaking the link by pointing the variable label somewhere else
temporary_node = None      # Reference count drops to 0!

print(" -> The original Node(25) object is now isolated in memory.")
print(" -> Python's Garbage Collector automatically reclaims that memory slice.")
print(" -> Safely prevents memory leaks without manual code management.\n")


# =====================================================================
# THE REWARD: MASTERING ELEMENT LINKING
# =====================================================================
print("--- WRAPPING UP: Linking Multiple Objects Safely ---")

# Now that you understand pointers, look at how easily we can link structural paths:
root = Node("First Item")
child = Node("Second Item")

root.next = child  # root.next now securely points to the child object location

print(f"Root item value: {root.val}")
print(f"Linked reference jump -> Next item value: {root.next.val}")

"""
Python Data Structures Roadmap: Section 5 Intermediate Custom Structures (Linear)
This script demonstrates Node Concepts, Linked Lists, Stacks, and Queues.
"""
from collections import deque

# =====================================================================
# CONCEPT 1 & 2: THE NODE CONCEPT & SINGLY LINKED LIST IMPLEMENTATION (CHAPTER 5)
# =====================================================================
print("--- CONCEPTS 1 & 2: Nodes & Singly Linked Lists ---")


class LinkedListNode:
    """The fundamental building block containing data and a forward pointer."""

    def __init__(self, data):
        self.data = data
        self.next = None  # Pointer to the next node sequence


class SinglyLinkedList:
    """A linear collection of data elements called nodes linked by pointers."""

    def __init__(self):
        self.head = None  # Entry point pointer to the start of the list

    def append(self, data):
        """Adds a new node to the very end of the list: O(n) time complexity."""
        new_node = LinkedListNode(data)

        # Scenario A: The list is completely empty
        if self.head is None:
            self.head = new_node
            return

        # Scenario B: Traverse from the entry head down to the final tail node
        current = self.head
        while current.next is not None:
            current = current.next

        # Link the final tail node pointer directly to our fresh node instance
        current.next = new_node

    def display(self):
        """Traverses and prints out the structural link chain sequentially."""
        elements = []
        current = self.head
        while current is not None:
            elements.append(str(current.data))
            current = current.next
        print("Linked List Chain: " + " -> ".join(elements) + " -> None")


# Testing our custom structural linked array layout
my_list = SinglyLinkedList()
my_list.append("Node #1")
my_list.append("Node #2")
my_list.append("Node #3")
my_list.display()
print("")

# =====================================================================
# CONCEPT 3: STACK IMPLEMENTATION (Last-In-First-Out / LIFO)
# =====================================================================
print("--- CONCEPT 3: Stacks (LIFO) ---")


class CustomStack:
    """A LIFO linear container optimized for end-element modifications."""

    def __init__(self):
        # We can implement a clean stack by wrapping a built-in dynamic list array
        self._storage = []

    def push(self, element):
        """Adds an item to the top of the stack: O(1) time complexity."""
        self._storage.append(element)

    def pop(self):
        """Removes and returns the top item: O(1) time complexity."""
        if self.is_empty():
            raise IndexError("Underflow Alert: Cannot pop elements out of an empty stack.")
        return self._storage.pop()

    def peek(self):
        """Inspects the topmost element without removing it from storage."""
        if self.is_empty():
            return None
        return self._storage[-1]

    def is_empty(self):
        return len(self._storage) == 0

    def show_stack(self):
        print(f"Stack View (Top -> Bottom): {self._storage[::-1]}")


# Testing the Stack operations
browser_history = CustomStack()
browser_history.push("homepage.com")
browser_history.push("dashboard.org")
browser_history.push("settings.net")

browser_history.show_stack()
print(f"User hits back button. Leaving: {browser_history.pop()}")
print(f"Current active visible page: {browser_history.peek()}\n")

# =====================================================================
# CONCEPT 4: QUEUE IMPLEMENTATION (First-In-First-Out / FIFO)
# =====================================================================
print("--- CONCEPT 4: Queues (FIFO) ---")


class CustomQueue:
    """A FIFO linear container designed for ordered step processing workflows."""

    def __init__(self):
        # Using a list causes slow O(n) element shifting when popping from index 0.
        # Instead, we wrap a collections.deque object to get native true O(1) shifts!
        self._storage = deque()

    def enqueue(self, element):
        """Adds an item to the back of the queue line: O(1) time complexity."""
        self._storage.append(element)

    def dequeue(self):
        """Removes and returns the front item: O(1) time complexity."""
        if self.is_empty():
            raise IndexError("Underflow Alert: Cannot dequeue elements out of an empty line.")
        return self._storage.popleft()  # Pops cleanly from the front edge layout directly

    def is_empty(self):
        return len(self._storage) == 0

    def show_queue(self):
        print(f"Queue Line (Front -> Back): {list(self._storage)}")


# Testing the Queue operations
printer_line = CustomQueue()
printer_line.enqueue("Invoice_PDF")
printer_line.enqueue("Photo_JPEG")
printer_line.enqueue("Report_DOCX")

printer_line.show_queue()
print(f"Printer processing and clearing out: {printer_line.dequeue()}")
printer_line.show_queue()

"""
Python Data Structures Roadmap: Section 6 Advanced Custom Structures (Non-Linear)
This script demonstrates Recursion, Binary Search Trees, and Graphs.
"""
#                             SECTION 6
# =====================================================================
# CONCEPT 1 & 2: RECURSION AND BINARY SEARCH TREES (BST)
# =====================================================================
print("--- CONCEPTS 1 & 2: Recursion & Binary Search Trees ---")

class TreeNode:
    """A hierarchical element block with data, left child, and right child pointers."""
    def __init__(self, key):
        self.val = key
        self.left = None   # Points to smaller child values
        self.right = None  # Points to larger child values


class BinarySearchTree:
    """A sorted branching structure optimized for O(log n) searches."""
    def __init__(self):
        self.root = None

    def insert(self, key):
        """Public method to add data to the tree framework layout."""
        if self.root is None:
            self.root = TreeNode(key)
        else:
            self._insert_recursive(self.root, key)

    def _insert_recursive(self, current_node, key):
        """Helper method using recursion to find the correct branch spot."""
        if key < current_node.val:
            # Go down the left branch path layout
            if current_node.left is None:
                current_node.left = TreeNode(key)
            else:
                # Recursive Call: Repeat same check logic on left child
                self._insert_recursive(current_node.left, key)
        else:
            # Go down the right branch path layout
            if current_node.right is None:
                current_node.right = TreeNode(key)
            else:
                # Recursive Call: Repeat same check logic on right child
                self._insert_recursive(current_node.right, key)

    def display_in_order(self):
        """Prints all elements in sorted ascending order."""
        elements = []
        self._in_order_recursive(self.root, elements)
        print(f"BST In-Order Sorting Traversal: {elements}")

    def _in_order_recursive(self, current_node, elements):
        """Left -> Root -> Right traversal pattern pattern."""
        if current_node is not None:
            self._in_order_recursive(current_node.left, elements)   # Visit Left
            elements.append(current_node.val)                      # Visit Root
            self._in_order_recursive(current_node.right, elements)  # Visit Right


# Testing our custom structural search tree
bst = BinarySearchTree()
# Insert unsorted elements
for item in:
    bst.insert(item)

# Output prints perfectly sorted data thanks to recursive tree properties!
bst.display_in_order()
print("")


# =====================================================================
# CONCEPT 3 & 4: GRAPHS & INTERCONNECTED NETWORKS
# =====================================================================
print("--- CONCEPTS 3 & 4: Graphs & Network Implementations ---")

class CustomGraph:
    """A network mapping connections (edges) between unique items (vertices)."""
    def __init__(self):
        # Using an Adjacency List layout (Dictionary of lists) for mapping
        self.adjacency_list = {}

    def add_vertex(self, vertex):
        """Adds a standalone item node endpoint to our network map."""
        if vertex not in self.adjacency_list:
            self.adjacency_list[vertex] = []

    def add_edge(self, vertex1, vertex2):
        """Creates a mutual bi-directional connection relationship link."""
        if vertex1 in self.adjacency_list and vertex2 in self.adjacency_list:
            self.adjacency_list[vertex1].append(vertex2)
            self.adjacency_list[vertex2].append(vertex1)

    def display_network(self):
        """Prints graph system layout map profiles clearly."""
        print("Graph Network Layout Map:")
        for vertex, connections in self.adjacency_list.items():
            print(f" Node [{vertex}] is directly connected to -> {connections}")


# Building a miniature social network graph map
social_map = CustomGraph()

# 1. Establish independent vertex data elements
social_map.add_vertex("Alice")
social_map.add_vertex("Bob")
social_map.add_vertex("Charlie")
social_map.add_vertex("David")

# 2. Wire connections together across our matrix layouts
social_map.add_edge("Alice", "Bob")
social_map.add_edge("Alice", "Charlie")
social_map.add_edge("Bob", "David")

social_map.display_network()
