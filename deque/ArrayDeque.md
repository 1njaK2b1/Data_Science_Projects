🧱 ArrayDeque

📘 Overview

In this project, you will implement a double-ended queue (Deque) using a resizable circular array.
This project builds upo LinkedListDeque, where the same interface was implemented using linked nodes.
Your new implementation should support efficient addition and removal of elements from both ends while managing memory effectively.

⸻

🧩 Goals
	•	Understand array-based data structure design.
	•	Implement a circular buffer for efficient front and back operations.
	•	Handle dynamic resizing to maintain space efficiency.
	•	Reinforce interface-based programming and code reusability.

⸻

⚙️ Implementation Details

🏗 Class Structure

You will implement:

public class ArrayDeque61B<T> implements Deque61B<T> {
    // Your fields and methods here
}

This class must provide the same API as your linked-list deque:
	•	addFirst(T item)
	•	addLast(T item)
	•	removeFirst()
	•	removeLast()
	•	get(int index)
	•	isEmpty()
	•	size()
	•	toList()

The getRecursive(int index) method may simply throw UnsupportedOperationException since recursion doesn’t make sense in this context.

⸻

🧮 Core Concepts

1. Circular Array Logic

Instead of shifting elements, maintain two indices:
	•	front – points to the first element
	•	back – points to the next insertion position at the end

Use modular arithmetic to wrap around the array boundaries:


index = (index + 1) % array.length;


2. Dynamic Resizing
	•	Expand: When the array is full, double its capacity before adding.
	•	Shrink: When usage falls below 25% and size ≥ 16, halve the capacity.

Resizing should preserve the logical order of elements.

3. Constant-Time Operations

All primary operations (addFirst, addLast, removeFirst, removeLast, get, isEmpty, size) must run in amortized O(1) time.

⸻

🧠 Additional Features

✅ Object Methods

Implement the following:
	•	iterator() — supports enhanced for-loops.
	•	equals(Object o) — compares deques element-by-element.
	•	toString() — returns a human-readable representation like [a, b, c].

🧰 Utility Class: Maximizer61B

Create a generic utility for finding the maximum element:


public static <T extends Comparable<T>> T max(Iterable<T> iterable);
public static <T> T max(Iterable<T> iterable, Comparator<T> comp);


If the iterable is empty, return null.

⸻

🪈 Optional Extension – Guitar Hero

You can use your deque implementation as a ring buffer to simulate guitar string vibrations (Karplus–Strong algorithm) in the Guitar Hero Lite extension:
	•	Implement GuitarString using your deque.
	•	Run GuitarHeroLite to play notes interactively.

⸻

📂 Example Project Structure

proj1b/
├── src/
│   ├── deque/
│   │   ├── ArrayDeque61B.java
│   │   ├── LinkedListDeque61B.java
│   │   ├── Deque61B.java
│   │   └── Maximizer61B.java
│   └── gh2/                 # Optional: Guitar Hero
└── tests/
    └── ArrayDeque61BTest.java



🧩 Key Takeaways
	•	Learn how to implement dynamic arrays efficiently.
	•	Explore data structure abstraction via interfaces.
	•	Understand how iterators, generics, and object methods improve usability.
	•	Strengthen debugging and testing practices with edge cases and resizing logic.


```python

```
