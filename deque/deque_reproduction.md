A deque (double-ended queue) is a linear data structure that allows insertion and deletion of elements from both ends — front and rear. It generalizes both stacks (LIFO) and queues (FIFO), offering flexible, efficient operations for real-time data processing.

Operation	Description	Time Complexity
append(x)	Add element x to the right end	O(1)
appendleft(x)	Add element x to the left end	O(1)
pop()	Remove and return the rightmost element	O(1)
popleft()	Remove and return the leftmost element	O(1)
extend(iterable)	Add multiple elements to the right end	O(k)
extendleft(iterable)	Add multiple elements to the left end	O(k)
rotate(n)	Rotate elements n steps right (negative for left)	O(k)

Advantages
	1.	Flexibility — Supports both stack and queue behaviors.
	2.	Efficiency — Constant-time insertion/removal from both ends.
	3.	Memory optimization — More efficient than using lists for queue-like operations (no O(n) shifting).
	4.	Thread-safe (in Python’s collections.deque) — Useful for multi-threaded applications.

⸻

Disadvantages
	1.	Limited random access — Access by index is O(n).
	2.	Not ideal for sorting or searching — Inefficient compared to arrays or linked lists for these tasks.
	3.	Slightly higher overhead — Compared to lists for pure append/pop from one end.

⸻

Use Cases
	•	Sliding window problems (e.g., moving averages, maximums)
	•	Task scheduling and job queues
	•	Palindrome checking
	•	Undo/Redo functionality
	•	BFS (Breadth-First Search) in graph algorithms
