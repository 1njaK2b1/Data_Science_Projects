package deque;

import java.util.Iterator;
import java.util.List;
import java.util.ArrayList; // import the ArrayList class


public class LinkedListDeque61B<T> implements Deque61B<T> {
    public Iterator<T> iterator() {
        return new LinkedListIterator();
    }



    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null) {
            return false;
        }
        if (!(o instanceof Deque61B)) {
            return false;
        }
        Deque61B<T> ol = (Deque61B<T>) o;
        if (ol.size() != this.size()) {
            return false;
        }
        for (int i = 0; i < size; i++) {
            if (!(ol.get(i).equals(this.get(i)))) {
                return false;
            }
        }
        return true;
    }


    private class Node {
        private T item;
        private Node prev;
        private Node next;

        public Node(T x, Node p, Node n) {
            item = x;
            prev = p;
            next = n;
        }
    }


    private class LinkedListIterator implements Iterator<T> {
        private int posInt;

        private LinkedListIterator() {
            posInt = 0;
        }

        public boolean hasNext() {
            return posInt < size;
        }

        public T next() {
            T item = get(posInt);
            posInt += 1;
            return item;
        }
    }

    private Node sentinel;
    private int size;

    public LinkedListDeque61B() {
        sentinel = new Node(null, null, null);
        sentinel.next = sentinel;
        sentinel.prev = sentinel;
        size = 0;
    }

    @Override
    public void addFirst(T x) {
        sentinel.next = new Node(x, sentinel, sentinel.next);
        sentinel.next.next.prev = sentinel.next;
        size += 1;
    }

    @Override
    public void addLast(T x) {
        sentinel.prev = new Node(x, sentinel.prev, sentinel);
        sentinel.prev.prev.next = sentinel.prev;
        size += 1;
    }

    @Override
    public List<T> toList() {
        List<T> returnList = new ArrayList<>();
        Node current = sentinel.next;
        while (current != sentinel) {
            returnList.add(current.item);
            current = current.next;
        }
        return returnList;
    }

    @Override
    public boolean isEmpty() {
        if (size == 0) {
            return true;
        }
        return false;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public T removeFirst() {
        if (isEmpty()) {
            return null;
        }
        T rest = sentinel.next.item;
        sentinel.next = sentinel.next.next;
        sentinel.next.prev = sentinel;
        size--;
        return rest;
    }

    @Override
    public T removeLast() {
        if (isEmpty()) {
            return null;
        }
        T rest = sentinel.prev.item;
        sentinel.prev.prev.next = sentinel;
        sentinel.prev = sentinel.prev.prev;
        size--;
        return rest;
    }

    @Override
    public T get(int index) {
        int count = 0;
        Node point = sentinel;
        while (point.next != sentinel) {
            point = point.next;
            if (count == index) {
                return point.item;
            }
            count++;
        }
        return null;
    }

    @Override
    public T getRecursive(int index) {
        if (index >= size || index < 0) {
            return null;
        }
        int count = 0;
        Node point = sentinel.next;
        return getRecursiveHelper(index, count, point);
    }


    public T getRecursiveHelper(int index, int count, Node point) {
        if (index == count) {
            return point.item;
        }
        return getRecursiveHelper(index, count + 1, point.next);
    }

    @Override
    public String toString() {
        List<T> list = toList();
        return list.toString();
    }
}
