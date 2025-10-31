package deque;

import java.util.List;

public interface Deque61B<T> extends Iterable<T> {
    void addFirst(T x);


    void addLast(T x);

    List<T> toList();


    boolean isEmpty();


    int size();

    T removeFirst();

    T removeLast();

    T get(int index);

    T getRecursive(int index);
}
