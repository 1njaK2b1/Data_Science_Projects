package deque;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class ArrayDeque61B<T> implements Deque61B<T> {

    private T[] arrayDeque61B;
    private int arraysize;
    private int theLastIndexInTheArray;
    private static final int INITIAL_SIZE = 8;
    private int theFirstIndexInTheArray;

    private void increasingSizeResizing() {
        T[] IncreasedSizedArray = (T[]) new Object[arrayDeque61B.length * 2];
        arrayDeque61B = GenerateANewCopyOfAnArrayIncreasingSize(arrayDeque61B, IncreasedSizedArray);
        theFirstIndexInTheArray = 0;
        theLastIndexInTheArray = arraysize - 1;
    }

    private T[] GenerateANewCopyOfAnArrayIncreasingSize(T[] arrayDeque, T[] arrayDequeCopy) {
        int sizeOfTheOriginalArrayDeque = arrayDeque61B.length - theFirstIndexInTheArray;
        for (int i = 0; i < sizeOfTheOriginalArrayDeque; i++) {
            arrayDequeCopy[i] = arrayDeque[theFirstIndexInTheArray + i];
        }
        for (int i = sizeOfTheOriginalArrayDeque; i < arraysize; i++) {
            arrayDequeCopy[i] = arrayDeque[i - sizeOfTheOriginalArrayDeque];
        }

        return arrayDequeCopy;

    }

    @Override
    public void addLast(T x) {
        if (arraysize == 0) {
            theFirstIndexInTheArray = 0;
            arrayDeque61B[0] = x;
            theLastIndexInTheArray = 0;
            arraysize++;
            return;
        }
        if (arraysize == arrayDeque61B.length) {
            increasingSizeResizing();
        }
        if (theLastIndexInTheArray == arrayDeque61B.length - 1) {
            theLastIndexInTheArray = 0;
        } else {
            theLastIndexInTheArray++;
        }
        arrayDeque61B[theLastIndexInTheArray] = x;
        arraysize++;
    }

    @Override
    public void addFirst(T x) {
        if (arraysize == 0) {
            theLastIndexInTheArray = 0;
            arrayDeque61B[0] = x;
            theFirstIndexInTheArray = 0;
            arraysize++;
            return;
        }

        if (arraysize == arrayDeque61B.length) {
            increasingSizeResizing();
        }
        if (theFirstIndexInTheArray == 0) {
            theFirstIndexInTheArray = arrayDeque61B.length - 1;
        } else {
            theFirstIndexInTheArray--;
        }
        arrayDeque61B[theFirstIndexInTheArray] = x;
        arraysize++;

    }



    private void decreasingSizeResizing() {
        T[] DecreasedSizedArray = (T[]) new Object[arrayDeque61B.length / 2];
        arrayDeque61B = GenerateANewCopyOfAnArrayDecreasingSize(arrayDeque61B, DecreasedSizedArray);
        theFirstIndexInTheArray = 0;
        theLastIndexInTheArray = arraysize - 1;

    }

    private T[] GenerateANewCopyOfAnArrayDecreasingSize(T[] arrayDeque, T[] arrayDequeCopy) {
        if (theLastIndexInTheArray < theFirstIndexInTheArray) {
            int sizeOfTheOriginialArrayDeque = arrayDeque.length - theFirstIndexInTheArray;
            for (int i = 0; i < sizeOfTheOriginialArrayDeque; i++) {
                arrayDequeCopy[i] = arrayDeque[theFirstIndexInTheArray + i];
            }

            for (int i = 0; i < arraysize - sizeOfTheOriginialArrayDeque; i++) {
                arrayDequeCopy[i + sizeOfTheOriginialArrayDeque] = arrayDeque[i];
            }
        } else {
            for (int i = 0; i < arraysize; i++) {
                arrayDequeCopy[i] = arrayDeque[(theFirstIndexInTheArray + i) % arrayDeque.length];
            }
        }
        return arrayDequeCopy;

    }

    @Override
    public boolean isEmpty() {
        return arraysize == 0;
    }

    @Override
    public int size() {
        if (arraysize <= 0) {
            return 0;
        }
        return arraysize;
    }

    @Override
    public T removeFirst() {
        if (arraysize == 0) {
            return null;
        }
        T theNodeAtTheLastIndexToBeRemoved = arrayDeque61B[theFirstIndexInTheArray];
        arrayDeque61B[theFirstIndexInTheArray] = null;
        if (theFirstIndexInTheArray != arrayDeque61B.length - 1) {
            theFirstIndexInTheArray++;
        } else {
            theFirstIndexInTheArray = 0;
        }

        arraysize--;
        if (arraysize == 0) {
            theFirstIndexInTheArray = 0;
            theLastIndexInTheArray = 0;
        }
        if (arraysize < arrayDeque61B.length / 4) {
            decreasingSizeResizing();
        }

        return theNodeAtTheLastIndexToBeRemoved;
    }

    @Override
    public T removeLast() {
        if (arraysize == 0) {
            return null;
        }
        T theNodeAtTheLastIndexToBeRemoved = arrayDeque61B[theLastIndexInTheArray];
        arrayDeque61B[theLastIndexInTheArray] = null;
        if (theLastIndexInTheArray != 0) {
            theLastIndexInTheArray--;
        } else {
            theLastIndexInTheArray = arrayDeque61B.length - 1;
        }
        arraysize--;
        if (arraysize == 0) {
            theFirstIndexInTheArray = 0;
            theLastIndexInTheArray = 0;
        }
        if (arraysize < arrayDeque61B.length / 4) {
            decreasingSizeResizing();
        }

        return theNodeAtTheLastIndexToBeRemoved;
    }
    @Override
    public T get(int index) {
        int newIndexOfTheArrayDequeAdjusted = (theFirstIndexInTheArray + index) % arrayDeque61B.length;
        if (newIndexOfTheArrayDequeAdjusted < 0) {
            // If the positive index is negative, wrap around to the end of the array
            newIndexOfTheArrayDequeAdjusted += arrayDeque61B.length;
        }
        return arrayDeque61B[newIndexOfTheArrayDequeAdjusted];
    }

    @Override
    public List<T> toList() {
        List<T> aListToAddAllTheItemsInArrayDeque61B = new ArrayList<>();

        for (int i = 0; i < arraysize; i++) {
            aListToAddAllTheItemsInArrayDeque61B.add((T) arrayDeque61B[(theFirstIndexInTheArray + i) % arrayDeque61B.length]);
        }

        return aListToAddAllTheItemsInArrayDeque61B;
    }

    @Override
    public String toString() {
        List<T> list = toList();
        return list.toString();
    }

    @Override
    public Iterator<T> iterator() {
        return new IteratorOfTheArrayDeque61B();
    }

    @Override
    public T getRecursive(int index) {
        throw new UnsupportedOperationException("No need to implement getRecursive for proj 1b");
    }

    private class IteratorOfTheArrayDeque61B implements Iterator<T> {
        private int count;
        private IteratorOfTheArrayDeque61B() {
            count = 0;
        }
        public boolean hasNext() {
            return count < arraysize;
        }
        public T next() {
            T item = get(count);
            count += 1;
            return item;
        }
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
        Deque61B<T> oa = (Deque61B<T>) o;
        if (oa.size() != this.size()) {
            return false;
        }
        for (int i = 0; i < arraysize; i += 1) {
            if (!(oa.get(i).equals(this.get(i)))) {
                return false;
            }
        }
        return true;
    }
    public ArrayDeque61B() {
        theFirstIndexInTheArray = 0;
        theLastIndexInTheArray = 0;
        arrayDeque61B = (T[]) new Object[INITIAL_SIZE];
    }
}
