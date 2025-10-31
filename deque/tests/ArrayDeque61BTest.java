import static org.junit.Assert.*;
import org.junit.Test;
import deque.ArrayDeque61B;

import java.util.Iterator;

public class ArrayDeque61BTest {

    /**
     * Test method for iterating through the elements of the deque.
     */
    @Test
    public void testIterator() {
        ArrayDeque61B<String> deque = new ArrayDeque61B<>();
        deque.addLast("front");
        deque.addLast("middle");
        deque.addLast("back");

        Iterator<String> iterator = deque.iterator();

        while (iterator.hasNext()) {
            System.out.println(iterator.next());
        }
    }
    /**
     * Test method for checking if two deques are equal.
     */

    @Test
    public void testEquals() {
        ArrayDeque61B<String> deque1 = new ArrayDeque61B<>();
        deque1.addLast("front");
        deque1.addLast("middle");
        deque1.addLast("back");

        ArrayDeque61B<String> deque2 = new ArrayDeque61B<>();
        deque2.addLast("front");
        deque2.addLast("middle");
        deque2.addLast("back");

        ArrayDeque61B<String> deque3 = new ArrayDeque61B<>();
        deque3.addLast("front");
        deque3.addLast("middle");

        assertTrue(deque1.equals(deque2));
        assertFalse(deque1.equals(deque3));
    }

    /**
     * Test method for checking the string representation of the deque.
     */
    @Test
    public void testToString() {
        ArrayDeque61B<String> deque = new ArrayDeque61B<>();
        deque.addLast("front");
        deque.addLast("middle");
        deque.addLast("back");

        assertEquals("[front, middle, back]", deque.toString());
    }

    /**
     * Test method for checking the string representation of the deque.
     */
    @Test
    public void testToStringRemoving() {
        ArrayDeque61B<String> deque = new ArrayDeque61B<>();
        deque.addLast("back");
        deque.addFirst("middle");
        deque.addFirst("front");
        deque.removeLast();

        assertEquals("[front, middle]", deque.toString());
    }

    @Test
    public void testToStringresizing() {
        ArrayDeque61B<String> deque = new ArrayDeque61B<>();
        deque.addLast("5");
        deque.addFirst("4");
        deque.addFirst("3");
        deque.addFirst("2");
        deque.addLast("6");
        deque.addLast("7");
        deque.addLast("8");
        deque.addLast("9");
        deque.addLast("10");
        deque.addFirst("1");
        deque.addFirst("0");
        deque.removeLast();
        deque.removeFirst();

        assertEquals("[1, 2, 3, 4, 5, 6, 7, 8, 9]", deque.toString());
    }
}
