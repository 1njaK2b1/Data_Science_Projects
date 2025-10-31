import deque.LinkedListDeque61B;
import org.junit.Test;

import java.util.Iterator;

import static org.junit.Assert.*;

public class LinkedListDeque61BTest {
    /**
     * Test method for iterating through the elements of the deque.
     */
    @Test
    public void testIterator() {
        LinkedListDeque61B<String> deque = new LinkedListDeque61B<>();
        deque.addLast("front");
        deque.addLast("middle");
        deque.addLast("back");

        // Get an iterator for the deque
        Iterator<String> iterator = deque.iterator();

        // Iterate through the elements and concatenate them to a StringBuilder
        StringBuilder sb = new StringBuilder();
        while (iterator.hasNext()) {
            sb.append(iterator.next()).append(" ");
        }

        // Check if the concatenated string matches the expected format
        assertEquals("front middle back ", sb.toString());
    }

    /**
     * Test method for checking if two deques are equal.
     */
    @Test
    public void testEquals() {
        LinkedListDeque61B<String> deque1 = new LinkedListDeque61B<>();
        deque1.addLast("front");
        deque1.addLast("middle");
        deque1.addLast("back");

        LinkedListDeque61B<String> deque2 = new LinkedListDeque61B<>();
        deque2.addLast("front");
        deque2.addLast("middle");
        deque2.addLast("back");

        LinkedListDeque61B<String> deque3 = new LinkedListDeque61B<>();
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
        LinkedListDeque61B<String> deque = new LinkedListDeque61B<>();
        deque.addLast("front");
        deque.addLast("middle");
        deque.addLast("back");

        assertEquals("[front, middle, back]", deque.toString());
    }
}
