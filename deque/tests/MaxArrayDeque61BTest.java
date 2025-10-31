import org.junit.jupiter.api.*;

import java.util.Comparator;
import java.util.NoSuchElementException;

import deque.MaxArrayDeque61B;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.assertEquals;

/**
 * Test class for MaxArrayDeque61B, containing tests for basic functionality and methods inherited from Deque61B.
 */
public class MaxArrayDeque61BTest {
    /**
     * Comparator for natural ordering of integers.
     */
    Comparator<Integer> naturalOrder = Comparator.naturalOrder();

    /**
     * Comparator for ordering strings based on length.
     */
    private static class StringLengthComparator implements Comparator<String> {
        /**
         * Compares two strings based on their lengths.
         *
         * @param a the first string to be compared
         * @param b the second string to be compared
         * @return a negative integer, zero, or a positive integer as the length of the first string is less than, equal to, or greater than the length of the second string.
         */
        public int compare(String a, String b) {
            return a.length() - b.length();
        }
    }

    /**
     * Test method for testing the basic functionality of the MaxArrayDeque61B class.
     */
    @Test
    public void basicTest() {
        MaxArrayDeque61B<String> mad = new MaxArrayDeque61B<>(new StringLengthComparator());
        mad.addFirst("");
        mad.addFirst("2");
        mad.addFirst("fury road");
        assertThat(mad.max()).isEqualTo("fury road");
    }

    /**
     * Test method for testing the add and get functionality of the MaxArrayDeque61B class.
     */
    @Test
    public void testAddAndGet() {
        MaxArrayDeque61B<Integer> deque = new MaxArrayDeque61B<>(naturalOrder);
        deque.addLast(1);
        deque.addLast(2);
        deque.addLast(3);

        assertEquals(1, (int) deque.get(0));
        assertEquals(2, (int) deque.get(1));
        assertEquals(3, (int) deque.get(2));
    }

    /**
     * Test method for testing the remove functionality of the MaxArrayDeque61B class.
     */
    @Test
    public void testRemove() {
        MaxArrayDeque61B<Integer> deque = new MaxArrayDeque61B<>(naturalOrder);
        deque.addLast(1);
        deque.addLast(2);
        deque.addLast(3);

        assertEquals(1, (int) deque.removeFirst());
        assertEquals(2, (int) deque.removeFirst());
        assertEquals(3, (int) deque.removeFirst());
    }

    /**
     * Test method for testing the size functionality of the MaxArrayDeque61B class.
     */
    @Test
    public void testSize() {
        MaxArrayDeque61B<Integer> deque = new MaxArrayDeque61B<>(naturalOrder);
        assertEquals(0, deque.size());

        deque.addLast(1);
        assertEquals(1, deque.size());

        deque.removeFirst();
        assertEquals(0, deque.size());
    }


    /**
     * Test method for testing the max functionality of the MaxArrayDeque61B class.
     */
    @Test
    public void testMax() {
        MaxArrayDeque61B<Integer> deque = new MaxArrayDeque61B<>(naturalOrder);
        deque.addLast(3);
        deque.addLast(1);
        deque.addLast(2);

        assertEquals(3, (int) deque.max());
    }
}
