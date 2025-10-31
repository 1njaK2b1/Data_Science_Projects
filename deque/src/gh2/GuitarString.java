package gh2;

import deque.Deque61B;
import deque.ArrayDeque61B;

public class GuitarString {
    private static final int SR = 44100;      
    private static final double DECAY = .996;

    private Deque61B<Double> buffer = new ArrayDeque61B<>();


    public GuitarString(double frequency) {
        int capacity = (int) Math.round(SR / frequency);
        for (int i = 0; i < capacity; i++) {
            buffer.addLast(0.0);
        }
    }

    public void pluck() {
        for (int i = 0; i < buffer.size(); i++) {
            double r = Math.random() - 0.5;
            buffer.removeFirst();
            buffer.addLast(r);
        }
    }

    public void tic() {
        double newItem = ((buffer.get(0) + buffer.get(1)) / 2.0)  * DECAY;
        buffer.removeFirst();
        buffer.addLast(newItem);
    }

    public double sample() {
        return buffer.get(0);
    }
}
