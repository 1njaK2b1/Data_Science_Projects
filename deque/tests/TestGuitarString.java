/* Imports the required audio library from the
 * edu.princeton.cs.algs4 package. */
import edu.princeton.cs.algs4.StdAudio;
import org.junit.jupiter.api.Test;
import gh2.GuitarString;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

/** Tests the GuitarString class.
 *  @author Josh Hug
 */
public class TestGuitarString  {
    /**
     * Test method for plucking the A string and playing the sound.
     */
    @Test
    public void testPluckTheAString() {
        double CONCERT_A = 440.0;
        GuitarString aString = new GuitarString(CONCERT_A);
        aString.pluck();
        for (int i = 0; i < 50000; i += 1) {
            StdAudio.play(aString.sample());
            aString.tic();
        }
    }
    /**
     * Test method for sampling the guitar string.
     * Verifies that the sample is not 0 after plucking.
     * Verifies that successive samples do not change the state of the string.
     */
    @Test
    public void testSample() {
        GuitarString s = new GuitarString(100);
        assertThat(s.sample()).isEqualTo(0.0);
        assertThat(s.sample()).isEqualTo(0.0);
        assertThat(s.sample()).isEqualTo(0.0);
        s.pluck();

        double sample = s.sample();
        assertWithMessage("After plucking, your samples should not be 0").that(sample).isNotEqualTo(0);

        String errorMsg = "Sample should not change the state of your string";
        assertWithMessage(errorMsg).that(s.sample()).isWithin(0.0).of(sample);
        assertWithMessage(errorMsg).that(s.sample()).isWithin(0.0).of(sample);
    }

    /**
     * Test method for advancing the simulation one time step by performing one iteration of the Karplus-Strong algorithm.
     * Verifies that the sample changes after tic.
     */
    @Test
    public void testTic() {
        GuitarString s = new GuitarString(100);
        assertThat(s.sample()).isEqualTo(0.0);
        assertThat(s.sample()).isEqualTo(0.0);
        assertThat(s.sample()).isEqualTo(0.0);
        s.pluck();

        double sample1 = s.sample();
        assertWithMessage("After plucking, your samples should not be 0").that(sample1).isNotEqualTo(0);

        s.tic();
        String errorMsg = "After tic(), your samples should not stay the same";
        assertWithMessage(errorMsg).that(s.sample()).isNotEqualTo(sample1);
    }

    /**
     * Test method for checking the calculations of tic method.
     * Verifies that the tic method updates the sample correctly based on the Karplus-Strong algorithm.
     */
    @Test
    public void testTicCalculations() {
        GuitarString s = new GuitarString(11025);
        s.pluck();

        double s1 = s.sample();
        s.tic();
        double s2 = s.sample();
        s.tic();
        double s3 = s.sample();
        s.tic();
        double s4 = s.sample();

        s.tic();

        double s5 = s.sample();
        double expected = 0.996 * 0.5 * (s1 + s2);

        String errorMsg = "Wrong tic value. Try running the testTic method in TestGuitarString.java";
        assertWithMessage(errorMsg).that(s5).isWithin(0.001).of(expected);
    }
}
