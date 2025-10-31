package browser;

import java.util.List;

public record NgordnetQuery(List<String> words,
        int startYear,
        int endYear,
        int k,
        NgordnetQueryType ngordnetQueryType) {
    public NgordnetQuery(List<String> words, int startYear, int endYear, int k) {
        this(words, startYear, endYear, k, null); // Assume null for queryType
    }

    public NgordnetQuery(List<String> words, int startYear, int endYear, int k, NgordnetQueryType ngordnetQueryType) {
        this.words = words;
        this.startYear = startYear;
        this.endYear = endYear;
        this.k = k;
        this.ngordnetQueryType = ngordnetQueryType;
    }

    public List<String> words() {
        return this.words;
    }

    public int startYear() {
        return this.startYear;
    }

    public int endYear() {
        return this.endYear;
    }

    public int k() {
        return this.k;
    }

    public NgordnetQueryType type() {
        return this.ngordnetQueryType;
    }
}
