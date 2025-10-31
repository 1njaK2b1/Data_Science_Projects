package ngrams;

import edu.princeton.cs.algs4.In;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

public class NGramMap {
    private TimeSeries timeSeriescount = new TimeSeries();
    private Map<String, TimeSeries> aMapOfWordAndTimeSeries = new HashMap<>();
    private static final int MIN_YEAR = 1400;
    private static final int MAX_YEAR = 2100;

    public NGramMap(String wordsFilename, String countsFilename) {
        In wordFileReader = new In(wordsFilename);
        In countFileReader = new In(countsFilename);
        wordFileReader(wordFileReader);
        int count = getLineNumberOfAFile(countsFilename);
        countFileReader(countFileReader, count);
    }


    public void countFileReader(In countFileReader, int numberOfLineInAFile) {
        int count = 0;
        while (count < numberOfLineInAFile) {
            String lineByLineOneLineFromCountFileReader = countFileReader.readLine();
            String[] anArrayWithALlTheLinesFromCountFileReader =
                    lineByLineOneLineFromCountFileReader.split(",");
            int oneOfTheYearIntFromCountFileReader =
                    Integer.parseInt(anArrayWithALlTheLinesFromCountFileReader[0]);
            double oneOfTheCountDoubleFromCountFileReader =
                    Double.parseDouble(anArrayWithALlTheLinesFromCountFileReader[1]);
            timeSeriescount.put(oneOfTheYearIntFromCountFileReader,
                    oneOfTheCountDoubleFromCountFileReader);
            count++;
        }
    }

    public void wordFileReader(In wordFileReader) {
        while (wordFileReader.hasNextLine()) {
            String lineByLineOneLineFromWordFileReader = wordFileReader.readLine();
            String[] anArrayWithALlTheLinesFromWordFileReader = lineByLineOneLineFromWordFileReader.split("\t");
            String word = anArrayWithALlTheLinesFromWordFileReader[0];
            int oneOfTheYearIntFromWordFileReader = Integer.parseInt(anArrayWithALlTheLinesFromWordFileReader[1]);
            double oneOfTheCountDoubleFromWordFileReader =
                    Double.parseDouble(anArrayWithALlTheLinesFromWordFileReader[2]);
            if (aMapOfWordAndTimeSeries.containsKey(word)) {
                aMapOfWordAndTimeSeries.get(word).put(oneOfTheYearIntFromWordFileReader,
                        oneOfTheCountDoubleFromWordFileReader);
            } else {
                TimeSeries ts = new TimeSeries();
                ts.put(oneOfTheYearIntFromWordFileReader, oneOfTheCountDoubleFromWordFileReader);
                aMapOfWordAndTimeSeries.put(word, ts);
            }
        }
    }

    public TimeSeries countHistory(String word, int startYear, int endYear) {
        TimeSeries aTimeSeries = new TimeSeries();
        return countHistoryHelper(aTimeSeries, this.aMapOfWordAndTimeSeries,
                word, startYear, endYear);

    }

    public TimeSeries countHistoryHelper(TimeSeries aTimeSeries, Map<String,
            TimeSeries> oneMapOfWordAndTimeSeries, String word,
                                         int startYear, int endYear) {
        if (oneMapOfWordAndTimeSeries.containsKey(word)) {
            TimeSeries alltheWordsInAMapOfWordAndTimeSeries = oneMapOfWordAndTimeSeries.get(word);
            for (int year: alltheWordsInAMapOfWordAndTimeSeries.keySet()) {
                if (year >= startYear && year <= endYear) {
                    aTimeSeries.put(year,
                            alltheWordsInAMapOfWordAndTimeSeries.get(year));
                }
            }
        }
        return aTimeSeries;
    }

    public TimeSeries countHistory(String word) {
        TimeSeries aTimeSeries = new TimeSeries();
        return anotherCountHistoryHelper(aTimeSeries,
                this.aMapOfWordAndTimeSeries, word);
    }

    public TimeSeries anotherCountHistoryHelper(TimeSeries aTimeSeries, Map<String,
            TimeSeries> oneMapOfWordAndTimeSeries, String word) {
        if (oneMapOfWordAndTimeSeries.containsKey(word)) {
            aTimeSeries = oneMapOfWordAndTimeSeries.get(word);
        }
        return aTimeSeries;
    }

    public TimeSeries totalCountHistory() {
        TimeSeries aTimeSeries = new TimeSeries();
        return totalCountHistoryHelper(aTimeSeries,
                this.timeSeriescount, timeSeriescount);
    }

    public TimeSeries totalCountHistoryHelper(TimeSeries aTimeSeries,
                                              TimeSeries thisTimeSeriescount, TimeSeries anotherTimeSeriescount) {
        for (int year: thisTimeSeriescount.keySet()) {
            aTimeSeries.put(year, anotherTimeSeriescount.get(year));
        }
        return aTimeSeries;
    }

    public TimeSeries weightHistory(String word, int startYear, int endYear) {
        TimeSeries aTimeSeries = countHistory(word, startYear, endYear);
        return weightHistoryHelper(this.timeSeriescount, aTimeSeries);
    }

    public TimeSeries weightHistoryHelper(TimeSeries thisTimeSeriescount,
                                          TimeSeries aTimeSeries) {
        if (aTimeSeries == null) {
            return null;
        }
        return aTimeSeries.dividedBy(thisTimeSeriescount);
    }



    public TimeSeries weightHistory(String word) {
        TimeSeries aTimeSeries = countHistory(word);
        return anotherWeigthHistoryhelper(this.timeSeriescount, aTimeSeries);
    }

    public TimeSeries anotherWeigthHistoryhelper(TimeSeries thisTimeSeriescount,
                                                 TimeSeries aTimeSeries) {
        if (aTimeSeries == null) {
            return null;
        }
        return aTimeSeries.dividedBy(thisTimeSeriescount);
    }

    
    public TimeSeries summedWeightHistory(Collection<String> words,
                                          int startYear, int endYear) {
        TimeSeries aTimeSeries = new TimeSeries();
        return summedWeightHistoryHelper(aTimeSeries, words,
                startYear, endYear);
    }

    public TimeSeries summedWeightHistoryHelper(TimeSeries aTimeSeries,
                                                Collection<String> words,
                                                int startYear, int endYear) {
        for (String word: words) {
            TimeSeries ts = weightHistory(word, startYear, endYear);
            if (ts != null) {
                aTimeSeries = aTimeSeries.plus(ts);
            }
        }
        return aTimeSeries;
    }

    
    public TimeSeries summedWeightHistory(Collection<String> words) {
        TimeSeries aTimeSeries = new TimeSeries();
        return anotherSummedWeightHistoryHelper(aTimeSeries, words);
    }

    public TimeSeries anotherSummedWeightHistoryHelper(TimeSeries aTimeSeries,
                                                       Collection<String> words) {
        for (String word : words) {
            TimeSeries ts = weightHistory(word);
            if (ts != null) {
                aTimeSeries = aTimeSeries.plus(ts);
            }
        }
        return aTimeSeries;
    }


    public static int getLineNumberOfAFile(String onefilePath) {
        In wordInput = new In(onefilePath);
        int countNumberOfLines = 0;
        while (wordInput.hasNextLine()) {
            wordInput.readLine();
            countNumberOfLines += 1;
        }
        return countNumberOfLines;

    }
}
