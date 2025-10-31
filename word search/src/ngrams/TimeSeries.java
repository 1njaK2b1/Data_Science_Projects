package ngrams;

import java.util.*;

public class TimeSeries extends TreeMap<Integer, Double> {

    public static final int MIN_YEAR = 1400;
    public static final int MAX_YEAR = 2100;

    public TimeSeries() {
        super();
    }

    public TimeSeries(TimeSeries ts, int startYear, int endYear) {
        super();
        oneTimeSeriesHelper(ts, startYear, endYear);
    }

    public void oneTimeSeriesHelper(TimeSeries timeSeries, int startYear, int endYear) {
        for (Integer oneKeyInTheSeries : timeSeries.keySet()) {
            if (oneKeyInTheSeries >= startYear && oneKeyInTheSeries <= endYear) {
                this.put(oneKeyInTheSeries, timeSeries.get(oneKeyInTheSeries));
            }
        }
    }

    public List<Integer> years() {
        List<Integer> aListOfYears = new ArrayList<>();
        return oneYearsHelper(aListOfYears);

    }

    public List<Integer> oneYearsHelper(List<Integer> aListOfYears) {
        for (Integer oneYear : this.keySet()) {
            aListOfYears.add(oneYear);
        }
        return aListOfYears;
    }
    
    public List<Double> data() {
        List<Double> aListOfData = new ArrayList<>();
        return oneDataHelper(aListOfData);
    }

    public List<Double> oneDataHelper(List<Double> aListOfData) {
        for (Double oneOfTheData: this.values()) {
            aListOfData.add(oneOfTheData);
        }
        return aListOfData;
    }

    public TimeSeries plus(TimeSeries ts) {
        TimeSeries sumOfTheTimeSeriesYear = new TimeSeries();
        return onePlusHelper(ts, sumOfTheTimeSeriesYear);
    }

    public TimeSeries onePlusHelper(TimeSeries inputTimeSeries, TimeSeries timeSeriesSum) {
        for (Integer oneYear: this.keySet()) {
            if (!inputTimeSeries.containsKey(oneYear)) {
                timeSeriesSum.put(oneYear, this.get(oneYear));
            } else {
                timeSeriesSum.put(oneYear, this.get(oneYear) + inputTimeSeries.get(oneYear));
            }
        }
        for (Integer oneYearInInputTimeSeries: inputTimeSeries.keySet()) {
            if (!this.containsKey(oneYearInInputTimeSeries)) {
                timeSeriesSum.put(oneYearInInputTimeSeries,
                        inputTimeSeries.get(oneYearInInputTimeSeries));
            }
        }
        return timeSeriesSum;
    }

    public TimeSeries dividedBy(TimeSeries ts) {
        TimeSeries dividedByNewTimeSeries = new TimeSeries();
        return oneDividedByHelper(ts, dividedByNewTimeSeries);
    }

    public TimeSeries oneDividedByHelper(TimeSeries timeSeries, TimeSeries dividedByNewTimeSeries) {
        for (Integer oneYear: this.keySet()) {
            if (timeSeries.containsKey(oneYear)) {
                dividedByNewTimeSeries.put(oneYear, this.get(oneYear) / timeSeries.get(oneYear));
            } else {
                throw new IllegalArgumentException();
            }
        }
        return dividedByNewTimeSeries;
    }
}
