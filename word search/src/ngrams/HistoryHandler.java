package ngrams;

import browser.NgordnetQuery;
import browser.NgordnetQueryHandler;

import plotting.Plotter;
import org.knowm.xchart.XYChart;

import java.util.ArrayList;
import java.util.List;

public class HistoryHandler extends NgordnetQueryHandler {
    private NGramMap oneNGramMap;
    public HistoryHandler(NGramMap map) {
        this.oneNGramMap = map;
    }
    @Override
    public String handle(NgordnetQuery q) {
        List<String> wordsOfNgordnetQuery = q.words();
        int startYearOfNgordnetQuery = q.startYear();
        int endYearOfNgordnetQuery = q.endYear();
        ArrayList<TimeSeries> aListOfTimeSeries = new ArrayList<>();
        ArrayList<String> aListOfLabels = new ArrayList<>();

        return oneHandleHelper(wordsOfNgordnetQuery, startYearOfNgordnetQuery,
                endYearOfNgordnetQuery, aListOfTimeSeries, aListOfLabels);
    }

    public String oneHandleHelper(List<String> wordsOfNgordnetQuery,
                                  int startYearOfNgordnetQuery, int endYearOfNgordnetQuery,
                                  ArrayList<TimeSeries> aListOfTimeSeries, ArrayList<String> aListOfLabels) {
        for (String oneWordInTheWordList: wordsOfNgordnetQuery) {
            TimeSeries oneTimeSeries = oneNGramMap.weightHistory(oneWordInTheWordList,
                    startYearOfNgordnetQuery, endYearOfNgordnetQuery);
            aListOfTimeSeries.add(oneTimeSeries);
            aListOfLabels.add(oneWordInTheWordList);
        }
        XYChart xyChartPlotGeneratingTimeSeriesCharts = Plotter.generateTimeSeriesChart(aListOfLabels, aListOfTimeSeries);
        String aEncodedImageAddress = Plotter.encodeChartAsString(xyChartPlotGeneratingTimeSeriesCharts);
        return aEncodedImageAddress;
    }
}
