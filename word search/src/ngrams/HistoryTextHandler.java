package ngrams;

import browser.NgordnetQuery;
import browser.NgordnetQueryHandler;
import ngrams.NGramMap;
import ngrams.TimeSeries;

import java.util.List;

public class HistoryTextHandler extends NgordnetQueryHandler {
    private NGramMap aNGramMap;
    public HistoryTextHandler(NGramMap map) {
        this.aNGramMap = map;
    }

    @Override
    public String handle(NgordnetQuery q) {
        String aOutputString = "";
        List<String> aListOfWords = q.words();
        int theStartYear = q.startYear();
        int theEndYear = q.endYear();
        return aHandleHelper(aOutputString, aListOfWords, theStartYear, theEndYear);
    }

    public String aHandleHelper(String aOutputString, List<String> aListOfWords,
                                int theStartYear, int theEndYear) {
        for (String oneOfTheWord: aListOfWords) {
            TimeSeries timeSeries = aNGramMap.weightHistory(oneOfTheWord, theStartYear, theEndYear);
            aOutputString += oneOfTheWord + ": " + timeSeries.toString() + "\n";
        }
        return aOutputString;
    }
}
