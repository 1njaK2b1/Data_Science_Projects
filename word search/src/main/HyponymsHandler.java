package main;

import browser.NgordnetQuery;
import browser.NgordnetQueryHandler;
import browser.NgordnetQueryType;
import ngrams.NGramMap;
import ngrams.TimeSeries;

import java.util.*;

public class HyponymsHandler extends NgordnetQueryHandler {
    private WordNetFindHyponymsAndHypernyms wordnet;
    private NGramMap ngrammap;
    @Override
    public String handle(NgordnetQuery q) {
        List<String> wordsList = q.words();
        int startingYear = q.startYear();
        NgordnetQueryType nqtype = q.type();
        int endingYear = q.endYear();
        int kvalue = q.k();
        Set<String> wordset = (nqtype == NgordnetQueryType.ANCESTORS)
                ? wordnet.hypernymsFound(wordsList) : wordnet.hyponymsFound(wordsList);
        if (wordsList.isEmpty()) {
            return " ";
        }
        if (kvalue != 0) {
            Map<String, Double> map = new HashMap<>();
            for (String str : wordset) {
                TimeSeries tseries = ngrammap.countHistory(str, startingYear, endingYear);
                if (tseries.size() != 0) {
                    double dbnum = 0;
                    for (double val : tseries.values()) {
                        dbnum += val;
                    }
                    map.put(str, dbnum);
                }
            }
            if (map.size() >= kvalue) {
                Set<String> set = new HashSet<>();
                for (int i = 0; i < kvalue; i++) {
                    double maxnum = 0;
                    String wordmaximum = "";
                    for (String word : wordset) {
                        if (map.containsKey(word)) {
                            if (map.get(word) > maxnum) {
                                maxnum = map.get(word);
                                wordmaximum = word;
                            }
                        }
                    }
                    set.add(wordmaximum);
                    map.remove(wordmaximum);
                }
                List<String> lst1 = new ArrayList<>(), lst2 = new ArrayList<>(), result = new ArrayList<>();
                for (String item : set) {
                    lst1.add(item);
                    lst2.add(item);
                }
                for (String item : lst2) {
                    String min = lst1.get(0);
                    for (String item1 : lst1) {
                        if (item1.compareTo(min) < 0) {
                            min = item1;
                        }
                    }
                    result.add(min);
                    lst1.remove(min);
                }
                return result.toString();

            } else {
                List<String> lst1 = new ArrayList<>(), lst2 = new ArrayList<>(), result = new ArrayList<>();
                for (String key : map.keySet()) {
                    lst1.add(key);
                    lst2.add(key);
                }
                for (String k2 : lst2) {
                    String min = lst1.get(0);
                    for (String key : lst1) {
                        if (key.compareTo(min) < 0) {
                            min = key;
                        }
                    }
                    result.add(min);
                    lst1.remove(min);
                }
                return result.toString();
            }
        } else {
            return wordset.toString();
        }
    }
    public HyponymsHandler(NGramMap ngrammap,
                           WordNetFindHyponymsAndHypernyms wordnet) {
        this.wordnet = wordnet;
        this.ngrammap = ngrammap;
    }
}
