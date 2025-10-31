package main;

import browser.NgordnetQueryHandler;
import ngrams.NGramMap;

public class AutograderBuddy {
    public static NgordnetQueryHandler getHyponymsHandler(
            String wordFile, String countFile,
            String synsetFile, String hyponymFile) {
        NGramMap ngrammap = new NGramMap(wordFile, countFile);
        WordNetFindHyponymsAndHypernyms wordnet =
                new WordNetFindHyponymsAndHypernyms(synsetFile, hyponymFile);
        return new HyponymsHandler(ngrammap, wordnet);
    }
}
