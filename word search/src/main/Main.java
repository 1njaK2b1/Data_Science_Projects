package main;

import browser.NgordnetServer;
import ngrams.HistoryHandler;
import ngrams.HistoryTextHandler;
import ngrams.NGramMap;
import org.slf4j.LoggerFactory;

public class Main {
    static {
        LoggerFactory.getLogger(Main.class).info("\033[1;38mChanging text color to white");
    }
    public static void main(String[] args) {
        NgordnetServer hns = new NgordnetServer();


        String wordFile = "./data/ngrams/top_14377_words.csv";
        String countFile = "./data/ngrams/total_counts.csv";
        NGramMap ngrammap = new NGramMap(wordFile, countFile);
        String synsetFile = "./data/wordnet/synsets.txt";
        String hyponymFile = "./data/wordnet/hyponyms.txt";
        WordNetFindHyponymsAndHypernyms wn = new WordNetFindHyponymsAndHypernyms(synsetFile, hyponymFile);
        hns.startUp();
        hns.register("history", new HistoryHandler(ngrammap));
        hns.register("historytext", new HistoryTextHandler(ngrammap));
        hns.register("hyponyms", new HyponymsHandler(ngrammap, wn));

        System.out.println("Finished server startup! Visit http://localhost:4567/ngordnet.html");
    }
}
