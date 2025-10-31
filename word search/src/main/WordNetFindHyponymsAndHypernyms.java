package main;

import java.util.List;
import java.util.Set;

public class WordNetFindHyponymsAndHypernyms {
    private FindHypernymAndHyponymGraph graph;
    private String synsetFile;
    private String hyponymFile;

    public WordNetFindHyponymsAndHypernyms(String synsetFile,
                                           String hyponymFile) {
        this.synsetFile = synsetFile;
        this.hyponymFile = hyponymFile;
        graph = new FindHypernymAndHyponymGraph(hyponymFile, synsetFile);
    }

    public Set<String> hyponymsFound(List<String> words) {
        return graph.hyponymsFound(words);
    }

    public Set<String> hypernymsFound(List<String> words) {
        return graph.findHypernymsFindHypernymsInCommon(words);
    }
}
