package main;

import edu.princeton.cs.algs4.In;

import java.util.*;

public class FindHypernymAndHyponymGraph {
    private List<List<String>> nodesListListString;
    private HashMap<Integer, List<Integer>> wordnetHashMapIntegerListInteger;
    private void parseSynsets(String synsetsInputs) {
        int lineNumber = getLineNumbersInAFile(synsetsInputs);
        nodesListListString = new ArrayList<>(lineNumber);
        for (int i = 0; i < lineNumber; i++) {
            nodesListListString.add(new ArrayList<>());
        }

        In synsetsReader = new In(synsetsInputs);
        parseSynsetsHelper(lineNumber,
                nodesListListString, synsetsReader);
    }

    public FindHypernymAndHyponymGraph(String hyponymsInput, String synsetsInput) {
        parseSynsets(synsetsInput);
        parseHyponyms(hyponymsInput);
    }

    public static int getLineNumbersInAFile(String filePath) {
        In wordInput = new In(filePath);
        int countNumberOfLines = 0;
        while(wordInput.hasNextLine()) {
            wordInput.readLine();
            countNumberOfLines += 1;
        }
        return countNumberOfLines;

    }

    private List<List<String>> parseSynsetsHelper(int lineNumberInTheFile,
                                               List<List<String>> nodeList, In synsetsReader) {
        for (int i = 0; i < lineNumberInTheFile; i++) {
            String oneLineInTheSynsets = synsetsReader.readLine();
            String[] words = oneLineInTheSynsets.split(",")[1].split(" ");
            for (String word: words) {
                nodeList.get(i).add(word);
            }
        }
        return nodeList;
    }



    public List<Integer> allSynsetParentFound(String onewordstring) {
        List<Integer> aListOfAllchildrenInnodesListListString = new ArrayList<>();

        return findAllParentOfSynsetHelper(onewordstring,
                aListOfAllchildrenInnodesListListString);
    }

    public List<Integer> findAllParentOfSynsetHelper(String onewordstring,
                                                     List<Integer> aListOfAllchildren) {
        for (int i = 0; i < nodesListListString.size(); i++) {
            List<String> currentListOfNodes =
                    nodesListListString.get(i);
            if (currentListOfNodes != null
                    && currentListOfNodes.contains(onewordstring)) {
                aListOfAllchildren.add(i);
            }
        }
        return aListOfAllchildren;

    }

    private void parseHyponyms(String hyponymsInput) {
        this.wordnetHashMapIntegerListInteger = new HashMap<>();

        In hyponymsreader = new In(hyponymsInput);
        String oneLineHyponymsreaderRead = "";
        parseHyponymsHelper(hyponymsreader,
                oneLineHyponymsreaderRead,
                this.wordnetHashMapIntegerListInteger);
    }

    private HashMap<Integer, List<Integer>> parseHyponymsHelper(In hyponymsreader,
                                                                String oneLineHyponymsreaderRead,
                                                                HashMap<Integer, List<Integer>> wordnetHashMap) {
        while (hyponymsreader.hasNextLine()) {
            oneLineHyponymsreaderRead =
                    hyponymsreader.readLine();
            String[] aListOfItemFromHypernymsreader =
                    oneLineHyponymsreaderRead.split(",");
            int parentNode =
                    Integer.parseInt(aListOfItemFromHypernymsreader[0]);
            List<Integer> childrenNode =
                    new ArrayList<>();
            for (int i = 1; i < aListOfItemFromHypernymsreader.length; i++) {
                childrenNode
                        .add(Integer.valueOf(aListOfItemFromHypernymsreader[i]));
            }
            if (wordnetHashMap
                    .containsKey(parentNode)) {
                wordnetHashMap.get(parentNode)
                        .addAll(childrenNode);
            } else {
                wordnetHashMap
                        .put(parentNode,
                                childrenNode);
            }
        }

        return wordnetHashMap;

    }

    public Set<String> findAllHypernymsofInWordNet(String word) {
       List<Integer> synsetsInAListOfStringsFromListListString =
               allSynsetParentFound(word);
       Set<Integer> synsetsOfAllParentNodeOfASynsetAsASetOfInteger =
               new HashSet<>();

       return findHypernymsHelper(word,
                                synsetsInAListOfStringsFromListListString,
                                synsetsOfAllParentNodeOfASynsetAsASetOfInteger,
                                this.nodesListListString);
    }

    public Set<String> findHypernymsHelper(String word,
                                           List<Integer> synsetsList,
                                           Set<Integer> synsetsOfParentNode,
                                           List<List<String>> nodesList) {
        for (int aSynsetInteger: synsetsList) {
            Set<Integer> parentOfsynsets = parentsFound(aSynsetInteger);
            synsetsOfParentNode.addAll(parentOfsynsets);
        }
        Set<String> wordsOfAllSynsetsParentNode = new TreeSet<>();
        for (int synsetsParentNode : synsetsOfParentNode) {
            wordsOfAllSynsetsParentNode
                    .addAll(nodesList.get(synsetsParentNode));
        }
        return wordsOfAllSynsetsParentNode;
    }


    public Set<String> findHypernymsFindHypernymsInCommon(List<String> alistofstring) {
       if (alistofstring.isEmpty()) {
           return null;
       }
       int indexOfTheItemInTheWordList = 0;
       String word = alistofstring.get(0);
       indexOfTheItemInTheWordList++;
       Set<String> commonHypernymsAmongAllHyperymsOfAWord =
               findAllHypernymsofInWordNet(word);
       if (commonHypernymsAmongAllHyperymsOfAWord.isEmpty()) {
           return Collections.emptySet();
       }
       findHypernymsInCommonHelper(alistofstring,
               indexOfTheItemInTheWordList,
               commonHypernymsAmongAllHyperymsOfAWord);
       return commonHypernymsAmongAllHyperymsOfAWord;


    }

    public Set<String> findHypernymsInCommonHelper(List<String> alistofstring,
                                                   int indexOfTheItemInTheWordList,
                                                   Set<String> commonHypernymsAmongAllHyperymsOfAWord) {
        if (commonHypernymsAmongAllHyperymsOfAWord.isEmpty()) {
            return Collections.emptySet();
        }
        int listsize = alistofstring.size();
        while (indexOfTheItemInTheWordList < listsize) {
            String otherWordInThisList = alistofstring.get(indexOfTheItemInTheWordList);
            indexOfTheItemInTheWordList++;
            Set<String> theotherHypernymsIntheListOfWords =
                    findAllHypernymsofInWordNet(otherWordInThisList);
            commonHypernymsAmongAllHyperymsOfAWord
                    .retainAll(theotherHypernymsIntheListOfWords);
        }
        return commonHypernymsAmongAllHyperymsOfAWord;
    }

    public Set<Integer> parentsFound(Integer node) {
       Set<Integer> parent = new HashSet<>();
       parentBeingFoundHelper(node, parent);
       parent.add(node);
       return parent;
    }

    public void parentBeingFoundHelper(Integer node, Set<Integer> parent) {
       parent.add(node);
       parentFoundOneWord(wordnetHashMapIntegerListInteger, node, parent);
    }

    public void parentFoundOneWord(HashMap<Integer,
                                List<Integer>> wordnet,
                                 Integer child,
                                 Set<Integer> parent) {
       for (Map.Entry<Integer, List<Integer>> entry: wordnet.entrySet()) {
           if (entry.getValue().contains(child)) {
               parent.add(entry.getKey());
               parentFoundOneWord(wordnet, entry.getKey(), parent);
           }
       }
    }

    public List<Integer> findALlSynsetsListHelper(String awordstring,
                                                  List<Integer> integerList,
                                                  List<List<String>> nodesList) {
        int nodesListSize = nodesList.size();
        for (int i = 0; i < nodesListSize; i++) {
            List<String> oneNodeInTheList = nodesList.get(i);
            if (oneNodeInTheList != null && oneNodeInTheList.contains(awordstring)) {
                integerList.add(i);
            }
        }
        return integerList;
    }

    public List<Integer> allSynsetsFoundList(String awordstring) {
        List<Integer> asynsetlist = new ArrayList<>();
        asynsetlist = findALlSynsetsListHelper(awordstring, asynsetlist, nodesListListString);
        return asynsetlist;
    }



    public Set<String> findAllHyponymsOfOneWord(String oneWord) {
        List<Integer> wordSynsetsForHyponyms = allSynsetsFoundList(oneWord);
        Set<Integer> synsetsHashSet = new HashSet<>();

        return findHyponymsOfOneWordHelper(oneWord,
                                    wordSynsetsForHyponyms,
                                    synsetsHashSet,
                                    this.nodesListListString);
    }

    public Set<String> findHyponymsOfOneWordHelper(String oneWord,
                                                   List<Integer> wordSynsetsForHyponyms,
                                                   Set<Integer> synsetsHashSet,
                                                   List<List<String>> nodesList) {
        for (int synsetHyponyms : wordSynsetsForHyponyms) {
            Set<Integer> childrenNodes = findAllChildrenNodes(synsetHyponyms);
            synsetsHashSet.addAll(childrenNodes);
        }
        Set<String> words = new TreeSet<>();
        for (int synset : synsetsHashSet) {
            words.addAll(nodesList.get(synset));
        }
        return words;
    }

    public void setOfChildrenFoundHelper(Set<Integer> childrenIntegerSet, Integer aIntegernode) {
        if (!wordnetHashMapIntegerListInteger.containsKey(aIntegernode)) {
            childrenIntegerSet.add(aIntegernode);
            return;
        }
        for (Integer childIntegernode : wordnetHashMapIntegerListInteger.get(aIntegernode)) {
            setOfChildrenFoundHelper(childrenIntegerSet, childIntegernode);
            childrenIntegerSet.add(childIntegernode);
        }
    }

    public Set<String> hyponymsFound(List<String> alistofstring) {
        if (alistofstring.isEmpty()) {
            return null;
        }

        int indexOfTheItemInTheWordList = 0;
        String word = alistofstring.get(0);
        indexOfTheItemInTheWordList++;
        Set<String> commonHyponymsAmongAllHyponymsOfAWord = findAllHyponymsOfOneWord(word);
        if (commonHyponymsAmongAllHyponymsOfAWord.isEmpty()) {
            return Collections.emptySet();
        }
        findHyponymsInCommonHelper(alistofstring,
                indexOfTheItemInTheWordList,
                commonHyponymsAmongAllHyponymsOfAWord);
        return commonHyponymsAmongAllHyponymsOfAWord;

    }

    public Set<Integer> findAllChildrenNodes(Integer aIntegernode) {
        Set<Integer> childrenIntegerSet = new HashSet<>();
        setOfChildrenFoundHelper(childrenIntegerSet, aIntegernode);
        childrenIntegerSet.add(aIntegernode);
        return childrenIntegerSet;
    }



    public Set<String> findHyponymsInCommonHelper(List<String> list,
                                                   int indexOfTheItemInTheWordList,
                                                   Set<String> commonHyponymsAmongAllHyponymsOfAWord) {
        if (commonHyponymsAmongAllHyponymsOfAWord.isEmpty()) {
            return Collections.emptySet();
        }
        int listsize = list.size();
        while (indexOfTheItemInTheWordList < listsize) {
            String otherWordInThisList = list.get(indexOfTheItemInTheWordList);
            indexOfTheItemInTheWordList++;
            Set<String> theotherHyponymsIntheListOfWords =
                    findAllHyponymsOfOneWord(otherWordInThisList);
            commonHyponymsAmongAllHyponymsOfAWord
                    .retainAll(theotherHyponymsIntheListOfWords);
        }
        return commonHyponymsAmongAllHyponymsOfAWord;
    }
}
