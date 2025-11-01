NGordnet (WordNet)
⸻

📘 Overview

In this project, you’ll extend your text-usage explorer from by incorporating semantic information from the WordNet dataset. You will implement a system that not only looks at historical word-usage trends (ngrams) but also explores hyponym/hypernym (is-a) relationships between words.  

Put simply:
    •    You already built a backend that handled word frequencies over time.
    •    Now you’ll build the part that handles “What words are kinds of this word or kinds of those words?” (hyponyms) and integrate that with your existing data.
    •    You’ll update or add handler(s) so that when users click the “Hyponyms” button in the front-end, your backend returns the appropriate list of words (and optionally further data).  

⸻

📁 Project Structure

Your repository should look something like:

proj2b/
├── data/
│   ├── ngrams/
│   └── wordnet/
├── src/
│   ├── ngrams/                 ← your code from (TimeSeries, NGramMap, etc)
│   ├── wordnet/                ← new package for WordNet handling
│   ├── main/                   ← Main class, handler registration
│   └── browser/                ← Browser / server glue
├── static/                     ← front-end html/js (provided)
└── tests/                      ← unit tests for hyponyms etc


⚙️ Implementation Details

🧬 WordNet Dataset

You’ll work with two primary data files:
    •    synsets.txt (or similar) — each line: synset ID, list of words, definition.  
    •    hyponyms.txt (or similar) — each line: synset ID, one or more hyponym synset IDs.  

Each synset groups words that are synonyms for one meaning; edges in the hyponym file represent “is-a” relationships (hyponym → hypernym). For example: the synset “jump, parachuting” is a hyponym of “descent”.  

🔍 Hyponyms Handler & Graph Representation

Your main task centers around a new handler (HyponymsHandler or similar) that supports queries for hyponyms of given words (or lists of words), possibly constrained by time-range and top-k parameters.  

Some focal details:
    •    You should design helper classes so that reading/parsing the WordNet files happens once (typically in constructor) — you cannot re‐read the files for each query.  
    •    You need to represent the WordNet relation graph (synsets as nodes, hyponym relations as directed edges). You may write your own simple graph class; do not import a full graph library.  
    •    When the handler receives a request (e.g., for word(s) “change”, startYear=1900, endYear=2000, k=10), you must:
    1.    Translate the word(s) to the corresponding synset(s).
    2.    Find all hyponyms (direct and indirect) of these synset(s).
    3.    Map those synset words to usage data (ngrams) for the given year-range using the code from 2A.
    4.    If k ≠ 0, pick the top k hyponym words based on frequency/count or another relevant metric.
    5.    Return the output in a format the front-end expects (likely JSON or a simple string list) in alphabetical order, no repeats.  

🎯 Example Behavior
    •    If the user inputs: word = "descent", startYear = any, endYear = any, k = 0
→ You return: [descent, jump, parachuting]
(assuming the dataset includes those hyponyms).  
    •    If multiple words are given (e.g., "gallery, bowl") you must handle the union of their hyponyms.  

⸻

🧩 Key Concepts
    •    Graph traversal (e.g., BFS or DFS) to find all reachable hyponym synsets.
    •    Integration of two datasets: usage data (from Project 2A) + semantic graph (WordNet).
    •    Efficient lookups and data structures: map word → synset IDs, synset ID → words, graph edges.
    •    API design & handler registration in your main class (Main.java or similar) so the front-end button “Hyponyms” triggers your code.
