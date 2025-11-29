This project is a fork of nanochat that I will be modifying for some LLM research.

Look over the files and get an understanding of the project and how everything works.

docs/partial_collapse_prd.md contains a PRD describing the idea of using top-K "partially collapsed" embeddings instead of fully collapsed (one hot) token.

docs/partial_collapse_tdd.md describes a technical approach on how to break the problem down and implement it.

Ultimately we will want the partial_collapse mode to be a flag we can pass into the run10.sh script such that we can train the base model WITHOUT using partial collapse (train as is in normal nanochat scripts).  Ideally this flag would just be used at certain points in the scripts to do the special partial collapse based training when it is specified and most of the code would be the same for base training vs partial_collapse training if possible.