# UniGuide AI

A Retrieval-Augmented Generation (RAG) scholarship information assistant for Tunisian students.

## Scope
- Destinations: Africa, Europe, Asia, North America, South America, Oceania
- Degree levels: Bachelor's, Master's, PhD
- Funding: fully funded, partially funded, tuition waivers, and other documented support

## Data principles
- Prefer official scholarship-provider, government, and university sources.
- Record the academic year and date accessed for every document.
- Verify Tunisian nationality eligibility separately for each opportunity.
- Use `Unknown` or `Not stated` when information is missing; do not guess.
- Keep historical scholarship calls clearly labelled as historical.

## Project structure
- `data/raw/`: original documents, organized by destination continent
- `data/processed/`: extracted and cleaned text
- `data/metadata/`: scholarship and document CSV metadata
- `evaluation/`: evaluation questions and expected answers
- `notebooks/`: experiments
- `src/`: RAG pipeline source code
- `app/`: user interface
- `vectorstore/`: local vector index
