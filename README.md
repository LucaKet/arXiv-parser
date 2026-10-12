# arXiv Parser
[_Note: this project was coded using AI (DeepSeek v4 flash/pro + the 'pi' coding agent)_.]

## Idea
Instead of viewing arXiv's paper each day listed by order of publication, 
I think a better idea would be to have some AI go through all the entries
published in a given set of categories and rank them based on the reader's 
interests.

Importantly the 'digested' list of entries contains exactly the same amount
of information as the original list, no entry is eliminated or modified.
To achieve this the only accepted output from the AI is a permutation of the 
number of entries, which are then re-arranged accordingly.

## How it works
The papers today on arXiv in the selected categories are:
1. 'example A' 
2. 'example B'
3. 'example C'

These entries (titles+authors+abstracts) get sent to an LLM with a user-generated
prompt (written in 'interests.md') describing his/her interests.
The LLM then goes over the prompt and the entries, and spits out a permutation 
of the listings, for example: [3,1,2].

This will generate a file 'date.md' containing the full entries (titles+authors+abstracts)
in the new order, in this case it would be:
1. 'example C'
2. 'example A'
3. 'example B'

The location of the generated file 'date.md' can be set in the 'dir' tag of `config.toml`:
dir = "path_to_output_directory".
The data sent to the LLM are obtained via an API call to arXiv's server.
The categories from which the entries are obtained are also set in `config.toml`, in the 
[arxiv] section.
In order to send it to the LLM for them to be ranked, an API key to one of the providers (OpenAI, 
Anthropic, Google, DeepSeek, ...) is necessary. Once the API key is obtained, set the 
corresponding values in the [llm] section of `config.toml`.

---
## Technical details

- **Date:** you specify the digest date you care about (`--date
  YYYY-MM-DD`); with no argument, today is assumed. Batch mode matches the
  digest email you received on the morning of `D` (the batch dated D-1 or
  D). For dates the listing page no longer covers, the script falls back to
  the API submission-date window (approximate) and says so.
- **Window (fallback):** in `mode = "api"`, or when the listing batch is
  unavailable for the requested date, the script queries
  `submittedDate:[D-2 .. D]` (UTC) — configurable via
  `arxiv.window_days_back`. This mode cannot see replacements or older
  cross-lists (the API has no announcement-date search field).
- **Ranking:** the LLM receives your `interests.md` and returns only a JSON
  array of indices (a permutation) — never rewritten content. Digests up to
  `ranker.two_pass_threshold` papers are ranked in one call with full
  abstracts; larger digests use two passes: a cheap ranking of everything by
  title/authors/categories, then a full-abstract refinement of the top
  `ranker.refine_top` papers. Invalid permutations trigger one retry, then
  fall back to the original order. Speed over precision by design.
- **Output:** `digests/yyyy-mm-dd.md`, one file per queried date, each entry
  keeping title, authors, arXiv id/link, categories, and the full abstract.
  Titles and abstracts are hard-wrapped at 80 columns and entries are
  separated by horizontal rules.

## Requirements

- Python ≥ 3.11 
- An LLM API key

## Setup

1. Copy `config.example.toml` to `config.toml` and edit it:
   - `arxiv.categories`: the categories you subscribe to (your arXiv account
     page lists them; the [category taxonomy](https://arxiv.org/category_taxonomy)
     has the identifiers, e.g. `cond-mat.mes-hall`).
   - `output.dir`: where the daily files go.
2. Copy `interests.example.md` to `interests.md` and edit it with the topics, keywords, and methods you care
   about. It is read fresh on every run — no other step needed after edits.
3. (Optional) Enable LLM ranking:
   - Set `llm.provider = "openai-compatible"` — this works with OpenAI,
     Gemini (via Google's OpenAI-compatible endpoint), Groq, DeepSeek,
     Ollama, LM Studio, and most other providers.
   - Set `llm.model`, and optionally `llm.fallback_models` — models tried
     in order when the primary fails (e.g. free-tier "high demand"
     errors). Example for the Gemini free tier:

     ```toml
     [llm]
     provider = "openai-compatible"
     model = "gemini-3.8-flash"
     fallback_models = ["gemini-3.7-flash", "gemini-3.5-flash-lite"]
     base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
     api_key_env = "GEMINI_API_KEY"
     ```

   - Export the API key in the environment variable named in
     `llm.api_key_env` (e.g. `export GEMINI_API_KEY="..."` in your shell
     profile). 
   - `[ranker]` controls ranking behavior: `temperature`, `max_retries`,
     and the two-pass settings `two_pass_threshold` and `refine_top`.

## Usage

```bash
# Today's digest (exact announcement batch)
python3 -m arxiv_parser

# A specific date, e.g. replaying a past digest
python3 -m arxiv_parser --date 2026-10-02

# Fetch without ranking (even if an LLM is configured)
python3 -m arxiv_parser --no-rank

# Check what would be written without writing it
python3 -m arxiv_parser --dry-run
```

Example output:

```
  exact announcement batch [cond-mat.mes-hall, cond-mat.str-el,
    cond-mat.supr-con, cond-mat.stat-mech, cond-mat.dis-nn]
    (Showing new listings for Friday, 2 October 2026), 151 papers
  ranking with openai-compatible (gemini-3.8-flash) -> ...
    first pass: ranking all 151 papers by title/authors ...
    second pass: refining top 30 with full abstracts ...
    done in 15.1s
Wrote digests/2026-10-03.md (151 papers)
```

The resulting file lists papers from most to least interesting, each with
full metadata and abstract. Re-running the same date overwrites that day's
file — useful when you refine `interests.md` and want a better ordering.

## Notes and limits

- **Batch mode** (`arxiv.mode = "auto"`, default) reproduces your digest
  email exactly: new submissions, cross-lists, and replacements, in email
  order. It uses the listing page as the index and enriches through the
  API; if the listing page is unreachable it falls back to the window mode
  and says so.
- **Fallback window mode** (`arxiv.mode = "api"`) can only search by
  *submission* date, so replacement submissions and cross-lists of papers
  submitted before the window are not included — the API simply does not
  expose announcement-batch membership.
- The listing page only covers the current batch, so exact reproduction is
  available for recent dates; older `--date` values fall back to the
  window mode automatically.
- Weekend runs are handled: batch mode simply reports the latest batch, and
  the fallback window catches the most recent weekday submissions.
- arXiv asks API users to be gentle: this client paginates with delays and
  uses a descriptive User-Agent.

## Development

```bash
python3 -m unittest discover -s tests
```

The tests use a recorded API response (`tests/fixtures/`) and fake LLM
providers — no network or API key required.
