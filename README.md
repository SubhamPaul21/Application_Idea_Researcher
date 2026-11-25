# App Idea Researcher (`idea-miner`)

Finds real problems people describe in public, through sanctioned APIs only, and turns them into ranked app ideas and
PRD specs. This is a personal, non-commercial research tool.

```
Sources (legal APIs) ──▶ 0 regex prefilter ──▶ 1 metadata score ──▶ 2 LLM classifier ──▶ 3 embed + cluster + rank
                                                                                            │
          specs/*.md + *.json  ◀── human review ◀── 4b PRD spec  ◀── 4a "does it exist?" ◀─┘
```

## Quick start (Windows / PowerShell)

```powershell
uv venv .venv --python 3.13
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
copy .env.example .env          # add at least one LLM key; source keys are optional
.venv\Scripts\idea-miner smoke  # check every data source works today
.venv\Scripts\idea-miner ui     # opens the Streamlit app
```

Optional: `uv pip install --python .venv\Scripts\python.exe -e ".[embeddings]"` adds sentence-transformers. It clusters
better and handles Hindi and Hinglish, but PyTorch is about 2 GB. Without it, clustering uses TF-IDF.

## Using the app

**Sidebar**

1. **Niche** (multi-select) plus optional **custom keywords**.
2. **Idea type** presets: quick-win scripts/CLIs, micro-SaaS, consumer apps, developer tools, browser extensions,
   desktop utilities, and high-pain problems people pay for. Open **Fine-tune idea filters** to adjust app types,
   audience, build effort, willingness to pay, severity, and the minimum number of distinct people.
3. **Region** (App Store country / YouTube region), sources, look-back window, search depth.
4. **AI model**: any provider (see below), with an optional separate "writer" model and a per-run cost cap.

**Tabs**

| Tab | What it does |
|---|---|
| Discover | **Find app ideas** runs fetch, prefilter, classify and rank. Each stage can also run on its own. |
| Ideas | Ranked idea cards filtered by idea type. Shows evidence links, **Check if it exists**, and **Write PRD spec**. |
| Specs | Review drafts. **Approve & save** writes `specs/<date>-<slug>.md` plus a `.json` contract for build agents. |
| Problem cards | Every classified post as a table, with CSV export. |
| Sources & compliance | Source status, licenses, exclusions, live health check, retention purge, LLM token usage. |

## LLM providers

The classifier and writer are generic. Pick them in the sidebar, or use `--provider/--model` on the CLI.

| Provider | Key in `.env` | How |
|---|---|---|
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | Official SDK: native structured outputs, per-task effort, prompt caching, refusal fallbacks |
| OpenAI | `OPENAI_API_KEY` | OpenAI-compatible adapter, `json_schema` mode |
| OpenRouter, Groq, Gemini, DeepSeek, Mistral, Together | `<NAME>_API_KEY` | OpenAI-compatible adapter, `json_object` mode |
| Ollama, LM Studio | none | Local server |
| Anything else OpenAI-compatible | `CUSTOM_LLM_BASE_URL`, `CUSTOM_LLM_API_KEY` | |

If a provider rejects a JSON mode or token parameter, the adapter falls back automatically
(`json_schema` → `json_object` → prompt-only). Every reply is validated with Pydantic and gets one repair retry.
To check a model, run `idea-miner llm-test groq <model-id>`.

To add a provider, add a preset to `PROVIDERS` in `src/idea_miner/llm/factory.py`. If it isn't OpenAI-compatible,
subclass `LLMClient` and implement `_complete()`.

## Data sources

| Source | Key | Notes |
|---|---|---|
| Hacker News (Algolia) | none | Ask HN posts and comments |
| Stack Exchange API | `STACKAPPS_KEY` (optional) | 300 requests/day without a key, 10k/day with one. CC BY-SA, so specs link back. |
| GitHub Issues | `GITHUB_TOKEN` (optional) | Open `enhancement` issues with many reactions |
| Apple App Store | none | 1–3★ reviews of top apps per search term (iTunes Search + review RSS) |
| CFPB complaints | none | Finance niches only. The API no longer returns narratives, so complaints are grouped per issue and the count is the signal. |
| Bluesky | `BLUESKY_HANDLE` + `BLUESKY_APP_PASSWORD` (required) | Public search returned 403 in Nov 2026, so an app password is needed |
| YouTube comments | `YOUTUBE_API_KEY` (required) | Comments on tutorial videos |

These are excluded on purpose (reasons are in `config/sources.yaml` and the app):

- Google Play scraping and Reddit dumps: off by owner decision.
- X/Twitter, Quora, LinkedIn, Facebook, Discord, G2/Capterra/Trustpilot, consumercomplaints.in, MouthShut: their terms
  prohibit this kind of access.

## Privacy and compliance built in

- Emails, phone numbers, PAN/Aadhaar-like IDs, @handles and profile URLs are scrubbed before storage and before any
  LLM call. Usernames are stored only as salted hashes, which are still enough to count distinct people.
- Raw text is deleted after `retention_days` (default 30). Problem cards keep only paraphrases and links.
- Specs contain paraphrases and links, never original post text. Evidence and attribution sections are written by
  code, not the LLM, and existence-check alternatives must come from real search results.
- Every record carries its source license and terms status.

## Configuration

- `config/niches.yaml`: niches and their search terms per source. Add your own.
- `config/sources.yaml`: source registry, compliance metadata, engagement weights.
- `config/prefilter.yaml`: wish-phrase regexes.
- `config/prompts.yaml`: the four LLM system prompts (classifier, idea naming, existence check, PRD spec). Edits apply
  on the next run. The allowed values listed in them must match `core/schemas.py`; `pytest tests/test_prompts.py`
  checks this.
- `config/settings.yaml`: batch sizes, workers, retention, default models, Claude effort per task.

## CLI

```
idea-miner ui | smoke | niches | purge
idea-miner run -n small_business -n personal_finance --region India --provider groq --model <id>
idea-miner run -k "dental clinic" -k "appointment reminders" --stages fetch,filter
idea-miner llm-test anthropic claude-opus-5-5
```

## Tests

`.venv\Scripts\python -m pytest`. Tests run offline: the end-to-end test drives the whole pipeline with a fake LLM.
