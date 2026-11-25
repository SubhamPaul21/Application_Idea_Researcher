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
uv pip sync --require-hashes --python .venv\Scripts\python.exe requirements.lock   # exact, hash-checked versions
uv pip install --no-deps --python .venv\Scripts\python.exe -e .                    # the app itself
copy .env.example .env          # add at least one LLM key; source keys are optional
.venv\Scripts\idea-miner smoke  # check every data source works today
.venv\Scripts\idea-miner ui     # opens the Streamlit app
```

`requirements.lock` pins every package (including test tools) to versions that passed `pip-audit` with no known
vulnerabilities, with SHA-256 hashes so a tampered download is rejected. To upgrade deliberately:

```powershell
uv pip compile pyproject.toml --extra dev --upgrade --generate-hashes --universal --python-version 3.13 -o requirements.lock
uvx pip-audit --disable-pip -r requirements.lock   # check before installing
uv pip sync --require-hashes --python .venv\Scripts\python.exe requirements.lock
uv pip install --no-deps --python .venv\Scripts\python.exe -e .
```

Optional (not in the lock): `uv pip install --python .venv\Scripts\python.exe -e ".[embeddings]"` adds sentence-transformers. It clusters
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
| Specs | Review drafts in three views: rendered, raw Markdown (the exact file), and the JSON build contract. **Approve & save** writes exactly what you reviewed to `specs/<date>-<spec id>-<slug>.md` plus `.json`, and never overwrites a file. **Reject** on an approved spec moves its files to `specs/revoked/`. |
| Problem cards | Every classified post as a table, with CSV export (cells starting with `=`, `+`, `-` or `@` get a leading `'` so spreadsheets don't run them). Posts that failed classification 3 times are listed here and no longer re-sent. |
| Sources & compliance | Source status, licenses, exclusions, live health check, retention purge, LLM token usage. |

## LLM providers

The classifier and writer are generic. Pick them in the sidebar, or use `--provider/--model` on the CLI.

| Provider | Key in `.env` | How |
|---|---|---|
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | Official SDK: native structured outputs, per-task effort, prompt caching, refusal fallbacks |
| OpenAI | `OPENAI_API_KEY` | OpenAI-compatible adapter, `json_schema` mode |
| OpenRouter, Groq, Gemini, DeepSeek, Mistral, Together | `<NAME>_API_KEY` | OpenAI-compatible adapter, `json_object` mode |
| Ollama, LM Studio | none | Local server |
| Anything else OpenAI-compatible | `CUSTOM_LLM_BASE_URL`, `CUSTOM_LLM_API_KEY` | The stored key is only sent to the stored URL. If you type another URL in the sidebar, paste a key for it. |

`OPENAI_ORG_ID`, `OPENAI_PROJECT_ID` and `OPENAI_CUSTOM_HEADERS` are only sent to OpenAI itself, never to other providers.

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

GitHub queries always include `is:public`, so even a token that can see private repositories only returns public results.

These are excluded on purpose (reasons are in `config/sources.yaml` and the app):

- Google Play scraping and Reddit dumps: off by owner decision.
- X/Twitter, Quora, LinkedIn, Facebook, Discord, G2/Capterra/Trustpilot, consumercomplaints.in, MouthShut: their terms
  prohibit this kind of access.

## Security

The app has **no login**, so it only accepts connections from this computer. `app/.streamlit/config.toml` (and
`idea-miner ui`) bind it to `127.0.0.1` and accept only the host names `localhost` and `127.0.0.1`, which also blocks
DNS-rebinding web pages. To use it from another device, change `address` in that file to `0.0.0.0` only on a network
you trust: anyone who can reach the port can spend your API quota and approve specs.

Start the app with `idea-miner ui` (or `streamlit run app/server.py`). That entry point also adds headers that stop
other websites from showing the app inside a frame and tricking you into clicking its buttons (clickjacking).
`streamlit run app/streamlit_app.py` still works and stays on this computer, but without those headers.

Other protections:

- **Keys come only from you.** Claude needs `ANTHROPIC_API_KEY` in `.env` or a key pasted in the sidebar; the app
  never falls back to other credentials on the machine, and always talks to `https://api.anthropic.com` (change
  `llm.anthropic.base_url` in `config/settings.yaml` to use a proxy). Values in `.env`, including empty ones,
  take precedence over environment variables set elsewhere in Windows.

- **Posts are kept apart.** Each post is sent to the classifier as escaped JSON with a random label, so one post
  cannot fake another post's block or name it, and every result must match exactly one label.
- **Model output is shown literally.** Titles, problems and gaps from the model are escaped in the UI. Spec files
  have images, hidden Markdown and non-https links removed.
- **Spec approval is bound to what you saw.** Buttons are tied to one spec, and approval is refused if the spec
  changed after you viewed it.
- **Search problems are reported.** A failed existence search is never recorded as "unsolved", and GitHub searches
  are restricted to public repositories.
- **Errors stay short.** The UI shows a brief error; the full traceback goes to the terminal.
- **Keys stay encrypted in transit.** A key is never sent over plain `http://` to another machine; plain http is
  allowed only for servers on this computer, or for a keyless server (for example Ollama on your LAN).
- **Downloads are bounded.** API responses over `http.max_response_mb` (default 10) are refused, and redirects are
  only followed within the same host. The YouTube key is sent in a header, not in the URL.
- **The CLI rejects mistakes.** Unknown `--niche`, `--source`, `--region`, `--stages` or `--provider` values and
  out-of-range numbers stop the run instead of being ignored.
- **Existence checks respect your sources.** A source disabled in `config/sources.yaml` is not searched either.

## Privacy and compliance built in

- Emails, phone numbers (including several in a row), PAN/Aadhaar-like IDs, @handles and profile links (with or
  without `https://`) are removed from post text and from source details (video, thread and app titles) before
  storage and before any LLM call. Author names are stored only as salted hashes, which are still enough to count
  distinct people.
- Each item keeps a link to the original public post, and that link identifies the account that posted it.
  Bluesky links use the account's permanent ID; GitHub context names the repository but not its owner.
- Raw text and source details are deleted after `retention_days` (default 30). The purge runs at the start of every
  pipeline run and hourly while the app is open, and expired items are never classified. App Store review text is
  deleted as soon as its card is saved. Deleted text is overwritten in the database file.
- Problem cards keep model-written paraphrases. The model is instructed not to copy; cards that copy 12 or more
  consecutive words of a post are rejected; personal data is removed from every model-written field and lengths are
  capped.
- Specs contain paraphrases and links. Evidence and attribution sections are written by code, not the LLM, and
  existence-check alternatives (names and links) come from real search results.
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
