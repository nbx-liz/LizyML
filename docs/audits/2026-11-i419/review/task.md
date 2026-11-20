# H-0149 review tasks: model choice

Codex lists `gpt-6.2-sol` as its newest model (`~/.codex/models_cache.json`, 2026-11-20).
lizy-reviewer refused it: run `h0149-proposal-review-20261119T164809Z-3dfece31`,
`REFUSED(capability_unverified)`, "tool_surface: the request carries a top-level tools entry;
bundle 0.167.2, model gpt-6.2-sol, effort high". The tasks therefore pin `gpt-6.1-sol`, the
newest model the reviewer's bundle has verified.
