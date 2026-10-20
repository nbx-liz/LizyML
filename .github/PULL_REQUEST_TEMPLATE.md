## Summary
<!-- 1-3 bullet points describing the change -->

## Checklist
- [ ] Tests added/updated
- [ ] `make ci` passes locally
- [ ] HISTORY.md Proposal added (if spec change)
- [ ] CHANGELOG.md updated (if user-facing change)
- [ ] Layers 1-3 changed: none of the following outside the two exceptions in BLUEPRINT §2.1, checked by review until H-0131's implementation and the matching architecture tests are merged: (1) concrete assembly, including selection by Config type, `name` or a value derived from them; (2) concrete type dispatch; (3) registry lookups that bypass the registry helpers; (4) rewriting registrations (an implementation module registering its own types is allowed)

## Test plan
- [ ] ...
