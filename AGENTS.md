# Development Data Compatibility

This project is in active development. Persisted files and databases under
`data/` are generated artifacts and may be deleted and rebuilt.

- Implement only the current cache, manifest, database, and serialized-data
  formats.
- Do not add migration scripts, legacy readers, compatibility defaults,
  version guards, or old/new format branches.
- When a persisted format changes, require developers to delete the generated
  data and rerun ingestion.
- Add migration or backward-compatibility behavior only when the user
  explicitly requests it.
