# Domain Documentation

MyClaw has one bounded context. Before exploring a feature, read the relevant definitions in [CONTEXT.md](../../CONTEXT.md) and current decisions in [docs/adr/](../adr/).

- [GitHub Issues](https://github.com/Totoro-debug/myclaw/issues) are the authoritative product requirements and discussion history. Follow [issue-tracker.md](issue-tracker.md) to retrieve the relevant issue and its accepted decisions.
- [CONTEXT.md](../../CONTEXT.md) defines domain vocabulary, without API inventories or implementation plans.
- `docs/adr/` records current architectural decisions and their consequences. Consolidate still-valid decisions when replacing an older design; use Git and GitHub for historical versions.
- [README.md](../../README.md) is the installation, configuration, and usage guide.

Use glossary terms consistently in issues, proposals, code, and tests. Trace uncertain behavior through the implementation and relevant issue before resolving a conflict. Surface unresolved design choices to the user instead of silently selecting an obsolete document as authority.

Keep local documents focused on current information. Do not retain duplicate PRDs, completed implementation plans, migration ledgers, or historical release counts as active contracts. Tests should verify behavior, architecture, persistence, and documentation links rather than require old prose or deleted documents.
