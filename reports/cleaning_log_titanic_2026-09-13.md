# Cleaning Log: titanic
Generated: 2026-09-13

- Added binary flag `has_cabin` from presence/absence of `Cabin`.
- Imputed `Age` (177 missing) with median = 28.00.
- Imputed `Embarked` (2 missing) with mode = 'S'.
- Dropped columns: Ticket, Name, PassengerId, Cabin.