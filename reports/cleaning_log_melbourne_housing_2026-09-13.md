# Cleaning Log: melbourne_housing
Generated: 2026-09-13

- Added binary flag `has_building_area` from presence/absence of `BuildingArea`.
- Imputed `YearBuilt` (5375 missing) with median = 1970.00.
- Filled `CouncilArea` (1369 missing) with 'Unknown' category.
- Imputed `Car` (62 missing) with median = 2.00.
- Dropped columns: Address, BuildingArea.