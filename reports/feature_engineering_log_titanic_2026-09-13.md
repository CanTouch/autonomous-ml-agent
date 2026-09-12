# Feature Engineering Log: titanic
Generated: 2026-09-13

- Binary-encoded `Sex` using mapping {'male': 0, 'female': 1}.
- One-hot encoded ['Embarked'] → new columns: ['Embarked_Q', 'Embarked_S'].