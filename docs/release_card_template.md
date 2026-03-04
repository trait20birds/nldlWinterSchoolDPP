# Synthetic DPP Release Card (Draft Template)

## 1. Dataset purpose
This synthetic dataset is generated from an internal BOM-derived DPP subset to support research, prototyping, and benchmarking when real DPP data is limited or restricted.

## 2. Source data summary (restricted internal data)
- Origin: Partner-provided BOM/DPP subset (CE-RISE)
- Row definition: 
- Number of real rows used: 
- Number of columns used: 
- Sensitive fields removed/generalized: 

## 3. Schema summary
- Numeric fields: 
- Categorical fields: 
- Binary fields: 
- Target label (if used): 
- Fairness grouping: 

## 4. Synthetic generation method(s)
- Tool: SynthCity
- Generator(s): 
- Training settings: 
- Synthetic rows generated: 

## 5. Utility evaluation
- Distribution similarity (univariate): 
- Dependency/correlation preservation: 
- Downstream task: 
- TSTR/TRTS result summary: 

## 6. Privacy evaluation
- Privacy metrics used: 
- Key findings: 
- Residual risks: 

## 7. Fairness evaluation
- Grouping variable: 
- Metrics used: 
- Key findings: 

## 8. Intended use
- Research benchmarking
- Pipeline development / testing
- Demonstration data for DPP tooling

## 9. Out-of-scope / prohibited use
- Decision-making on real suppliers/products
- Compliance/legal claims without validation
- Reverse engineering of source partner records

## 10. Limitations
- Flattened representation may lose hierarchical BOM relations
- Synthetic data may not preserve all domain constraints
- Fairness conclusions depend on source data coverage and grouping choices

## 11. Reproducibility artifacts
- Code repository: 
- Configs/scripts: 
- Random seeds: 
- Date/version: 
