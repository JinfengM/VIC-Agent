# VIC Auto Modeling evidence reproduction

This repository accompanies the four evidence products used to support the manuscript contributions. Each figure is generated from a machine-readable audit artifact rather than manually entered values. The experiments below were run for the Luanhe case study; their purpose is to test the software claims under controlled conditions, not to establish universal hydrological performance.

## Environment and paths

Run all commands from the repository root:

```bash
cd /yourpath/vic_auto_modeling
conda run -n hydrolib python --version
```

The public repository uses a `src/` layout. Make the packaged modules available
to the command-line scripts and the Streamlit application before running them:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export VIC_SOURCE_DIR="$PWD/runtime/linux-x86_64"
```

The verified environment is `hydrolib`. Model executions use the bundled
Linux x86-64 MPI-VIC runtime and a locally reconstructed baseline run named
`web_demo`. Generated experiment artifacts remain isolated under
`runs/<run_id>/`; publication graphics and their compact evidence records are
written to `report_assets/figures/`.

### Bundled MPI-VIC runtime

`runtime/linux-x86_64/` contains the exact executable pair called by
`vic_auto_modeling.vic_runner`: `MAC_MPI_VIC.X` and its direct `$ORIGIN`
dependency `rout.so`. Their SHA-256 digests are:

```text
fc78fc2217b9a6b7b162d37ec1d31a1dafd83ecac5765058459eae034d9d1f95  MAC_MPI_VIC.X
5d1217d8cff731c155eafc34b637d5c2b8588f9c58c7ef2b7534499e8d17b473  rout.so
```

The binaries require a compatible Linux x86-64 system with OpenMPI
(`libmpi.so.40`), GNU Fortran runtime (`libgfortran.so.5`), and standard glibc
libraries. These operating-system libraries are not vendored. Check linkage on
the target host with `ldd runtime/linux-x86_64/MAC_MPI_VIC.X`. The bundled
runtime retains the licensing terms of its upstream components and is not
covered by claims about portability to other platforms or MPI ABIs.

The scientific-decision experiment calls an OpenAI-compatible internal endpoint. Supply the credential through the environment; do not place it in source files:

```bash
export VIC_AGENT_API_KEY='<your-key>'
export VIC_AGENT_BASE_URL='<your-URL>'
export VIC_AGENT_MODEL='<your-model>'
```

## Evidence map

| Manuscript contribution | Publication asset | Generating script | Primary machine-readable evidence |
|---|---|---|---|
| Deterministic model construction | `deterministic_model_construction.png` | `src/scripts/report/create_deterministic_construction_figure.py` | `report_assets/figures/deterministic_model_construction_evidence.json` |
| Run-level semantic assurance | `run_level_lineage_assurance.png` | `src/scripts/report/run_lineage_assurance_experiment.py`; `src/scripts/report/create_lineage_assurance_figure.py` | `runs/lineage_demo/output/lineage_audit/lineage_audit.json` |
| Evidence-grounded fault diagnosis | `table2_fault_diagnosis.png` | `src/scripts/report/run_fault_diagnosis_experiment.py`; `src/scripts/report/create_fault_diagnosis_table.py` | `runs/diagnosis_demo/output/diagnosis_audit/diagnosis_summary.json` |
| Human-supervised scientific decision | `human_supervised_scientific_decision.png` | `src/scripts/report/run_scientific_decision_experiment.py`; `src/scripts/report/run_approved_scientific_experiments.py`; aggregation and figure scripts | `runs/decision_demo/output/decision_audit/scientific_experiment_execution_audit.json` |

Each figure also has a PDF version and a `*_evidence.json` sidecar in `report_assets/figures/`. The sidecar records the displayed values and, where applicable, the SHA-256 digest of its source audit.

## 1. Deterministic model construction and spatial consistency

### Objective

Test whether the automated construction chain produces a complete and mutually consistent VIC experiment: a stable active-cell identity, one forcing and flux series per active cell, valid routing topology, a successful model run, and routed discharge products at the expected temporal resolutions.

### Design

The figure audits the archived `web_demo` run across four linked checks:

1. grid and parameter identity across the 812 active cells;
2. one-to-one forcing/flux file identity, finite-value scans, and configured 3,288-day support;
3. routing topology and outlet connectivity; and
4. successful VIC/routing execution and the expected daily, monthly, and 12-month climatological output lengths.

The deterministic construction entry points are, in order:

```bash
conda run -n hydrolib python src/scripts/grid/create_fishnet.py
conda run -n hydrolib python src/scripts/veg/create_veg_param.py
conda run -n hydrolib python src/scripts/elevation/create_average_elevation.py
conda run -n hydrolib python src/scripts/soil/create_soil_majority.py
conda run -n hydrolib python src/scripts/forcing/create_forcing.py
conda run -n hydrolib python src/scripts/flow/create_flow.py
conda run -n hydrolib python src/scripts/model/create_model_inputs.py
```

These commands rebuild the shared `output/` workflow. The publication evidence intentionally reads the immutable `runs/web_demo` snapshot so that figure regeneration does not alter its source data.

### Reproduce the figure

```bash
conda run -n hydrolib python src/scripts/report/create_deterministic_construction_figure.py \
  --run-id web_demo \
  --output-dir report_assets/figures
```

### Result and conclusion

The audit found 812 active and uniquely identified cells, 812 forcing files, 812 flux files, identical forcing/flux names, zero non-finite forcing or flux values, and 3,288 daily records per series. The configured dates independently imply 3,288 daily and 108 monthly records; the routing `.year` product is a 12-row climatological monthly summary rather than 12 annual time steps. The routing audit also found zero invalid directions, zero cycles, zero missing routed cells, and VIC return code 0. This supports deterministic construction and internal spatial consistency for the reproduced case. It does not by itself establish correctness for arbitrary basins or input datasets.

## 2. Run-level semantic assurance

### Objective

Test whether a reported parameter vector, simulation, aligned evaluation series, and score belong to the same calibration evaluation. This addresses loss of evaluation identity when a shared `luanx.month` path is overwritten by a later iteration.

### Design

The source calibration contains 100 evaluations. The retained shared output is compared with the calibration history, then two valid evaluation chains are replayed with parameter, simulation, and aligned-series hashes. A controlled mismatch deliberately pairs the best parameters with the latest simulation. The independent validator recomputes the canonical parameter hash, both file hashes, the simulation values in the aligned series, and NSE; it also requires the audit `source_run_id` to match the requested run. It derives PASS or BLOCK from the evaluation identities rather than trusting stored summary counters.

### Reproduce the experiment and figure

```bash
conda run -n hydrolib python src/scripts/report/run_lineage_assurance_experiment.py \
  --source-run-id web_demo \
  --experiment-run-id lineage_demo \
  --source-dir "$VIC_SOURCE_DIR" \
  --processes 12

conda run -n hydrolib python src/scripts/report/create_lineage_assurance_figure.py \
  --run-id web_demo \
  --experiment-run-id lineage_demo \
  --output-prefix report_assets/figures/run_level_lineage_assurance
```

### Results and conclusion

The history identified E19 as the best evaluation (NSE 0.8878869) and E100 as the latest evaluation (NSE 0.8491450). The shared monthly file matched E100 rather than E19, demonstrating the unsafe association that occurs without evaluation lineage. The auditor accepted 2/2 valid replayed chains, blocked the deliberately mismatched E19-parameter/E100-simulation chain, and accepted 0/1 unsafe combinations. This is direct evidence that evaluation lineage protects comparison identity in this workflow. It is a controlled two-chain test, not a comprehensive test of every possible artifact corruption.

## 3. Evidence-grounded fault diagnosis

### Objective

Test whether the diagnostic path identifies the failed workflow stage and the object that should be corrected, while avoiding unsupported changes to calibration parameters or other uninvolved inputs.

### Design

Four faults are injected independently into isolated fixtures:

- D1 removes one active-cell forcing file;
- D2 moves observations outside the simulation period;
- D3 requests an undeclared station; and
- D4 assigns the outlet to an inactive routing cell.

For each case, the experiment first records the observed runtime evidence. A deterministic diagnostic rule engine then derives the attributed stage, corrective object, supporting evidence, and unrelated objects that should not be modified. The injected fault label is retained separately as the reference used for scoring. When the record is later queried through LangGraph, the diagnostic node reapplies the rules instead of trusting the stored PASS label.

### Reproduce the experiment and table

```bash
conda run -n hydrolib python src/scripts/report/run_fault_diagnosis_experiment.py \
  --run-id diagnosis_demo \
  --source-run-id web_demo \
  --source-dir "$VIC_SOURCE_DIR" \
  --processes 12

conda run -n hydrolib python src/scripts/report/create_fault_diagnosis_table.py \
  --run-id diagnosis_demo \
  --output-dir report_assets/figures
```

### Results and conclusion

All four injected cases had complete evidence, correct failed-stage attribution, and the prespecified corrective target (4/4 for each measure), with 0/4 unsupported modifications. The evidence includes the 812→811 forcing inventory and VIC return code 1 (D1), zero overlapping months (D2), absent station output (D3), and inactive-cell/routing evidence despite process return code 0 (D4). The experiment therefore supports evidence-grounded attribution and correction targeting for these four single-fault fixtures. It does not estimate diagnostic accuracy for unseen, ambiguous, or compound faults.

## 4. Human-supervised LLM scientific decision and execution

### Objective

Test the complete decision chain rather than LLM planning alone: whether the LLM selects an experiment that discriminates among competing scientific explanations, cites only supplied evidence, waits for explicit human approval, and then triggers a real, auditable VIC experiment whose result informs the question.

### Design

Three scientific questions are used:

- S1, parameter identifiability: boundary-hitting x1, x2, and x4 trigger inward-sensitivity tests and nine anchored conditional profiles. Each fixed-value search uses six Bayesian-optimization evaluations; the best solution identified dynamically from calibration history is also evaluated as an anchor so a short search cannot report a conditional optimum worse than a known feasible point.
- S2, temporal transferability: reciprocal 2011–2013/2014–2016 calibration and validation, with 15 calibration evaluations in each direction.
- S3, objective adequacy: matched 15-evaluation NSE-only and multi-objective arms. The treatment combines NSE, logNSE, and absolute-bias skill while holding the inputs, period, bounds, seed, and search budget fixed.

For LLM selection, each case is repeated three times with reordered candidate descriptions. The prespecified reference choice is used only for scoring. The selected plan is initially blocked; execution is permitted only after the confirmation token is approved by the modeller.

### Reproduce the decision records

```bash
conda run -n hydrolib python src/scripts/report/run_scientific_decision_experiment.py \
  --run-id decision_demo \
  --source-run-id web_demo \
  --lineage-run-id lineage_demo \
  --repeats 3 \
  --temperature 0.2
```

The command writes `pending_decisions.json`, including a new confirmation token for each case. Review the proposed experiment and record one explicit decision per token:

```bash
conda run -n hydrolib python src/scripts/report/review_scientific_decisions.py \
  --run-id decision_demo --case-id S1 --token '<S1-token>' --decision approve \
  --reviewer-note 'Approved by the human modeller.'
```

Repeat the review command for S2 and S3 with their own tokens. The currently published audit contains three real approvals; the placeholders above intentionally prevent credentials or stale confirmation tokens from being treated as reusable authorization.

### Reproduce the approved experiments

The published results were executed in three isolated run directories. The commands are safe to rerun because completed parameter simulations are cached by parameter hash:

```bash
conda run -n hydrolib python src/scripts/report/run_approved_scientific_experiments.py \
  --experiment S1 --execution-run-id decision_execution_demo \
  --source-dir "$VIC_SOURCE_DIR" --processes 12

conda run -n hydrolib python src/scripts/report/run_approved_scientific_experiments.py \
  --experiment S2 --execution-run-id decision_execution_s2 \
  --source-dir "$VIC_SOURCE_DIR" --processes 12

conda run -n hydrolib python src/scripts/report/run_approved_scientific_experiments.py \
  --experiment S3 --execution-run-id decision_execution_s3 \
  --source-dir "$VIC_SOURCE_DIR" --processes 12
```

Validate the three designs, bind their result paths back to the reviewed decisions, and render the final figure:

```bash
conda run -n hydrolib python src/scripts/report/aggregate_scientific_experiment_evidence.py
conda run -n hydrolib python src/scripts/report/create_scientific_decision_figure.py \
  --run-id decision_demo \
  --output-dir report_assets/figures
```

### Results and conclusions

The LLM matched the prespecified choice in 9/9 reordered-candidate trials, grounded 9/9 responses in supplied evidence, justified the rejected alternatives in 9/9, requested confirmation in 9/9, and claimed no execution before review. All three plans were human-approved and completed, producing 124 unique VIC simulations (S1: 69; S2: 30; S3: 25).

- S1: conditional NSE declined from 0.8879 to 0.8684 as x1 moved from 0.01 to 0.30, and from 0.8879 to 0.8681 as x2 moved from 1.00 to 0.50. For x4, NSE changed from 0.8879 at 1.00 to 0.8888 at 0.80 and then 0.8797 at 0.50. The parameters therefore show mixed behaviour: x1 and x2 are sensitive inward, whereas x4 is locally flat near its bound. This is an exploratory anchored conditional profile, not a formal confidence interval or proof of global identifiability.
- S2: validation NSE was 0.1228 for early-to-late transfer and 0.7816 for late-to-early transfer. The reciprocal result reveals strong temporal asymmetry, but does not by itself identify whether forcing, model structure, or hydrological non-stationarity is the cause.
- S3: relative to the NSE-only arm, multi-objective calibration improved PBIAS by 14.33 percentage points, low-flow PBIAS by 21.58 points, and logNSE by 3.62, while NSE decreased by 0.195. Objective choice therefore changes the scientific trade-off, but the remaining large biases mean that objective formulation alone does not explain all error.

Together, these cases support the contribution at the demonstrated scope: the system converts evidence into a discriminating, human-authorized experiment and executes it through deterministic VIC tools. General claims about decision superiority require additional basins, larger search budgets, independent experts, and unseen or compound decision cases.

The S1--S3 approval path is an experiment-specific reproducibility protocol. It is intentionally separate from the generic LangGraph pending-action branch used by the interactive application for automatic construction, VIC execution, and calibration. The `scientific_decision` LangGraph node does not silently rerun S1--S3; it verifies their source-run identity, approval status, completion event, and result availability, and then writes the reviewed decision and scientific result to `VicAgentState`.

## Streamlit interface

The repository includes the same four-tab interface used by the implementation:
automatic construction, VIC execution, calibration, and the agent assistant. The
assistant loads the selected run state before reasoning, exposes the four
contribution-specific scientific nodes through natural-language queries, and
requires explicit confirmation before a state-changing job can start.

Configure the optional LLM endpoint as described above, then launch the interface:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export VIC_SOURCE_DIR="$PWD/runtime/linux-x86_64"
conda run -n hydrolib streamlit run web_app/app.py \
  --server.address 0.0.0.0 \
  --server.port 8502 \
  --server.headless true \
  --browser.gatherUsageStats false
```

The repository intentionally contains no `runs/`, `output/`, or publication
result files. Consequently, a fresh interface starts without completed run
evidence; users must execute or reconstruct the selected experiment locally.
Large DEM and meteorological inputs and the external MPI-VIC binaries are also
not distributed in this repository.

## Code--evidence alignment safeguards

The manuscript-facing implementation was strengthened in the following order:

1. `vic_auto_modeling/agent/evidence_validation.py` added independent SHA-256, aligned-series, NSE, evaluation-identity, and source-run validation for lineage records.
2. The same module added deterministic evidence rules for missing forcing, zero temporal overlap, undeclared station selection, and inactive outlet mapping. Fault-injection code now supplies runtime evidence to these rules instead of directly assigning the diagnostic label.
3. The lineage, diagnosis, and scientific-decision adapters now reject evidence whose `source_run_id` differs from the run selected by the user.
4. The deterministic-construction audit now scans forcing and flux values for NaN or infinity, derives expected daily and monthly counts from the VIC configuration dates, and checks the 12-row routing climatology.
5. Scientific-decision evidence now locates the best evaluation dynamically from calibration history; neither its aligned-series path nor its explanatory text assumes E19.
6. The manuscript distinguishes the generic interactive confirmation branch from the standalone S1--S3 review and execution protocol.

The archived Lishui--Luanx evidence was regenerated after these changes. All four contribution adapters currently return PASS or COMPLETED for `web_demo`. The tests also include negative controls for a modified lineage hash and for C2--C4 evidence requested under the wrong source run.

## Integrity checks

The production LangGraph explicitly registers `deterministic_construction`, `audit_evaluation_lineage`, `diagnose_run_evidence`, and `scientific_decision`. These nodes validate the four archived contribution-evidence products and write their results to dedicated `VicAgentState` fields. The integration tests verify node registration, source-run identity, state write-back, evidence references, tampered-hash rejection, and fail-closed behaviour:

```bash
conda run -n hydrolib python -m unittest tests/test_agent_scientific_nodes.py -v
```

The following checks regenerate all four display assets from their audits and validate the reporting scripts without rerunning VIC:

```bash
conda run -n hydrolib python src/scripts/report/create_deterministic_construction_figure.py --run-id web_demo
conda run -n hydrolib python src/scripts/report/create_lineage_assurance_figure.py --run-id web_demo --experiment-run-id lineage_demo --output-prefix report_assets/figures/run_level_lineage_assurance
conda run -n hydrolib python src/scripts/report/create_fault_diagnosis_table.py --run-id diagnosis_demo
conda run -n hydrolib python src/scripts/report/aggregate_scientific_experiment_evidence.py
conda run -n hydrolib python src/scripts/report/create_scientific_decision_figure.py --run-id decision_demo
conda run -n hydrolib python -m py_compile src/scripts/report/*.py
```
