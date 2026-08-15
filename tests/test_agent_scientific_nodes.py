import json
import tempfile
import unittest
from pathlib import Path

from vic_auto_modeling.agent.evidence_validation import file_sha256, parameter_sha256
from vic_auto_modeling.agent.orchestrator import VicAgent
from vic_auto_modeling.agent.scientific_tools import (
    audit_evaluation_lineage,
    deterministic_construction,
)


class FakeLlmClient:
    def __init__(self, tool):
        self.tool = tool

    def chat(self, messages, **kwargs):
        if "按系统提示词输出" in messages[-1]["content"]:
            return json.dumps(
                {"type": "scientific_tool", "tool": self.tool, "args": {}}
            )
        return "Evidence-grounded summary."

    def stream_chat(self, messages, **kwargs):
        yield self.chat(messages, **kwargs)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class ScientificGraphNodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        write_json(
            self.root
            / "report_assets/figures/deterministic_model_construction_evidence.json",
            {
                "run_id": "web_demo",
                "active_cells": 2,
                "unique_ids": 2,
                "id_sets_equal": True,
                "forcing_files": 2,
                "flux_files": 2,
                "forcing_flux_names_equal": True,
                "forcing_non_finite": 0,
                "flux_non_finite": 0,
                "forcing_record_counts_match": True,
                "flux_record_counts_match": True,
                "routed_record_counts_match": True,
                "returncode": 0,
                "invalid_directions": 0,
                "cycles": 0,
                "not_found": 0,
                "upstream_cells": 1,
                "logged_upstream": 1,
                "daily_rows": 2,
                "monthly_rows": 1,
                "climatology_rows": 1,
            },
        )
        lineage_root = self.root / "runs/lineage_demo/output/lineage_audit"
        observation = self.root / "runs/lineage_demo/input/observation.csv"
        observation.parent.mkdir(parents=True, exist_ok=True)
        observation.write_text("year,month,flow\n2011,1,1\n2011,2,2\n", encoding="utf-8")
        valid_chains = []
        for evaluation_id in (19, 100):
            evaluation_root = lineage_root / f"evaluation_{evaluation_id:03d}"
            evaluation_root.mkdir(parents=True, exist_ok=True)
            monthly = evaluation_root / "luanx.month"
            aligned = evaluation_root / "luanx_aligned_monthly.csv"
            monthly.write_text("2011 1 1\n2011 2 1.5\n", encoding="utf-8")
            aligned.write_text(
                "date,year,month,observed,simulated\n"
                "2011-01-01,2011,1,1,1\n2011-02-01,2011,2,2,1.5\n",
                encoding="utf-8",
            )
            parameters = {name: evaluation_id for name in ("x1", "x2", "x3", "x4", "x5", "x6")}
            valid_chains.append(
                {
                    "run_id": "lineage_demo",
                    "evaluation_id": evaluation_id,
                    "parameters": parameters,
                    "parameter_sha256": parameter_sha256(parameters),
                    "monthly_path": str(monthly),
                    "monthly_sha256": file_sha256(monthly),
                    "aligned_path": str(aligned),
                    "aligned_sha256": file_sha256(aligned),
                    "expected_nse": 0.5,
                }
            )
        self.lineage_audit_path = lineage_root / "lineage_audit.json"
        write_json(
            self.lineage_audit_path,
            {
                "source_run_id": "web_demo",
                "experiment_run_id": "lineage_demo",
                "observation_path": str(observation),
                "observation_sha256": file_sha256(observation),
                "valid_chains": valid_chains,
                "controlled_mismatch": {
                    "parameter_evaluation_id": 19,
                    "simulation_evaluation_id": 100,
                    "aligned_evaluation_id": 100,
                    "metric_evaluation_id": 19,
                    "decision": "BLOCK",
                },
            },
        )
        write_json(
            self.root
            / "runs/diagnosis_demo/output/diagnosis_audit/diagnosis_summary.json",
            {
                "source_run_id": "web_demo",
                "metrics": {"cases_passed": 1},
                "cases": [
                    {
                        "case_id": "D1",
                        "run_evidence": {
                            "forcing_files_before": 2,
                            "forcing_files_after_injection": 1,
                            "missing_file": "forcing_1",
                            "vic_returncode": 1,
                        },
                        "diagnosis": {
                            "failed_stage": "forcing_preparation",
                            "correction_target": {
                                "object": "missing active-cell forcing file",
                                "path": "forcing_1",
                            },
                        },
                    }
                ],
            },
        )
        decision_root = self.root / "runs/decision_demo/output/decision_audit"
        completed = {
            "case_id": "S1",
            "status": "approved_for_execution",
            "execution_authorized": True,
            "execution_events": [{"status": "completed"}],
        }
        write_json(
            decision_root / "decision_audit.json",
            {
                "source_run_id": "web_demo",
                "metrics": {"human_approved": 1},
                "pending_decisions": [completed],
            },
        )
        write_json(
            decision_root / "scientific_experiment_execution_audit.json",
            {"results": {"S1": {"conclusion": "mixed identifiability"}}},
        )

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, tool):
        return VicAgent(FakeLlmClient(tool)).respond(
            "web_demo", "test", project_root=self.root
        )

    def test_graph_registers_four_contribution_nodes(self):
        graph = VicAgent(FakeLlmClient("audit_evaluation_lineage")).graph.get_graph()
        self.assertTrue(
            {
                "deterministic_construction",
                "audit_evaluation_lineage",
                "diagnose_run_evidence",
                "scientific_decision",
            }.issubset(graph.nodes)
        )

    def test_construction_result_is_written_to_agent_state(self):
        response = self.invoke("deterministic_construction")
        self.assertEqual(response.construction_result["decision"], "PASS")
        self.assertEqual(response.construction_result["summary"]["active_cells"], 2)
        self.assertEqual(len(response.evidence_refs), 1)

    def test_construction_audit_rejects_another_run(self):
        result = deterministic_construction("another_run", project_root=self.root)
        self.assertFalse(result["ok"])
        self.assertIn("not requested run", result["message"])

    def test_lineage_result_is_written_to_agent_state(self):
        response = self.invoke("audit_evaluation_lineage")
        self.assertEqual(response.lineage_audit["decision"], "PASS")
        self.assertEqual(response.lineage_audit["summary"]["mismatches_blocked"], 1)
        self.assertEqual(len(response.evidence_refs), 1)

    def test_diagnosis_result_is_written_to_agent_state(self):
        response = self.invoke("diagnose_run_evidence")
        self.assertEqual(response.diagnosis_result["decision"], "PASS")
        self.assertEqual(response.diagnosis_result["diagnoses"][0]["case_id"], "D1")

    def test_decision_and_execution_result_are_written_to_agent_state(self):
        response = self.invoke("scientific_decision")
        self.assertEqual(response.scientific_decision["decision"], "COMPLETED")
        self.assertEqual(
            response.scientific_result["S1"]["conclusion"], "mixed identifiability"
        )
        self.assertEqual(len(response.evidence_refs), 2)

    def test_streaming_path_reuses_scientific_adapter(self):
        metadata = {}
        agent = VicAgent(FakeLlmClient("diagnose_run_evidence"))
        list(
            agent.stream_interaction(
                "web_demo", "test", project_root=self.root, metadata=metadata
            )
        )
        self.assertEqual(metadata["diagnosis_result"]["decision"], "PASS")
        self.assertEqual(len(metadata["evidence_refs"]), 1)

    def test_missing_lineage_evidence_fails_closed(self):
        result = audit_evaluation_lineage(
            "web_demo", project_root=self.root, experiment_run_id="missing_lineage"
        )
        self.assertFalse(result["ok"])
        self.assertIn("not found", result["message"].lower())

    def test_tampered_lineage_hash_is_blocked(self):
        audit = json.loads(self.lineage_audit_path.read_text(encoding="utf-8"))
        audit["valid_chains"][0]["monthly_sha256"] = "0" * 64
        write_json(self.lineage_audit_path, audit)
        result = audit_evaluation_lineage("web_demo", project_root=self.root)
        self.assertFalse(result["ok"])
        self.assertTrue(any("hash mismatch" in error for error in result["validation_errors"]))

    def test_lineage_rejects_another_source_run(self):
        result = audit_evaluation_lineage("another_run", project_root=self.root)
        self.assertFalse(result["ok"])
        self.assertIn("source_run_id", " ".join(result["validation_errors"]))

    def test_diagnosis_rejects_another_source_run(self):
        response = VicAgent(FakeLlmClient("diagnose_run_evidence")).respond(
            "another_run", "test", project_root=self.root
        )
        self.assertFalse(response.diagnosis_result["ok"])

    def test_decision_rejects_another_source_run(self):
        response = VicAgent(FakeLlmClient("scientific_decision")).respond(
            "another_run", "test", project_root=self.root
        )
        self.assertFalse(response.scientific_decision["decision"] == "COMPLETED")


if __name__ == "__main__":
    unittest.main()
