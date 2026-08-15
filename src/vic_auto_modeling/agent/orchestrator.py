import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from vic_auto_modeling.agent.llm_client import QwenVllmClient
from vic_auto_modeling.agent.prompts import SYSTEM_PROMPT, build_user_prompt
from vic_auto_modeling.agent.scientific_tools import (
    SCIENTIFIC_TOOLS,
    execute_scientific_tool,
)
from vic_auto_modeling.agent.state import get_run_summary
from vic_auto_modeling.agent.tools import (
    READONLY_TOOLS,
    confirm_pending_action,
    execute_readonly_tool,
    prepare_pending_action,
    validate_action,
)


class VicAgentState(TypedDict, total=False):
    run_id: str
    user_message: str
    messages: list
    run_summary: dict
    llm_plan: dict | None
    llm_available: bool
    tool_result: dict | None
    proposed_action: dict | None
    confirmation_required: bool
    confirmation_token: str | None
    final_answer: str
    project_root: str
    construction_result: dict | None
    lineage_audit: dict | None
    diagnosis_result: dict | None
    scientific_decision: dict | None
    scientific_result: dict | None
    evidence_refs: list[str]


@dataclass
class AgentResponse:
    message: str
    run_summary: dict
    proposed_action: dict | None = None
    confirmation_required: bool = False
    confirmation_token: str | None = None
    tool_result: dict | None = None
    construction_result: dict | None = None
    lineage_audit: dict | None = None
    diagnosis_result: dict | None = None
    scientific_decision: dict | None = None
    scientific_result: dict | None = None
    evidence_refs: list[str] | None = None


@dataclass
class AgentStreamResponse:
    chunks: Any
    run_summary: dict
    proposed_action: dict | None = None
    confirmation_required: bool = False
    confirmation_token: str | None = None
    tool_result: dict | None = None
    construction_result: dict | None = None
    lineage_audit: dict | None = None
    diagnosis_result: dict | None = None
    scientific_decision: dict | None = None
    scientific_result: dict | None = None
    evidence_refs: list[str] | None = None


def _extract_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            return json.loads(match.group(0))
        raise


class VicAgent:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client or QwenVllmClient()
        self.graph = self._build_graph()

    def respond(self, run_id, user_message, project_root="."):
        initial = {
            "run_id": run_id,
            "user_message": user_message,
            "project_root": str(Path(project_root).resolve()),
            "messages": [],
            "confirmation_required": False,
            "confirmation_token": None,
        }
        result = self.graph.invoke(initial)
        return AgentResponse(
            message=result.get("final_answer", ""),
            run_summary=result.get("run_summary", {}),
            proposed_action=result.get("proposed_action"),
            confirmation_required=bool(result.get("confirmation_required")),
            confirmation_token=result.get("confirmation_token"),
            tool_result=result.get("tool_result"),
            construction_result=result.get("construction_result"),
            lineage_audit=result.get("lineage_audit"),
            diagnosis_result=result.get("diagnosis_result"),
            scientific_decision=result.get("scientific_decision"),
            scientific_result=result.get("scientific_result"),
            evidence_refs=result.get("evidence_refs"),
        )

    def respond_stream(self, run_id, user_message, project_root="."):
        initial = {
            "run_id": run_id,
            "user_message": user_message,
            "project_root": str(Path(project_root).resolve()),
            "messages": [],
            "confirmation_required": False,
            "confirmation_token": None,
        }
        state = {}
        for update in self.graph.stream(initial):
            node_update = next(iter(update.values()))
            if node_update:
                state.update(node_update)

        if state.get("confirmation_required"):
            message = self._confirmation_message(state)
            return AgentStreamResponse(
                chunks=iter([message]),
                run_summary=state.get("run_summary", {}),
                proposed_action=state.get("proposed_action"),
                confirmation_required=True,
                confirmation_token=state.get("confirmation_token"),
                tool_result=state.get("tool_result"),
                **self._scientific_response_fields(state),
            )

        if state.get("final_answer"):
            return AgentStreamResponse(
                chunks=iter([state.get("final_answer", "")]),
                run_summary=state.get("run_summary", {}),
                tool_result=state.get("tool_result"),
                **self._scientific_response_fields(state),
            )

        tool_result = state.get("tool_result")
        if tool_result is not None:
            if not state.get("llm_available", True):
                chunks = iter([self._format_tool_result(user_message, tool_result)])
            else:
                chunks = self._stream_tool_answer(user_message, tool_result)
            return AgentStreamResponse(
                chunks=chunks,
                run_summary=state.get("run_summary", {}),
                tool_result=tool_result,
                **self._scientific_response_fields(state),
            )

        return AgentStreamResponse(
            chunks=iter(["我还不能确定下一步动作。"]),
            run_summary=state.get("run_summary", {}),
            **self._scientific_response_fields(state),
        )

    def _scientific_response_fields(self, state):
        return {
            "construction_result": state.get("construction_result"),
            "lineage_audit": state.get("lineage_audit"),
            "diagnosis_result": state.get("diagnosis_result"),
            "scientific_decision": state.get("scientific_decision"),
            "scientific_result": state.get("scientific_result"),
            "evidence_refs": state.get("evidence_refs"),
        }

    def confirm(self, run_id, confirmation_token, project_root="."):
        project_root = str(Path(project_root).resolve())
        result = confirm_pending_action(run_id, confirmation_token, project_root=project_root)
        summary = get_run_summary(run_id, project_root=project_root)
        return AgentResponse(
            message=result.get("message", ""),
            run_summary=summary,
            tool_result=result,
            confirmation_required=False,
            confirmation_token=None,
        )

    def stream_interaction(self, run_id, user_message, project_root=".", metadata=None):
        metadata = metadata if metadata is not None else {}
        project_root = str(Path(project_root).resolve())
        run_summary = get_run_summary(run_id, project_root=project_root)
        metadata["run_summary"] = run_summary

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    run_id,
                    user_message,
                    json.dumps(run_summary, ensure_ascii=False, indent=2),
                ),
            },
        ]

        raw_parts = []
        thinking_started = False
        thinking_finished = False
        buffer = ""
        try:
            yield "### 规划思考\n\n"
            for chunk in self.llm_client.stream_chat(messages):
                raw_parts.append(chunk)
                if thinking_finished:
                    continue
                buffer += chunk
                while buffer:
                    if not thinking_started:
                        start = buffer.find("<think>")
                        if start < 0:
                            buffer = buffer[-16:]
                            break
                        thinking_started = True
                        buffer = buffer[start:]

                    end = buffer.find("</think>")
                    if end >= 0:
                        yield buffer[: end + len("</think>")]
                        buffer = buffer[end + len("</think>") :]
                        thinking_finished = True
                        break

                    if len(buffer) > 16:
                        yield buffer[:-16]
                        buffer = buffer[-16:]
                        break
                    break

            raw_text = "".join(raw_parts)
            plan = self._normalize_plan(user_message, _extract_json(raw_text))
            llm_available = True
        except Exception as exc:
            plan = self._fallback_plan(user_message, str(exc))
            llm_available = False
            if not thinking_started:
                yield f"规划阶段暂未获得可展示的模型思考：{exc}"

        yield "\n\n---\n\n### 智能体回复\n\n"
        yield from self._stream_plan_result(
            run_id,
            user_message,
            plan,
            run_summary,
            llm_available,
            project_root,
            metadata,
        )

    def _stream_plan_result(
        self,
        run_id,
        user_message,
        plan,
        run_summary,
        llm_available,
        project_root,
        metadata,
    ):
        if plan.get("type") == "readonly_tool" and plan.get("tool") in READONLY_TOOLS:
            tool_result = execute_readonly_tool(
                plan.get("tool"),
                plan.get("args") or {},
                run_id,
                project_root=project_root,
            )
            metadata["tool_result"] = tool_result
            if llm_available:
                yield from self._stream_tool_answer(user_message, tool_result)
            else:
                yield self._format_tool_result(user_message, tool_result)
            return

        if plan.get("type") == "scientific_tool" and plan.get("tool") in SCIENTIFIC_TOOLS:
            tool = plan["tool"]
            tool_result = execute_scientific_tool(
                tool,
                plan.get("args") or {},
                run_id,
                project_root=project_root,
            )
            metadata["tool_result"] = tool_result
            metadata["evidence_refs"] = self._scientific_evidence_refs(tool_result)
            if tool == "deterministic_construction":
                metadata["construction_result"] = tool_result
            elif tool == "audit_evaluation_lineage":
                metadata["lineage_audit"] = tool_result
            elif tool == "diagnose_run_evidence":
                metadata["diagnosis_result"] = tool_result
            else:
                metadata["scientific_decision"] = {
                    "decision": tool_result.get("decision"),
                    "decisions": tool_result.get("decisions", []),
                }
                metadata["scientific_result"] = tool_result.get("scientific_results", {})
            if llm_available:
                yield from self._stream_tool_answer(user_message, tool_result)
            else:
                yield self._format_tool_result(user_message, tool_result)
            return

        if plan.get("type") == "propose_action":
            action = plan.get("action")
            args = plan.get("args") or {}
            ok, message, args = validate_action(
                run_id,
                action,
                args,
                project_root=project_root,
            )
            if not ok:
                metadata["tool_result"] = {"ok": False, "message": message}
                yield message
                return
            pending = prepare_pending_action(
                run_id,
                action,
                args=args,
                reason=plan.get("reason", ""),
                project_root=project_root,
            )
            metadata["proposed_action"] = pending
            metadata["confirmation_required"] = True
            metadata["confirmation_token"] = pending["token"]
            yield self._confirmation_message({"proposed_action": pending})
            return

        yield plan.get("message", "我还不能确定下一步动作。")

    def _build_graph(self):
        graph = StateGraph(VicAgentState)
        graph.add_node("load_run_summary", self._load_run_summary)
        graph.add_node("llm_reason", self._llm_reason)
        graph.add_node("execute_readonly_tool", self._execute_readonly_tool)
        graph.add_node("deterministic_construction", self._deterministic_construction)
        graph.add_node("audit_evaluation_lineage", self._audit_evaluation_lineage)
        graph.add_node("diagnose_run_evidence", self._diagnose_run_evidence)
        graph.add_node("scientific_decision", self._scientific_decision)
        graph.add_node("prepare_confirmation", self._prepare_confirmation)
        graph.add_node("answer_directly", self._answer_directly)
        graph.add_node("compose_final_answer", self._compose_final_answer)
        graph.add_edge(START, "load_run_summary")
        graph.add_edge("load_run_summary", "llm_reason")
        graph.add_conditional_edges(
            "llm_reason",
            self._route_decision,
            {
                "execute_readonly_tool": "execute_readonly_tool",
                "deterministic_construction": "deterministic_construction",
                "audit_evaluation_lineage": "audit_evaluation_lineage",
                "diagnose_run_evidence": "diagnose_run_evidence",
                "scientific_decision": "scientific_decision",
                "prepare_confirmation": "prepare_confirmation",
                "answer_directly": "answer_directly",
            },
        )
        graph.add_edge("execute_readonly_tool", "compose_final_answer")
        graph.add_edge("deterministic_construction", "compose_final_answer")
        graph.add_edge("audit_evaluation_lineage", "compose_final_answer")
        graph.add_edge("diagnose_run_evidence", "compose_final_answer")
        graph.add_edge("scientific_decision", "compose_final_answer")
        graph.add_edge("prepare_confirmation", "compose_final_answer")
        graph.add_edge("answer_directly", "compose_final_answer")
        graph.add_edge("compose_final_answer", END)
        return graph.compile()

    def _load_run_summary(self, state):
        summary = get_run_summary(state["run_id"], project_root=state["project_root"])
        return {"run_summary": summary}

    def _llm_reason(self, state):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    state["run_id"],
                    state["user_message"],
                    json.dumps(state["run_summary"], ensure_ascii=False, indent=2),
                ),
            },
        ]
        try:
            text = self.llm_client.chat(messages)
            plan = _extract_json(text)
            plan = self._normalize_plan(state["user_message"], plan)
            available = True
        except Exception as exc:
            plan = self._fallback_plan(state["user_message"], str(exc))
            available = False
        return {"llm_plan": plan, "llm_available": available}

    def _normalize_plan(self, user_message, plan):
        text = user_message or ""
        if (
            any(keyword in text for keyword in ("NSE", "最优", "率定结果"))
            and "参数" in text
            and plan.get("type") == "readonly_tool"
            and plan.get("tool") == "explain_calibration_parameters"
        ):
            return {"type": "readonly_tool", "tool": "summarize_calibration", "args": {}}
        return plan

    def _fallback_plan(self, user_message, error):
        text = user_message or ""
        if any(keyword in text for keyword in ("确定性构建", "构建完整性", "自动建模证据")):
            return {"type": "scientific_tool", "tool": "deterministic_construction", "args": {}}
        if any(keyword in text for keyword in ("血缘", "语义一致", "比较对象")):
            return {"type": "scientific_tool", "tool": "audit_evaluation_lineage", "args": {}}
        if any(keyword in text for keyword in ("故障诊断", "诊断证据", "修改对象")):
            return {"type": "scientific_tool", "tool": "diagnose_run_evidence", "args": {}}
        if any(keyword in text for keyword in ("科学决策", "下一项实验", "S1", "S2", "S3")):
            return {"type": "scientific_tool", "tool": "scientific_decision", "args": {}}
        if any(keyword in text for keyword in ("进度", "状态", "当前", "检查")):
            return {"type": "readonly_tool", "tool": "inspect_run", "args": {}}
        if any(keyword in text for keyword in ("日志", "失败", "错误", "报错")):
            return {
                "type": "readonly_tool",
                "tool": "read_stage_logs",
                "args": {"stage": "vic_run"},
            }
        if any(keyword in text for keyword in ("参数", "NSE", "率定结果", "最优")):
            return {"type": "readonly_tool", "tool": "summarize_calibration", "args": {}}
        if "运行" in text and "VIC" in text.upper():
            return {
                "type": "propose_action",
                "action": "start_vic_run",
                "args": {"processes": 12},
                "reason": "用户请求运行 VIC 模型。LLM 服务暂时不可用，系统按默认参数生成待确认动作。",
            }
        if "率定" in text:
            match = re.search(r"(\d+)\s*次", text)
            iterations = int(match.group(1)) if match else 10
            return {
                "type": "propose_action",
                "action": "start_calibration",
                "args": {"iterations": iterations},
                "reason": "用户请求启动参数率定。LLM 服务暂时不可用，系统按默认配置生成待确认动作。",
            }
        return {
            "type": "answer",
            "message": f"LLM 服务暂时不可用或返回异常：{error}。你可以先询问当前进度、查看日志、解释参数，或稍后重试。",
        }

    def _route_decision(self, state):
        plan = state.get("llm_plan") or {}
        if plan.get("type") == "readonly_tool" and plan.get("tool") in READONLY_TOOLS:
            return "execute_readonly_tool"
        if plan.get("type") == "scientific_tool" and plan.get("tool") in SCIENTIFIC_TOOLS:
            return plan["tool"]
        if plan.get("type") == "propose_action":
            return "prepare_confirmation"
        return "answer_directly"

    def _execute_readonly_tool(self, state):
        plan = state.get("llm_plan") or {}
        result = execute_readonly_tool(
            plan.get("tool"),
            plan.get("args") or {},
            state["run_id"],
            project_root=state["project_root"],
        )
        return {"tool_result": result}

    def _scientific_tool_state(self, state, tool):
        plan = state.get("llm_plan") or {}
        result = execute_scientific_tool(
            tool,
            plan.get("args") or {},
            state["run_id"],
            project_root=state["project_root"],
        )
        update = {
            "tool_result": result,
            "evidence_refs": self._scientific_evidence_refs(result),
        }
        if tool == "deterministic_construction":
            update["construction_result"] = result
        elif tool == "audit_evaluation_lineage":
            update["lineage_audit"] = result
        elif tool == "diagnose_run_evidence":
            update["diagnosis_result"] = result
        else:
            update["scientific_decision"] = {
                "decision": result.get("decision"),
                "decisions": result.get("decisions", []),
            }
            update["scientific_result"] = result.get("scientific_results", {})
        return update

    def _audit_evaluation_lineage(self, state):
        return self._scientific_tool_state(state, "audit_evaluation_lineage")

    def _deterministic_construction(self, state):
        return self._scientific_tool_state(state, "deterministic_construction")

    def _diagnose_run_evidence(self, state):
        return self._scientific_tool_state(state, "diagnose_run_evidence")

    def _scientific_decision(self, state):
        return self._scientific_tool_state(state, "scientific_decision")

    def _scientific_evidence_refs(self, result):
        return [
            result[key]
            for key in ("audit_path", "execution_audit_path")
            if result.get(key)
        ]

    def _prepare_confirmation(self, state):
        plan = state.get("llm_plan") or {}
        action = plan.get("action")
        args = plan.get("args") or {}
        ok, message, args = validate_action(
            state["run_id"],
            action,
            args,
            project_root=state["project_root"],
        )
        if not ok:
            return {
                "tool_result": {"ok": False, "message": message},
                "confirmation_required": False,
                "proposed_action": None,
            }
        pending = prepare_pending_action(
            state["run_id"],
            action,
            args=args,
            reason=plan.get("reason", ""),
            project_root=state["project_root"],
        )
        return {
            "tool_result": {"ok": True, "message": "已生成待确认动作。"},
            "proposed_action": pending,
            "confirmation_required": True,
            "confirmation_token": pending["token"],
        }

    def _answer_directly(self, state):
        plan = state.get("llm_plan") or {}
        return {"final_answer": plan.get("message", "我还不能确定下一步动作。")}

    def _compose_final_answer(self, state):
        if state.get("final_answer"):
            return {}

        if state.get("confirmation_required"):
            return {"final_answer": self._confirmation_message(state)}

        tool_result = state.get("tool_result")
        if tool_result is not None:
            if not state.get("llm_available", True):
                return {
                    "final_answer": self._format_tool_result(
                        state["user_message"],
                        tool_result,
                    )
                }
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是 VIC 水文建模助手。请基于工具结果用中文简洁回答，不要编造。"
                        "解释参数时必须使用工具结果中的 parameter_definitions。"
                        "NSE 是 Nash-Sutcliffe efficiency，用于评价模拟流量与观测流量吻合程度，不要称为决定系数，也不要写成流速。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"用户问题：{state['user_message']}\n"
                        f"工具结果：{json.dumps(tool_result, ensure_ascii=False, indent=2)}"
                    ),
                },
            ]
            try:
                return {"final_answer": self.llm_client.chat(messages, max_tokens=1200)}
            except Exception:
                return {
                    "final_answer": json.dumps(tool_result, ensure_ascii=False, indent=2)
                }

        return {"final_answer": "我还不能确定下一步动作。"}

    def _confirmation_message(self, state):
        action = state.get("proposed_action") or {}
        reason = action.get("reason") or "该操作会启动长时间任务，需要确认。"
        return (
            f"{reason}\n\n"
            f"待确认动作：{action.get('action')}\n"
            f"参数：{json.dumps(action.get('args') or {}, ensure_ascii=False)}"
        )

    def _stream_tool_answer(self, user_message, tool_result):
        messages = [
            {
                "role": "system",
                "content": (
                    "你是 VIC 水文建模助手。请基于工具结果用中文简洁回答，不要编造。"
                    "解释参数时必须使用工具结果中的 parameter_definitions。"
                    "NSE 是 Nash-Sutcliffe efficiency，用于评价模拟流量与观测流量吻合程度，不要称为决定系数，也不要写成流速。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：{user_message}\n"
                    f"工具结果：{json.dumps(tool_result, ensure_ascii=False, indent=2)}"
                ),
            },
        ]
        try:
            yield from self.llm_client.stream_chat(messages, max_tokens=1200)
        except Exception:
            yield json.dumps(tool_result, ensure_ascii=False, indent=2)

    def _format_tool_result(self, user_message, tool_result):
        if isinstance(tool_result, dict) and {"inputs", "stages", "outputs"}.issubset(tool_result):
            inputs = tool_result["inputs"]
            stages = tool_result["stages"]
            outputs = tool_result["outputs"]
            return (
                "当前 run 状态如下：\n\n"
                f"边界文件：{'已上传' if inputs.get('boundary_uploaded') else '缺失'}\n"
                f"出口文件：{'已上传' if inputs.get('outlet_uploaded') else '缺失'}\n"
                f"观测文件：{'已上传' if inputs.get('observation_uploaded') else '缺失'}\n"
                f"自动建模：{stages.get('auto_modeling')}\n"
                f"模型运行：{stages.get('vic_run')}\n"
                f"参数率定：{stages.get('calibration')}\n"
                f"模型输入：{'已生成' if outputs.get('model_inputs_ready') else '未生成'}\n"
                f"月径流输出：{'存在' if outputs.get('monthly_flow_exists') else '缺失'}\n"
                f"最优 NSE：{outputs.get('best_nse')}"
            )
        if isinstance(tool_result, dict) and tool_result.get("history_exists"):
            return (
                "当前率定摘要如下：\n\n"
                f"迭代次数：{tool_result.get('iterations')}\n"
                f"最新 NSE：{tool_result.get('latest_nse')}\n"
                f"最优 NSE：{tool_result.get('best_nse')}\n"
                f"最优迭代：{tool_result.get('best_iteration')}\n"
                f"最优参数：{json.dumps(tool_result.get('best_params'), ensure_ascii=False)}"
            )
        return json.dumps(tool_result, ensure_ascii=False, indent=2)
