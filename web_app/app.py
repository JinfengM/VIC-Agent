import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vic_auto_modeling.flow.routing import FILL_ALGORITHMS  # noqa: E402
from vic_auto_modeling.agent.orchestrator import VicAgent  # noqa: E402
from vic_auto_modeling.agent.state import get_run_summary  # noqa: E402
from web_app.jobs import (  # noqa: E402
    context_for,
    copy_observation_to_run,
    has_model_inputs,
    has_model_output,
    history_path,
    read_status,
    run_auto_modeling_job,
    run_calibration_job,
    run_vic_job,
    save_boundary_and_outlet,
    start_job,
)


CALIBRATION_PARAMETERS = [
    {
        "优化变量": "x1",
        "VIC 参数": "b_infilt",
        "物理含义": "可变下渗曲线参数，控制降水在地表径流与入渗之间的分配；值越大，空间下渗能力差异越强。",
        "取值范围": "0.01 - 1.00",
    },
    {
        "优化变量": "x2",
        "VIC 参数": "Ds",
        "物理含义": "非线性基流参数，表示达到 Ws 阈值时的基流系数占 Dsmax 的比例。",
        "取值范围": "0.01 - 1.00",
    },
    {
        "优化变量": "x3",
        "VIC 参数": "Dsmax",
        "物理含义": "最大基流速度，控制深层土壤向河道贡献基流的最大能力。",
        "取值范围": "0.10 - 30.00",
    },
    {
        "优化变量": "x4",
        "VIC 参数": "Ws",
        "物理含义": "非线性基流启动阈值，表示土壤含水量达到最大含水量的该比例后基流快速增加。",
        "取值范围": "0.01 - 1.00",
    },
    {
        "优化变量": "x5",
        "VIC 参数": "depth[1]",
        "物理含义": "第二层土壤厚度，影响中层土壤蓄水、蒸散和径流调节。",
        "取值范围": "0.10 - 1.50 m",
    },
    {
        "优化变量": "x6",
        "VIC 参数": "depth[2]",
        "物理含义": "第三层土壤厚度，影响深层蓄水和基流过程。",
        "取值范围": "0.10 - 1.50 m",
    },
]


def plot_nse_history(history_df):
    import matplotlib.pyplot as plt
    import numpy as np
    import scienceplots  # noqa: F401

    with plt.style.context(["science","no-latex"]), plt.rc_context(
        {"axes.formatter.use_mathtext": False, "text.usetex": False}
    ):
        fig, ax = plt.subplots(figsize=(8, 3.2), constrained_layout=True)
        ax.plot(
            history_df["iteration"],
            history_df["nse"],
            marker="o",
            linewidth=1.5,
            label="Current NSE",
        )
        ax.plot(
            history_df["iteration"],
            history_df["best_nse"],
            marker="s",
            linewidth=1.5,
            label="Best NSE",
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("NSE")
        iterations = history_df["iteration"].astype(int).tolist()
        ax.set_xticks(iterations)
        ax.set_xticklabels([str(value) for value in iterations])
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.01, 0.1))
        ax.grid(True, alpha=0.3)
        ax.legend()
        return fig


def append_agent_chat(context, role, content):
    path = context.log_path("agent_chat.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"role": role, "content": content, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def stream_chatgpt_style(chunks, delay=0.012):
    for chunk in chunks:
        if not chunk:
            continue
        for char in chunk:
            yield char
            time.sleep(delay)


def install_auto_scroll():
    components.html(
        """
        <script>
        const getParentDocument = () => {
            try {
                return window.parent.document;
            } catch (error) {
                return null;
            }
        };

        const isScrollable = (element) => {
            if (!element) {
                return false;
            }
            const style = window.parent.getComputedStyle(element);
            const canScroll = /(auto|scroll|overlay)/.test(style.overflowY);
            return canScroll && element.scrollHeight > element.clientHeight + 8;
        };

        const findScrollRoot = (doc) => {
            const selectors = [
                ".stMain",
                "section.main",
                "section[data-testid='stAppViewContainer']",
                "div[data-testid='stAppViewContainer']",
                "main",
            ];
            const candidates = selectors
                .flatMap((selector) => Array.from(doc.querySelectorAll(selector)))
                .filter(isScrollable);
            if (candidates.length > 0) {
                return candidates.reduce((best, element) =>
                    element.scrollHeight > best.scrollHeight ? element : best
                );
            }

            const allScrollable = Array.from(doc.querySelectorAll("body *")).filter(isScrollable);
            if (allScrollable.length > 0) {
                return allScrollable.reduce((best, element) =>
                    element.scrollHeight > best.scrollHeight ? element : best
                );
            }

            return doc.scrollingElement || doc.documentElement || doc.body;
        };

        const scrollToBottom = () => {
            const doc = getParentDocument();
            if (!doc) {
                return;
            }
            const scrollRoot = findScrollRoot(doc);
            const state = window.parent.__vicAutoScrollState || {};
            if (state.disabled && !state.forceNext) {
                return;
            }
            state.root = scrollRoot;
            state.programmatic = true;
            scrollRoot.scrollTo({ top: scrollRoot.scrollHeight, behavior: "smooth" });
            state.lastTop = scrollRoot.scrollTop;
            state.forceNext = false;
            window.parent.__vicAutoScrollState = state;
            window.parent.setTimeout(() => {
                const current = window.parent.__vicAutoScrollState;
                if (current) {
                    current.programmatic = false;
                    current.lastTop = scrollRoot.scrollTop;
                }
            }, 300);
        };

        const installScrollListener = () => {
            const doc = getParentDocument();
            if (!doc) {
                return;
            }
            const scrollRoot = findScrollRoot(doc);
            const state = window.parent.__vicAutoScrollState || {};
            if (state.root && state.root !== scrollRoot && state.onScroll) {
                state.root.removeEventListener("scroll", state.onScroll);
                state.onScroll = null;
            }
            state.root = scrollRoot;
            if (!state.onScroll) {
                state.lastTop = scrollRoot.scrollTop;
                state.onScroll = () => {
                    const current = window.parent.__vicAutoScrollState;
                    if (!current || current.programmatic) {
                        return;
                    }
                    const root = current.root;
                    const distanceFromBottom =
                        root.scrollHeight - root.clientHeight - root.scrollTop;
                    if (root.scrollTop < current.lastTop - 24 && distanceFromBottom > 80) {
                        current.disabled = true;
                    } else if (distanceFromBottom < 80) {
                        current.disabled = false;
                    }
                    current.lastTop = root.scrollTop;
                };
                scrollRoot.addEventListener("scroll", state.onScroll, { passive: true });
            }
            state.disabled = false;
            state.forceNext = true;
            window.parent.__vicAutoScrollState = state;
        };

        const installMutationObserver = () => {
            const doc = getParentDocument();
            if (!doc) {
                return;
            }
            if (window.parent.__vicAutoScrollObserver) {
                window.parent.__vicAutoScrollObserver.disconnect();
            }
            const target = doc.querySelector(".stMain")
                || doc.querySelector("section[data-testid='stAppViewContainer']")
                || doc.querySelector("div[data-testid='stAppViewContainer']")
                || doc.body;
            window.parent.__vicAutoScrollObserver = new window.parent.MutationObserver(scrollToBottom);
            window.parent.__vicAutoScrollObserver.observe(target, {
                childList: true,
                subtree: true,
                characterData: true,
            });
        };

        installScrollListener();
        scrollToBottom();
        installMutationObserver();
        [80, 250, 600, 1200].forEach((delay) => {
            window.parent.setTimeout(scrollToBottom, delay);
        });
        window.parent.setTimeout(() => {
            if (window.parent.__vicAutoScrollObserver) {
                window.parent.__vicAutoScrollObserver.disconnect();
                window.parent.__vicAutoScrollObserver = null;
            }
        }, 30000);
        </script>
        """,
        height=0,
        width=0,
    )


def install_back_to_top():
    components.html(
        """
        <script>
        const getParentDocument = () => {
            try {
                return window.parent.document;
            } catch (error) {
                return null;
            }
        };

        const isScrollable = (element) => {
            if (!element) {
                return false;
            }
            const style = window.parent.getComputedStyle(element);
            const canScroll = /(auto|scroll|overlay)/.test(style.overflowY);
            return canScroll && element.scrollHeight > element.clientHeight + 8;
        };

        const findScrollRoot = (doc) => {
            const selectors = [
                ".stMain",
                "section.main",
                "section[data-testid='stAppViewContainer']",
                "div[data-testid='stAppViewContainer']",
                "main",
            ];
            const candidates = selectors
                .flatMap((selector) => Array.from(doc.querySelectorAll(selector)))
                .filter(isScrollable);
            if (candidates.length > 0) {
                return candidates.reduce((best, element) =>
                    element.scrollHeight > best.scrollHeight ? element : best
                );
            }

            const allScrollable = Array.from(doc.querySelectorAll("body *")).filter(isScrollable);
            if (allScrollable.length > 0) {
                return allScrollable.reduce((best, element) =>
                    element.scrollHeight > best.scrollHeight ? element : best
                );
            }

            return doc.scrollingElement || doc.documentElement || doc.body;
        };

        const doc = getParentDocument();
        if (doc && !doc.getElementById("vic-back-to-top")) {
            const button = doc.createElement("button");
            button.id = "vic-back-to-top";
            button.type = "button";
            button.title = "返回顶部";
            button.setAttribute("aria-label", "返回顶部");
            button.textContent = "↑";
            button.style.position = "fixed";
            button.style.right = "24px";
            button.style.bottom = "24px";
            button.style.zIndex = "2147483647";
            button.style.width = "44px";
            button.style.height = "44px";
            button.style.border = "1px solid rgba(15, 23, 42, 0.16)";
            button.style.borderRadius = "999px";
            button.style.background = "#0f172a";
            button.style.color = "#ffffff";
            button.style.fontSize = "24px";
            button.style.lineHeight = "40px";
            button.style.cursor = "pointer";
            button.style.boxShadow = "0 8px 24px rgba(15, 23, 42, 0.24)";
            button.style.opacity = "0";
            button.style.pointerEvents = "none";
            button.style.transition = "opacity 120ms ease, transform 120ms ease";

            const setVisible = () => {
                const scrollRoot = findScrollRoot(doc);
                const visible = scrollRoot.scrollTop > 320;
                button.style.opacity = visible ? "0.92" : "0";
                button.style.pointerEvents = visible ? "auto" : "none";
                button.style.transform = visible ? "translateY(0)" : "translateY(8px)";
            };

            const bindScrollRoot = () => {
                const scrollRoot = findScrollRoot(doc);
                if (button.__vicScrollRoot === scrollRoot) {
                    setVisible();
                    return;
                }
                if (button.__vicScrollRoot && button.__vicSetVisible) {
                    button.__vicScrollRoot.removeEventListener("scroll", button.__vicSetVisible);
                }
                button.__vicScrollRoot = scrollRoot;
                button.__vicSetVisible = setVisible;
                scrollRoot.addEventListener("scroll", setVisible, { passive: true });
                setVisible();
            };

            button.addEventListener("click", () => {
                const scrollRoot = findScrollRoot(doc);
                scrollRoot.scrollTo({ top: 0, behavior: "smooth" });
            });

            doc.body.appendChild(button);
            bindScrollRoot();
            window.parent.setInterval(bindScrollRoot, 1500);
        }
        </script>
        """,
        height=0,
        width=0,
    )


def render_run_summary(summary):
    inputs = summary.get("inputs", {})
    stages = summary.get("stages", {})
    outputs = summary.get("outputs", {})
    rows = [
        ("边界文件", "已上传" if inputs.get("boundary_uploaded") else "缺失"),
        ("出口文件", "已上传" if inputs.get("outlet_uploaded") else "缺失"),
        ("观测文件", "已上传" if inputs.get("observation_uploaded") else "缺失"),
        ("自动建模", stages.get("auto_modeling", "idle")),
        ("模型运行", stages.get("vic_run", "idle")),
        ("参数率定", stages.get("calibration", "idle")),
        ("模型输入", "已生成" if outputs.get("model_inputs_ready") else "未生成"),
        ("月径流输出", "存在" if outputs.get("monthly_flow_exists") else "缺失"),
        ("最优 NSE", outputs.get("best_nse")),
    ]
    html_rows = "".join(
        f"<tr><th>{name}</th><td>{'' if value is None else value}</td></tr>"
        for name, value in rows
    )
    st.markdown(
        f"<table class='history-table'>{html_rows}</table>",
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="VIC Auto Modeling", layout="wide")
install_back_to_top()
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
    }
    table.history-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
    }
    table.param-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.92rem;
        margin-bottom: 1rem;
    }
    table.history-table th,
    table.history-table td,
    table.param-table th,
    table.param-table td {
        border-bottom: 1px solid #e5e7eb;
        padding: 0.35rem 0.5rem;
    }
    table.history-table th,
    table.history-table td {
        text-align: right;
    }
    table.param-table th,
    table.param-table td {
        text-align: left;
        vertical-align: top;
    }
    table.history-table th:first-child,
    table.history-table td:first-child {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("VIC Auto Modeling")

with st.sidebar:
    run_id = st.text_input("Run ID", value="web_demo")
    source_dir = st.text_input(
        "VIC runtime directory", value=os.getenv("VIC_SOURCE_DIR", "")
    )
    processes = st.number_input("MPI processes", min_value=1, max_value=128, value=12, step=1)
    station_name = st.text_input("Station name", value="luanx")

if not run_id:
    st.stop()

context = context_for(run_id, project_root=PROJECT_ROOT)


def show_status(name):
    status = read_status(context, name)
    state = status.get("state", "idle")
    message = status.get("message", state)
    if state == "complete":
        st.success(message)
    elif state == "failed":
        st.error(message)
        with st.expander("Traceback"):
            st.code(status.get("traceback", ""))
    elif state == "running":
        st.info(message)
    else:
        st.caption("未开始")
    return status


tab_modeling, tab_run, tab_calibration, tab_agent = st.tabs(
    ["自动建模", "模型运行", "参数率定", "智能体助手"]
)

with tab_modeling:
    st.subheader("输入边界与出口")
    boundary_file = st.file_uploader("Boundary shapefile zip", type=["zip"], key="boundary_zip")
    outlet_file = st.file_uploader("Outlet shapefile zip", type=["zip"], key="outlet_zip")
    fill_algorithm = st.selectbox(
        "Flow fill algorithm",
        options=sorted(FILL_ALGORITHMS),
        index=sorted(FILL_ALGORITHMS).index("priority-flood"),
    )

    if st.button("开始自动建模", disabled=not (boundary_file and outlet_file)):
        boundary_zip, outlet_zip = save_boundary_and_outlet(
            boundary_file, outlet_file, run_id, project_root=PROJECT_ROOT
        )
        start_job(
            f"{run_id}:auto_modeling",
            run_auto_modeling_job,
            run_id,
            boundary_zip,
            outlet_zip,
            fill_algorithm,
            PROJECT_ROOT,
        )
        st.rerun()

    auto_status = show_status("auto_modeling")
    if has_model_inputs(run_id, project_root=PROJECT_ROOT):
        st.write("模型配置已生成。")
        st.code(str(context.output_path("model", "chanliu_input.txt")))
        st.code(str(context.output_path("model", "rout_input.txt")))

with tab_run:
    st.subheader("运行 VIC")
    can_run_model = has_model_inputs(run_id, project_root=PROJECT_ROOT)
    if st.button("运行 VIC 模型", disabled=not can_run_model):
        start_job(
            f"{run_id}:vic_run",
            run_vic_job,
            run_id,
            source_dir,
            int(processes),
            PROJECT_ROOT,
        )
        st.rerun()

    vic_status = show_status("vic_run")
    stdout = context.output_path("model", "vic_stdout.log")
    stderr = context.output_path("model", "vic_stderr.log")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("stdout")
        st.text(stdout.read_text(encoding="utf-8", errors="replace")[-4000:] if stdout.exists() else "")
    with col2:
        st.caption("stderr")
        st.text(stderr.read_text(encoding="utf-8", errors="replace")[-4000:] if stderr.exists() else "")

with tab_calibration:
    st.subheader("参数自动率定")
    st.markdown(
        pd.DataFrame(CALIBRATION_PARAMETERS).to_html(index=False, classes="param-table"),
        unsafe_allow_html=True,
    )
    model_ready = has_model_output(run_id, station_name=station_name, project_root=PROJECT_ROOT)
    observation_file = st.file_uploader("Observation CSV", type=["csv"], key="observation_csv")
    iterations = st.number_input("Iterations", min_value=1, max_value=10000, value=10, step=1)
    random_state = st.number_input("Random state", min_value=0, max_value=100000, value=1, step=1)
    xi = st.number_input("Probability-of-improvement xi", min_value=0.0, value=0.1, step=0.01)

    if st.button("开始参数率定", disabled=not (model_ready and observation_file)):
        observation_path = copy_observation_to_run(observation_file, run_id, project_root=PROJECT_ROOT)
        start_job(
            f"{run_id}:calibration",
            run_calibration_job,
            run_id,
            observation_path,
            int(iterations),
            source_dir,
            int(processes),
            station_name,
            int(random_state),
            float(xi),
            PROJECT_ROOT,
        )
        st.rerun()

    calibration_status = show_status("calibration")
    if calibration_status.get("state") == "running":
        progress_total = max(int(calibration_status.get("iterations", iterations)), 1)
        progress_now = int(calibration_status.get("iteration", 0))
        st.progress(min(progress_now / progress_total, 1.0))

    history = history_path(context)
    if history.exists():
        history_df = pd.read_csv(history)
        st.pyplot(plot_nse_history(history_df), clear_figure=True)
        st.markdown(
            history_df.tail(20).to_html(index=False, classes="history-table"),
            unsafe_allow_html=True,
        )

    plot_path = calibration_status.get("plot_path")
    if plot_path and Path(plot_path).exists():
        st.image(plot_path, caption="Observed vs simulated monthly streamflow")
    else:
        default_plot = context.output_path("model", "chanliu_result", f"{station_name}.png")
        if default_plot.exists():
            st.image(str(default_plot), caption="Observed vs simulated monthly streamflow")

with tab_agent:
    st.subheader("VIC 水文建模智能体")
    install_auto_scroll()
    agent = VicAgent()
    summary = get_run_summary(run_id, project_root=PROJECT_ROOT)
    render_run_summary(summary)

    chat_key = f"agent_chat:{run_id}"
    pending_key = f"agent_pending:{run_id}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    for message in st.session_state[chat_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("询问当前进度、诊断日志、解释率定结果，或请求启动下一阶段任务")
    if prompt:
        st.session_state[chat_key].append({"role": "user", "content": prompt})
        append_agent_chat(context, "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            install_auto_scroll()
            metadata = {}
            chunks = agent.stream_interaction(
                run_id,
                prompt,
                project_root=PROJECT_ROOT,
                metadata=metadata,
            )
            response_text = st.write_stream(stream_chatgpt_style(chunks))
            install_auto_scroll()
        st.session_state[chat_key].append({"role": "assistant", "content": response_text})
        append_agent_chat(context, "assistant", response_text)
        if metadata.get("confirmation_required"):
            st.session_state[pending_key] = {
                "token": metadata.get("confirmation_token"),
                "action": metadata.get("proposed_action"),
            }
        st.rerun()

    pending = st.session_state.get(pending_key)
    if pending:
        st.warning("智能体提出了需要确认的动作。")
        st.code(json.dumps(pending["action"], ensure_ascii=False, indent=2))
        if st.button("确认执行智能体建议动作"):
            response = agent.confirm(run_id, pending["token"], project_root=PROJECT_ROOT)
            st.session_state[chat_key].append(
                {"role": "assistant", "content": response.message}
            )
            append_agent_chat(context, "assistant", response.message)
            st.session_state.pop(pending_key, None)
            st.rerun()

if any(
    read_status(context, name).get("state") == "running"
    for name in ("auto_modeling", "vic_run", "calibration")
):
    time.sleep(3)
    st.rerun()
