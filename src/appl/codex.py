"""Codex app-server backend for the APPL tool loop.

Runs one APPL agent session as one Codex thread: the stage prompt replaces Codex's base
instructions, the session's APPL tools are the thread's only tools (Codex's shell, file,
browser, plugin and other built-in tools are disabled), and Codex's own ChatGPT login serves
the model. Credentials never pass through this module.

Differences from the Responses loop in ``appl.agent`` (recorded in every client
configuration, so a Codex run never shares an agent identity with an API run):

* Codex keeps the conversation and compacts it when needed. ``context_builder`` projections
  (the runtime agent's six-turn window) are not applied.
* A model turn that ends with a message instead of a tool call is continued with the fixed
  user message ``CONTINUE``; the Responses loop re-sent the same history instead.
* Arguments that fail a tool's JSON schema are returned to the model as a tool error.
  Strict function calling made them impossible in the Responses loop.
* ``max_output_tokens`` is left to Codex.

Sampling requests are counted as tool calls plus model turns that end without one, against
the same ``max_api_calls`` budget. Once ``tools.done()`` holds, the turn is interrupted so no
further sampling happens, as the Responses loop stops before its next request.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import select
import subprocess
import time

import jsonschema

from .journal import atomic, encode

CONTINUE = "Continue with the tools."
# Codex features that would give the agent capabilities outside the APPL tools.
DISABLED_FEATURES = (
    "apps", "browser_use", "browser_use_external", "browser_use_full_cdp_access",
    "computer_use", "image_generation", "goals", "hooks", "memories", "multi_agent",
    "multi_agent_v2", "plugins", "remote_plugin", "shell_tool", "shell_snapshot",
    "unified_exec", "view_image", "skill_search", "sleep_tool", "tool_suggest",
    "workspace_dependencies", "worktrees", "skill_mcp_dependency_install",
)


class CodexStop(RuntimeError):
    """The plan usage guard refused or ended a session."""


class CodexClient:
    provenance = "codex_app_server"
    codex = True

    def __init__(self, journal_root, *, model="gpt-5.6-sol", reasoning_effort="xhigh",
                 timeout_seconds=1800, max_used_percent=None):
        if not model or not reasoning_effort or timeout_seconds <= 0:
            raise ValueError("A model, reasoning effort and positive timeout are required")
        self.root = Path(journal_root)
        self.model = model
        self.reasoning = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.max_used = float(max_used_percent if max_used_percent is not None
                              else os.environ.get("APPL_CODEX_MAX_USED", 80))
        self.version = subprocess.run(["codex", "--version"], capture_output=True, text=True,
                                      check=True).stdout.strip()

    def configuration(self):
        return dict(backend="codex_app_server", codex_version=self.version, model=self.model,
                    reasoning_effort=self.reasoning, timeout_seconds=self.timeout_seconds,
                    disabled_features=list(DISABLED_FEATURES), continue_message=CONTINUE,
                    context_projection=False, schema_errors_returned=True)


class _Server:
    """A JSON-RPC connection to a private ``codex app-server`` process."""

    def __init__(self, cwd, log_path):
        args = ["codex", "app-server", "-c", 'web_search="disabled"', "-c", "project_doc_max_bytes=0"]
        for feature in DISABLED_FEATURES:
            args += ["--disable", feature]
        self.log = open(log_path, "a")
        self.proc = subprocess.Popen(args, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, text=True, bufsize=1)
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.stdin, self.stdout = self.proc.stdin, self.proc.stdout
        self.next_id = 0
        self.backlog = []

    def send(self, message):
        self.stdin.write(json.dumps(message) + "\n")
        self.stdin.flush()

    def request(self, method, params, timeout=120):
        self.next_id += 1
        ident = self.next_id
        self.send(dict(jsonrpc="2.0", id=ident, method=method, params=params))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            message = self._read_stream(deadline - time.monotonic())
            if message is None:
                continue
            if message.get("id") == ident and "method" not in message:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message["result"]
            self.backlog.append(message)
        raise TimeoutError(f"No answer to {method}")

    def read(self, timeout):
        """Next server message, oldest first, including those set aside by ``request``."""
        if self.backlog:
            return self.backlog.pop(0)
        return self._read_stream(timeout)

    def _read_stream(self, timeout):
        ready, _, _ = select.select([self.stdout], [], [], max(0.0, min(timeout, 5.0)))
        if not ready:
            if self.proc.poll() is not None:
                raise RuntimeError("codex app-server exited")
            return None
        line = self.stdout.readline()
        if not line:
            raise RuntimeError("codex app-server closed its output")
        return json.loads(line)

    def respond(self, ident, result=None, error=None):
        message: dict = dict(jsonrpc="2.0", id=ident)
        if error is not None:
            message["error"] = error
        else:
            message["result"] = result
        self.send(message)

    def close(self):
        try:
            self.proc.kill()
            self.proc.wait(timeout=10)
        finally:
            self.log.close()


def plan_usage(server):
    limits = server.request("account/rateLimits/read", {})
    primary = limits["rateLimits"]["primary"]
    return dict(used_percent=float(primary["usedPercent"]), resets_at=primary["resetsAt"],
                ordinary_usage_allowed=bool(limits.get("ordinaryUsageAllowed", True)))


def _content(result):
    """Map a tool result to dynamic-tool output content items."""
    if not isinstance(result, list):
        return [dict(type="inputText", text=encode(result))]
    items = []
    for part in result:
        if part.get("type") == "input_image":
            items.append(dict(type="inputImage", imageUrl=part["image_url"]))
        else:
            items.append(dict(type="inputText", text=part.get("text", encode(part))))
    return items


def run_session(loop, message, budget):
    """Run ``loop.tools`` to completion in one Codex thread; mirrors ``AgentLoop.run``."""
    j, tools, client = loop.journal, loop.tools, loop.client
    if j.get("codex_session_started") and not tools.done():
        raise RuntimeError("An earlier Codex session ended unfinished; no automatic retry")
    if tools.done():
        return j.get("frozen")
    root = j.root / "codex"
    root.mkdir(parents=True, exist_ok=True)
    workspace = root / "empty_workspace"
    workspace.mkdir(exist_ok=True)
    transcript = root / "transcript.jsonl"
    schemas = tools.schemas()
    definitions = {s["name"]: s for s in schemas}
    server = _Server(workspace, root / "app_server.log")
    calls = turns = 0
    usage = None

    def note(kind, **payload):
        with transcript.open("a") as stream:
            stream.write(encode(dict(kind=kind, time=time.time(), **payload)) + "\n")

    try:
        server.request("initialize", dict(clientInfo=dict(name="appl", version="1"),
                                          capabilities=dict(experimentalApi=True, requestAttestation=False)))
        server.send(dict(jsonrpc="2.0", method="initialized"))
        quota = plan_usage(server)
        j.event("codex_plan_usage", stage="start", **quota)
        if quota["used_percent"] >= client.max_used or not quota["ordinary_usage_allowed"]:
            raise CodexStop(f"Codex plan usage {quota['used_percent']}% >= {client.max_used}%; not starting")
        thread = server.request("thread/start", dict(
            model=client.model, baseInstructions=loop.prompt, cwd=str(workspace),
            approvalPolicy="never", sandbox="read-only", ephemeral=True, environments=[],
            dynamicTools=[dict(type="function", name=s["name"], description=s["description"],
                               inputSchema=s["parameters"]) for s in schemas]))
        thread_id = thread["thread"]["id"]
        j.set("codex_session_started", dict(thread_id=thread_id, client=client.configuration()))
        atomic(root / "session.json", dict(thread_id=thread_id, prompt=loop.prompt, tools=schemas,
                                           message=message, client=client.configuration()))

        def start_turn(text):
            server.request("turn/start", dict(threadId=thread_id, effort=client.reasoning,
                                              input=[dict(type="text", text=text, text_elements=[])]))

        start_turn(encode(message))
        turn_id = None
        last_activity = time.monotonic()
        while True:
            if time.monotonic() - last_activity > client.timeout_seconds:
                raise TimeoutError("Codex session idle beyond the configured timeout")
            event = server.read(client.timeout_seconds)
            if event is None:
                continue
            last_activity = time.monotonic()
            method = event.get("method")
            params = event.get("params", {})
            if method == "item/tool/call":
                if calls >= budget:
                    server.respond(event["id"], error=dict(code=-32000, message="API call budget exhausted"))
                    raise ValueError("API call budget exhausted")
                calls += 1
                name, arguments = params["tool"], params["arguments"]
                note("tool_call", call_id=params["callId"], tool=name, arguments=arguments)
                if name not in definitions:
                    result, success = dict(error="Unknown tool: " + str(name)), False
                else:
                    try:
                        jsonschema.validate(arguments, definitions[name]["parameters"])
                    except jsonschema.ValidationError as error:
                        result, success = dict(error="Arguments do not match the tool schema: " + error.message), False
                    else:
                        result = tools.execute(dict(type="function_call", name=name, call_id=params["callId"],
                                                    arguments=json.dumps(arguments)))
                        success = True
                note("tool_result", call_id=params["callId"], success=success,
                     result=result if not isinstance(result, list) else "[content items]")
                server.respond(event["id"], dict(contentItems=_content(result), success=success))
                if tools.done():
                    if turn_id:
                        server.request("turn/interrupt", dict(threadId=thread_id, turnId=turn_id))
                    break
            elif "id" in event and method:
                # Approvals or other requests for capabilities outside the APPL tools.
                note("declined_request", method=method)
                server.respond(event["id"], error=dict(code=-32601, message="Not available in APPL sessions"))
            elif method == "turn/started":
                turn_id = params["turn"]["id"]
            elif method == "item/completed":
                item = params.get("item", {})
                if item.get("type") in ("agentMessage", "reasoning", "contextCompaction"):
                    note(item["type"], item=item)
            elif method == "thread/tokenUsage/updated":
                usage = params.get("tokenUsage")
            elif method == "account/rateLimits/updated":
                limits = params.get("rateLimits", {}).get("primary") or {}
                if limits and float(limits.get("usedPercent", 0)) >= 99:
                    raise CodexStop("Codex plan window nearly exhausted during a session")
            elif method == "turn/completed":
                turns += 1
                status = params["turn"].get("status")
                note("turn_completed", status=status, error=params["turn"].get("error"))
                if status == "failed":
                    raise RuntimeError("Codex turn failed: " + encode(params["turn"].get("error")))
                if tools.done():
                    break
                if calls + turns >= budget:
                    raise ValueError("API call budget exhausted")
                start_turn(CONTINUE)
            elif method == "error":
                note("error", params=params)
        return j.get("frozen")
    finally:
        try:
            quota = plan_usage(server)
            j.event("codex_plan_usage", stage="end", **quota)
        except Exception:  # the session outcome matters more than the final usage read
            quota = None
        j.event("codex_session", tool_calls=calls, continued_turns=turns, usage=usage,
                done=bool(tools.done()), plan_usage=quota)
        server.close()
