"""Phase 10: documentation gates.

Pure text inspection of README.md and docs/*.md, cross-checked against the code they
describe: settings, routes, the job state machine, error types, metric names, limits and
CLI flags. Nothing is rendered, fetched or executed; when a doc and the code disagree,
one of these tests names the mismatch.
"""

import json
import re
import shlex
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
import yaml
from pydantic import SecretStr

from multimodal_vision_pipeline.api import create_app
from multimodal_vision_pipeline.api.health import Readiness
from multimodal_vision_pipeline.api.routes import IDEMPOTENCY_HEADER
from multimodal_vision_pipeline.core.errors import PROBLEM_TYPE_PREFIX, PipelineError
from multimodal_vision_pipeline.core.job_manager import TRANSITIONS
from multimodal_vision_pipeline.core.rasterizer import DEFAULT_CHUNK_SIZE, POPPLER_TIMEOUT_SECONDS
from multimodal_vision_pipeline.core.vision_builder import (
    DEFAULT_MAX_TOTAL_IMAGE_TOKENS,
    MAX_IMAGES_PER_REQUEST,
    MODEL_IMAGE_LIMITS,
)
from multimodal_vision_pipeline.evals.cli import build_parser
from multimodal_vision_pipeline.evals.perturbation import PROFILES
from multimodal_vision_pipeline.observability.metrics import PipelineMetrics
from multimodal_vision_pipeline.schemas import (
    IngestionOptions,
    JobStatus,
    ProblemDetails,
    ReconciliationPolicy,
    ReconciliationRule,
    RoundingMode,
    VisionSettings,
)
from multimodal_vision_pipeline.schemas.document import (
    DEFAULT_DPI,
    DEFAULT_MAX_PAGES,
    MAX_DPI,
    MAX_PAGE_LIMIT,
    MAX_PIXELS_PER_SIDE,
    MIN_DPI,
)
from multimodal_vision_pipeline.tools.vision_client import (
    API_VERSION,
    FALLBACK_BETA,
    RETRYABLE_STATUSES,
    RetryPolicy,
)

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
ARCHITECTURE = ROOT / "docs" / "architecture.md"
RUNBOOK = ROOT / "docs" / "runbook.md"
DOCUMENTS = (README, ARCHITECTURE, RUNBOOK)
ENV_PREFIX = VisionSettings.model_config.get("env_prefix", "")
LOCAL_BASE = "http://127.0.0.1:8000"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodeBlock:
    path: Path
    line: int
    language: str
    body: str

    @property
    def where(self) -> str:
        return f"{self.path.name}:{self.line}"


def code_blocks(text: str, path: Path = Path("<text>")) -> list[CodeBlock]:
    """Fenced blocks; an unterminated fence is an error, not an empty block."""
    blocks: list[CodeBlock] = []
    opened: tuple[int, str, list[str]] | None = None
    for number, line in enumerate(text.splitlines(), start=1):
        fence = re.match(r"^\s*```(.*)$", line)
        if fence is None:
            if opened is not None:
                opened[2].append(line)
            continue
        if opened is None:
            opened = (number, fence.group(1).strip(), [])
        else:
            assert not fence.group(1).strip(), f"{path.name}:{number}: closing fence has text"
            blocks.append(CodeBlock(path, opened[0], opened[1], "\n".join(opened[2])))
            opened = None
    assert opened is None, f"{path.name}:{opened[0]}: unclosed code fence"
    return blocks


def all_blocks(language: str | None = None) -> list[CodeBlock]:
    blocks = [block for path in DOCUMENTS for block in code_blocks(_read(path), path)]
    return [block for block in blocks if language is None or block.language == language]


def table_rows(text: str, heading: str) -> list[list[str]]:
    """Cells of every table row in the section under ``heading`` (header and rule excluded)."""
    section = text.split(f"\n{heading}\n", 1)[1]
    section = re.split(r"\n#{1,6} ", section, maxsplit=1)[0]
    rows: list[list[str]] = []
    for line in section.splitlines():
        if not line.startswith("|") or re.fullmatch(r"\|[\s:|-]+\|", line):
            continue
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows[1:] if rows else rows


def _code(cell: str) -> str:
    return cell.strip().strip("`")


def heading_anchor(heading: str) -> str:
    """GitHub's heading slug: lower case, punctuation dropped, spaces to hyphens."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


# ---------------------------------------------------------------------------
# Mermaid well-formedness
# ---------------------------------------------------------------------------

_DIAGRAM = re.compile(
    r"^(?:(?:flowchart|graph) (?:TB|TD|BT|LR|RL)|sequenceDiagram|stateDiagram-v2)$"
)
_PAIRS = {")": "(", "]": "[", "}": "{"}
_SEQUENCE_KEYWORDS = re.compile(
    r"^(?:participant|actor|autonumber|Note|activate|deactivate|else|and)\b"
)
_SEQUENCE_BLOCKS = re.compile(r"^(?:loop|alt|opt|par|critical|break|rect)\b")
_SEQUENCE_MESSAGE = re.compile(r"^(\w+)\s*(?:-->>|->>|-->|->|--x|-x|--\)|-\))\s*(\w+)\s*:\s*\S.*$")
_STATE_TRANSITION = re.compile(r"^(\[\*\]|\w+)\s*-->\s*(\[\*\]|\w+)(?:\s*:\s*\S.*)?$")
_DANGLING_EDGE = re.compile(r"(?:-->|-\.->|==>|->>|-->>|---)\s*(?:\|[^|]*\|)?\s*$")


def _bracket_problems(line: str) -> list[str]:
    if line.count('"') % 2:
        return [f"unbalanced quotes: {line}"]
    stack: list[str] = []
    outside_quotes = re.sub(r'"[^"]*"', '""', line)
    for char in outside_quotes:
        if char in "([{":
            stack.append(char)
        elif char in _PAIRS and (not stack or stack.pop() != _PAIRS[char]):
            return [f"unbalanced {char!r}: {line}"]
    return [f"unclosed {stack[-1]!r}: {line}"] if stack else []


def _flowchart_problems(statements: list[str]) -> list[str]:
    opened = sum(line.startswith("subgraph ") for line in statements)
    closed = sum(line == "end" for line in statements)
    return [] if opened == closed else [f"{opened} subgraph(s) but {closed} end(s)"]


def _sequence_problems(statements: list[str]) -> list[str]:
    problems: list[str] = []
    participants = {
        match.group(1)
        for line in statements
        if (match := re.match(r"^(?:participant|actor)\s+(\w+)", line))
    }
    depth = 0
    for line in statements:
        if _SEQUENCE_BLOCKS.match(line):
            depth += 1
        elif line == "end":
            depth -= 1
            if depth < 0:
                problems.append("'end' without an open block")
        elif message := _SEQUENCE_MESSAGE.match(line):
            undeclared = {message.group(1), message.group(2)} - participants
            if participants and undeclared:
                problems.append(f"undeclared participant(s) {sorted(undeclared)}: {line}")
        elif not _SEQUENCE_KEYWORDS.match(line):
            problems.append(f"not a sequence statement: {line}")
    if depth > 0:
        problems.append(f"{depth} block(s) without 'end'")
    return problems


def _state_problems(statements: list[str]) -> list[str]:
    return [
        f"not a state transition: {line}"
        for line in statements
        if not _STATE_TRANSITION.match(line) and not line.startswith(("note ", "state "))
    ]


def mermaid_problems(body: str) -> list[str]:
    """Structural problems in a Mermaid diagram; an empty list means well formed."""
    lines = [
        line.strip()
        for line in body.splitlines()
        if line.strip() and not line.strip().startswith("%%")
    ]
    if not lines:
        return ["empty diagram"]
    header, statements = lines[0], lines[1:]
    if not _DIAGRAM.match(header):
        return [f"unknown diagram type: {header}"]
    problems = [problem for line in statements for problem in _bracket_problems(line)]
    problems += [f"dangling edge: {line}" for line in statements if _DANGLING_EDGE.search(line)]
    if header == "sequenceDiagram":
        return problems + _sequence_problems(statements)
    if header == "stateDiagram-v2":
        return problems + _state_problems(statements)
    return problems + _flowchart_problems(statements)


def state_transitions(body: str) -> set[tuple[str, str]]:
    return {
        (match.group(1), match.group(2))
        for line in body.splitlines()
        if (match := _STATE_TRANSITION.match(line.strip()))
    }


# ---------------------------------------------------------------------------
# Files and Markdown structure
# ---------------------------------------------------------------------------


class TestDocumentsExist:
    @pytest.mark.parametrize("path", DOCUMENTS, ids=lambda p: p.name)
    def test_present_and_substantial(self, path: Path) -> None:
        text = _read(path)
        assert len(text.split()) > 400, f"{path.name} is a stub"
        assert text.startswith("# "), "starts with a title"
        assert text.endswith("\n")

    @pytest.mark.parametrize("path", DOCUMENTS, ids=lambda p: p.name)
    def test_fences_are_closed_and_labelled(self, path: Path) -> None:
        for block in code_blocks(_read(path), path):
            assert block.language, f"{block.where}: give every code block a language"
            assert block.body.strip(), f"{block.where}: empty code block"

    @pytest.mark.parametrize("path", DOCUMENTS, ids=lambda p: p.name)
    def test_relative_links_and_anchors_resolve(self, path: Path) -> None:
        text = _read(path)
        for target in re.findall(r"\]\((?!https?://|mailto:)([^)\s]+)\)", text):
            file_part, _, anchor = target.partition("#")
            linked = path if not file_part else (path.parent / file_part).resolve()
            assert linked.exists(), f"{path.name}: broken link {target}"
            if anchor:
                headings = re.findall(r"^#{1,6} (.+)$", _read(linked), re.M)
                assert anchor in {heading_anchor(h) for h in headings}, (
                    f"{path.name}: no heading for #{anchor} in {linked.name}"
                )

    def test_readme_covers_the_required_topics(self) -> None:
        text = _read(README)
        for heading in (
            "## Core tenets",
            "## Quickstart",
            "### Local development (uv)",
            "### Docker Compose",
            "### Configuration",
            "## Using the API",
            "## Benchmarks and evaluation",
        ):
            assert f"\n{heading}\n" in text, heading
        for tenet in ("Hermetic", "Streaming memory bounds", "Exact cent math", "RFC 9457"):
            assert tenet in text


class TestMarkdownHelpers:
    def test_unclosed_fence_is_reported(self) -> None:
        with pytest.raises(AssertionError, match="unclosed"):
            code_blocks("text\n```mermaid\nflowchart LR\n")

    def test_table_rows_skip_header_and_rule(self) -> None:
        text = "\n## T\n\n| a | b |\n|---|---:|\n| `x` | 1 |\n\n## Next\n| z | z |\n"
        assert table_rows(text, "## T") == [["`x`", "1"]]

    def test_heading_anchor(self) -> None:
        assert heading_anchor("Concurrency, deadlines and shutdown") == (
            "concurrency-deadlines-and-shutdown"
        )
        assert heading_anchor("Local development (uv)") == "local-development-uv"


# ---------------------------------------------------------------------------
# Mermaid
# ---------------------------------------------------------------------------


class TestMermaid:
    def test_diagrams_present(self) -> None:
        headers = [block.body.strip().splitlines()[0] for block in all_blocks("mermaid")]
        assert any(h.startswith("flowchart") for h in headers)
        assert "sequenceDiagram" in headers
        assert "stateDiagram-v2" in headers

    @pytest.mark.parametrize("block", all_blocks("mermaid"), ids=lambda b: b.where)
    def test_well_formed(self, block: CodeBlock) -> None:
        assert mermaid_problems(block.body) == []

    @pytest.mark.parametrize(
        ("body", "problem"),
        [
            ("", "empty"),
            ("pie title x", "unknown diagram"),
            ('flowchart LR\n  a["open --> b', "quotes"),
            ("flowchart LR\n  a[label --> b", "unclosed"),
            ("flowchart LR\n  a(label] --> b", "unbalanced"),
            ("flowchart LR\n  a --> ", "dangling"),
            ("flowchart LR\n  subgraph s\n  a --> b", "subgraph"),
            ("sequenceDiagram\n  participant A\n  A->>B: hi", "undeclared"),
            ("sequenceDiagram\n  participant A\n  loop x\n  A->>A: hi", "without 'end'"),
            ("sequenceDiagram\n  participant A\n  end", "without an open"),
            ("sequenceDiagram\n  participant A\n  A says hi", "not a sequence"),
            ("stateDiagram-v2\n  a -> b", "not a state transition"),
        ],
    )
    def test_checker_catches_broken_diagrams(self, body: str, problem: str) -> None:
        problems = mermaid_problems(body)
        assert problems, body
        assert any(problem in p for p in problems), problems

    def test_state_diagram_matches_the_job_state_machine(self) -> None:
        (block,) = [
            b for b in all_blocks("mermaid") if b.body.strip().startswith("stateDiagram-v2")
        ]
        transitions = state_transitions(block.body)
        start = {target for source, target in transitions if source == "[*]"}
        final = {source for source, target in transitions if target == "[*]"}
        documented = {(s, t) for s, t in transitions if "[*]" not in (s, t)}
        expected = {(s.value, t.value) for s, targets in TRANSITIONS.items() for t in targets}
        assert documented == expected
        assert start == {JobStatus.QUEUED.value}
        assert final == {status.value for status in JobStatus if status.is_terminal}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _env_name(field: str) -> str:
    return f"{ENV_PREFIX}{field}".upper()


def _env_default(field: str) -> str:
    default = VisionSettings.model_fields[field].default
    if default is None:
        return "*(required)*"
    if isinstance(default, bool):
        return str(default).lower()
    if isinstance(default, Enum):
        return str(default.value)
    if isinstance(default, float) and default.is_integer():
        return str(int(default))
    return str(default).rstrip("/")


class TestConfigurationDocs:
    def test_documented_variables_match_settings_exactly(self) -> None:
        expected = {_env_name(field) for field in VisionSettings.model_fields}
        for path in DOCUMENTS:
            mentioned = set(re.findall(r"\bMVP_[A-Z][A-Z0-9_]*\b", _read(path)))
            unknown = mentioned - expected
            assert not unknown, f"{path.name} documents settings that do not exist: {unknown}"
        table = {_code(row[0]) for row in table_rows(_read(README), "### Configuration")}
        assert table == expected, f"missing {expected - table}, extra {table - expected}"

    def test_table_defaults_match_the_code(self) -> None:
        names = {_env_name(field): field for field in VisionSettings.model_fields}
        for name_cell, default_cell, *_ in table_rows(_read(README), "### Configuration"):
            field = names[_code(name_cell)]
            assert _code(default_cell) == _env_default(field), _code(name_cell)


# ---------------------------------------------------------------------------
# API examples
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Curl:
    where: str
    method: str
    url: str
    forms: dict[str, str]
    headers: dict[str, str]


def _shell_commands(block: CodeBlock) -> Iterator[list[str]]:
    joined = re.sub(r"\\\n\s*", " ", block.body)
    for line in joined.splitlines():
        tokens = shlex.split(line, comments=True)
        if "|" in tokens:
            tokens = tokens[: tokens.index("|")]
        if tokens:
            yield tokens


def curl_commands() -> list[Curl]:
    found: list[Curl] = []
    for block in all_blocks("bash"):
        for tokens in _shell_commands(block):
            if "curl" not in tokens:
                continue
            args = tokens[tokens.index("curl") + 1 :]
            method, url = "GET", ""
            forms: dict[str, str] = {}
            headers: dict[str, str] = {}
            for flag, value in zip(args, [*args[1:], ""], strict=True):
                if flag in {"-X", "--request"}:
                    method = value
                elif flag in {"-F", "--form"}:
                    name, _, content = value.partition("=")
                    forms[name] = content
                    method = "POST" if method == "GET" else method
                elif flag in {"-H", "--header"}:
                    name, _, content = value.partition(":")
                    headers[name.strip()] = content.strip()
            url = next(arg for arg in args if arg.startswith("http"))
            found.append(Curl(block.where, method, url, forms, headers))
    return found


@pytest.fixture(scope="module")
def routes() -> set[tuple[str, str]]:
    """``(METHOD, path template)`` of every API route, from the app's OpenAPI document.

    FastAPI includes routers lazily, so ``app.routes`` does not list them; the schema is the
    public, authoritative registry.
    """
    app = create_app(
        VisionSettings(anthropic_api_key=SecretStr("sk-ant-docs-0000"), log_json=False)
    )
    return {
        (method.upper(), path)
        for path, operations in app.openapi()["paths"].items()
        for method in operations
    }


def _template_for(path: str, routes: set[tuple[str, str]], method: str) -> str | None:
    for route_method, template in routes:
        pattern = re.sub(r"\\\{\w+\\\}", r"[^/]+", re.escape(template))
        if route_method == method and re.fullmatch(pattern, path):
            return template
    return None


class TestApiExamples:
    def test_every_curl_targets_a_registered_route(self, routes: set[tuple[str, str]]) -> None:
        commands = curl_commands()
        assert len(commands) >= 6
        for curl in commands:
            parts = urlsplit(curl.url)
            assert f"{parts.scheme}://{parts.netloc}" == LOCAL_BASE, curl.where
            template = _template_for(parts.path, routes, curl.method)
            assert template is not None, f"{curl.where}: {curl.method} {parts.path} is not a route"

    def test_every_route_has_a_curl_example(self, routes: set[tuple[str, str]]) -> None:
        documented = {
            (curl.method, _template_for(urlsplit(curl.url).path, routes, curl.method))
            for curl in curl_commands()
        }
        assert routes <= documented, f"undocumented: {routes - documented}"

    def test_upload_example_is_a_valid_request(self) -> None:
        (upload,) = [c for c in curl_commands() if c.method == "POST" and "options" in c.forms]
        assert set(upload.forms) == {"file", "options"}
        assert upload.forms["file"].startswith("@")
        options = IngestionOptions.model_validate_json(upload.forms["options"])
        assert options.rasterization.pages is not None
        assert IDEMPOTENCY_HEADER in upload.headers
        for curl in curl_commands():
            assert set(curl.forms) <= {"file", "options"}, curl.where

    @pytest.mark.parametrize("block", all_blocks("json"), ids=lambda b: b.where)
    def test_json_examples_parse(self, block: CodeBlock) -> None:
        json.loads(block.body)

    def test_problem_and_readiness_examples_match_their_schemas(self) -> None:
        bodies = [json.loads(block.body) for block in all_blocks("json")]
        problems = [body for body in bodies if "type" in body and "status" in body]
        assert problems
        for body in problems:
            problem = ProblemDetails.model_validate_json(json.dumps(body))
            assert problem.type.startswith(PROBLEM_TYPE_PREFIX)
        readiness = [body for body in bodies if "worker" in body]
        assert readiness
        for body in readiness:
            Readiness.model_validate_json(json.dumps(body))


# ---------------------------------------------------------------------------
# Architecture facts
# ---------------------------------------------------------------------------


def _pipeline_errors() -> dict[str, int]:
    # api.errors (validation, not-ready) is loaded by the create_app import above.
    found: dict[str, int] = {}
    pending: list[type[PipelineError]] = [PipelineError]
    while pending:
        cls = pending.pop()
        found[cls.slug] = cls.status
        pending.extend(cls.__subclasses__())
    return found


def _exposed_metric_names() -> set[str]:
    suffixes = {
        "counter": ("_total",),
        "histogram": ("", "_bucket", "_count", "_sum"),
        "gauge": ("",),
    }
    return {
        family.name + suffix
        for family in PipelineMetrics().registry.collect()
        for suffix in suffixes[family.type]
    }


class TestArchitectureFacts:
    def test_error_catalogue_matches_the_code(self) -> None:
        rows = table_rows(_read(ARCHITECTURE), "## Error model")
        documented = {_code(slug): int(status) for slug, status, *_ in rows}
        assert documented == _pipeline_errors()

    def test_metric_names_exist_and_are_all_documented(self) -> None:
        exposed = _exposed_metric_names()
        for path in DOCUMENTS:
            mentioned = set(re.findall(r"\bmvp_[a-z0-9_]*[a-z0-9]\b", _read(path)))
            assert mentioned <= exposed, f"{path.name}: unknown metrics {mentioned - exposed}"
        table = {_code(row[0]) for row in table_rows(_read(ARCHITECTURE), "## Observability")}
        families = {
            family.name + ("_total" if family.type == "counter" else "")
            for family in PipelineMetrics().registry.collect()
        }
        assert table == families

    def test_reconciliation_rules_and_policy_defaults(self) -> None:
        text = _read(ARCHITECTURE)
        rules = table_rows(text, "## Mathematical reconciliation")
        documented = {
            _code(row[0])
            for row in rules
            if row and _code(row[0]) in {r.value for r in ReconciliationRule}
        }
        assert documented == {rule.value for rule in ReconciliationRule}
        policy = ReconciliationPolicy()
        assert f"| `tolerance_cents` | `{policy.tolerance_cents}` |" in text
        assert f"| `rounding` | `{policy.rounding.value}` |" in text

    def test_rounding_example_is_arithmetically_true(self) -> None:
        text = " ".join(_read(ARCHITECTURE).split())
        base, rate = 250, Decimal(9)
        up = ReconciliationPolicy(rounding=RoundingMode.HALF_UP).expected_tax(base, rate)
        even = ReconciliationPolicy(rounding=RoundingMode.HALF_EVEN).expected_tax(base, rate)
        assert (up, even) == (23, 22)
        assert f"{rate}% of {base} cents is 22.5 cents, which is {up} under `half_up`" in text
        assert f"and {even} under `half_even`" in text

    def test_limits_quoted_in_the_guide(self) -> None:
        text = " ".join(_read(ARCHITECTURE).split())  # Markdown wraps lines anywhere
        for fact in (
            f"one chunk of {DEFAULT_CHUNK_SIZE} pages",
            f"{MAX_PIXELS_PER_SIDE:,}",
            f"{MIN_DPI} to {MAX_DPI}, default {DEFAULT_DPI}",
            f"default {DEFAULT_MAX_PAGES}, at most {MAX_PAGE_LIMIT}",
            f"at most {MAX_IMAGES_PER_REQUEST} images",
            f"{POPPLER_TIMEOUT_SECONDS} s timeout",
            f"{DEFAULT_MAX_TOTAL_IMAGE_TOKENS:,} image tokens",
            f"`anthropic-version: {API_VERSION}`",
            f"`anthropic-beta: {FALLBACK_BETA}`",
        ):
            assert fact in text, fact
        statuses = sorted(RETRYABLE_STATUSES)
        assert f"{', '.join(map(str, statuses[:-1]))} and {statuses[-1]}" in text
        retry = RetryPolicy()
        assert f"min({retry.max_delay:g} s, uniform({retry.base_delay:g} s, 3 x previous))" in text
        assert f"never above {retry.max_retry_after:g} s" in text

    def test_model_image_limits_table(self) -> None:
        rows = {
            _code(row[0]): row
            for row in table_rows(_read(ARCHITECTURE), "## Vision payload builder")
        }
        for model, limits in MODEL_IMAGE_LIMITS.items():
            assert rows[model.value][1] == f"{limits.max_long_edge} px"
            assert rows[model.value][2] == f"{limits.max_pixels / 1_000_000:g} MP"


# ---------------------------------------------------------------------------
# Runbook
# ---------------------------------------------------------------------------


def _alert_rules() -> list[dict[str, Any]]:
    (block,) = [b for b in all_blocks("yaml") if b.path == RUNBOOK and "groups:" in b.body]
    document = yaml.safe_load(block.body)
    return [rule for group in document["groups"] for rule in group["rules"]]


class TestRunbook:
    def test_alert_rules_are_valid_prometheus_rule_shapes(self) -> None:
        rules = _alert_rules()
        assert len(rules) >= 5
        names = [rule["alert"] for rule in rules]
        assert len(set(names)) == len(names)
        for rule in rules:
            assert re.fullmatch(r"Mvp[A-Z][A-Za-z]+", rule["alert"])
            assert re.fullmatch(r"\d+[smh]", rule["for"])
            assert rule["labels"]["severity"] in {"page", "ticket"}
            assert rule["annotations"]["summary"]
            assert rule["annotations"]["runbook"]
            expr = rule["expr"]
            assert _bracket_problems(" ".join(expr.split())) == [], rule["alert"]
            assert re.search(r"\bmvp_\w+", expr), "every rule watches a service metric"

    def test_alert_label_values_exist(self) -> None:
        statuses = {status.value for status in JobStatus}
        for rule in _alert_rules():
            for value in re.findall(r'\bstatus="(\w+)"', rule["expr"]):
                assert value in statuses
            for route in re.findall(r'\broute="([^"]+)"', rule["expr"]):
                assert route in {"/api/v1/documents", "/api/v1/jobs/{job_id}", "unmatched"}

    def test_probe_guidance_matches_the_health_routes(self, routes: set[tuple[str, str]]) -> None:
        rows = table_rows(_read(RUNBOOK), "## Health probing")
        documented = {_code(row[0]).removeprefix("GET ") for row in rows}
        assert documented == {"/healthz", "/readyz"}
        assert documented <= {path for _, path in routes}

    def test_troubleshooting_names_real_problem_types(self) -> None:
        slugs = set(_pipeline_errors())
        rows = table_rows(_read(RUNBOOK), "## Troubleshooting")
        mentioned = {slug for row in rows for slug in re.findall(r"`([a-z]+(?:-[a-z]+)+)`", row[0])}
        assert len(mentioned) >= 10
        assert mentioned <= slugs, mentioned - slugs


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


class TestBenchmarkDocs:
    def test_cli_flags_exist(self) -> None:
        options = {option for action in build_parser()._actions for option in action.option_strings}
        commands = [
            tokens
            for block in all_blocks("bash")
            for tokens in _shell_commands(block)
            if "multimodal_vision_pipeline.evals" in tokens
        ]
        text_commands = re.findall(r"`[^`]*multimodal_vision_pipeline\.evals[^`]*`", _read(RUNBOOK))
        commands += [shlex.split(command.strip("`")) for command in text_commands]
        assert commands
        for tokens in commands:
            for token in tokens:
                if token.startswith("--"):
                    assert token in options, token
            if "--profiles" in tokens:
                selected = tokens[tokens.index("--profiles") + 1].split(",")
                assert set(selected) <= set(PROFILES)

    def test_profile_table_lists_every_profile(self) -> None:
        rows = table_rows(_read(README), "## Benchmarks and evaluation")
        assert [_code(row[0]) for row in rows] == list(PROFILES)
