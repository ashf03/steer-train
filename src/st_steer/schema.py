"""Artifact schema v1 — writers, validators, and read-side normalizers.

Stable on-disk contract (``docs/tool-mvp.md``). Breaking changes bump
``SCHEMA_VERSION``. Readers accept legacy rows (no ``schema`` / ``loss``
alias) unless ``strict=True``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

SCHEMA_VERSION = 1

# Toy setting id used by the synthetic trainer.
TOY_SETTING_ID = "toy-synthetic"


class SchemaError(ValueError):
    """Raised when an artifact fails schema validation."""

    def __init__(self, message: str, *, path: Path | None = None, line: int | None = None):
        self.path = path
        self.line = line
        loc = ""
        if path is not None:
            loc = str(path)
            if line is not None:
                loc = f"{loc}:{line}"
            loc = f"{loc}: "
        super().__init__(f"{loc}{message}")


@dataclass
class ValidationIssue:
    message: str
    path: Path | None = None
    line: int | None = None
    level: str = "error"  # error | warn

    def format(self) -> str:
        loc = ""
        if self.path is not None:
            loc = str(self.path)
            if self.line is not None:
                loc = f"{loc}:{self.line}"
            loc = f"{loc}: "
        return f"{self.level:5}  {loc}{self.message}"


@dataclass
class ValidationResult:
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    def extend(self, other: ValidationResult) -> None:
        self.issues.extend(other.issues)
        if not other.ok:
            self.ok = False

    def raise_if_errors(self) -> None:
        errors = [i for i in self.issues if i.level == "error"]
        if errors:
            raise SchemaError(errors[0].message, path=errors[0].path, line=errors[0].line)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def train_loss(row: Mapping[str, Any]) -> float:
    """Prefer ``train_loss``; accept toy alias ``loss``."""
    if "train_loss" in row and row["train_loss"] is not None:
        return float(row["train_loss"])
    if "loss" in row and row["loss"] is not None:
        return float(row["loss"])
    raise KeyError("metric row missing train_loss/loss")


def normalize_metric(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a schema-1 shaped metric dict for in-memory use.

    Does not mutate the input. Legacy rows without ``schema`` are tagged
    ``schema: 0`` in the normalized view (file on disk unchanged).
    """
    out = dict(row)
    schema = out.get("schema")
    if schema is None:
        out["schema"] = 0  # legacy
    if "train_loss" not in out and "loss" in out:
        out["train_loss"] = out["loss"]
    elif "loss" not in out and "train_loss" in out:
        out["loss"] = out["train_loss"]  # keep toy readers happy
    return out


def append_jsonl(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(obj), sort_keys=True) + "\n")
        handle.flush()


def write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(obj), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as err:
                raise SchemaError(f"invalid JSON ({err})", path=path, line=line_no) from err
            if not isinstance(obj, dict):
                raise SchemaError(
                    f"expected object, got {type(obj).__name__}",
                    path=path,
                    line=line_no,
                )
            yield line_no, obj


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def metric_record(
    *,
    step: int,
    lr: float,
    wall_ms: float,
    train_loss: float | None = None,
    loss: float | None = None,
    val_loss: float | None = None,
    epoch: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a schema-1 metrics row. Toy may pass ``loss=``; scale uses ``train_loss=``."""
    if train_loss is None and loss is None:
        raise ValueError("metric_record requires train_loss or loss")
    tl = float(train_loss if train_loss is not None else loss)
    row: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "step": int(step),
        "lr": float(lr),
        "wall_ms": float(wall_ms),
        "train_loss": tl,
        "loss": tl,  # toy alias always written for older readers
    }
    if val_loss is not None or epoch is not None:
        row["val_loss"] = val_loss
        row["epoch"] = epoch
    if extra:
        for k, v in extra.items():
            if k not in row:
                row[k] = v
    return row


def write_metric(path: Path, row: Mapping[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    """Append one metrics row (builds via ``metric_record`` if kwargs given)."""
    record = dict(row) if row is not None else metric_record(**kwargs)
    if "schema" not in record:
        record = {**metric_record(
            step=int(record["step"]),
            lr=float(record["lr"]),
            wall_ms=float(record["wall_ms"]),
            train_loss=float(record.get("train_loss", record.get("loss"))),
            val_loss=record.get("val_loss"),
            epoch=record.get("epoch"),
        ), **{k: v for k, v in record.items() if k not in (
            "schema", "step", "lr", "wall_ms", "train_loss", "loss", "val_loss", "epoch"
        )}}
    append_jsonl(path, record)
    return record


def config_snapshot_record(
    *,
    seed: int,
    steps: int,
    lr: float,
    started_at: str,
    setting_id: str = TOY_SETTING_ID,
    condition: str = "baseline",
    sleep_seconds: float | None = None,
    sleep_override: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "setting_id": setting_id,
        "condition": condition,
        "seed": int(seed),
        "steps": int(steps),
        "lr": float(lr),
        "started_at": started_at,
    }
    if sleep_seconds is not None:
        snap["sleep_seconds"] = float(sleep_seconds)
    if sleep_override is not None:
        snap["sleep_override"] = sleep_override
    if extra:
        for k, v in extra.items():
            if k not in snap:
                snap[k] = v
    return snap


def write_config_snapshot(path: Path, snapshot: Mapping[str, Any]) -> Path:
    record = dict(snapshot)
    if "schema" not in record:
        record["schema"] = SCHEMA_VERSION
    write_json(path, record)
    return path


def intervention_record(
    *,
    timestamp: str,
    rule_id: str,
    decision_step: int,
    evidence: Mapping[str, Any] | list[Any],
    action: Mapping[str, Any],
    effect_step: int | None = None,
    effect_delay_steps: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(evidence, list):
        evidence_obj: dict[str, Any] = {"steps": list(evidence)}
    else:
        evidence_obj = dict(evidence)
    record: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "timestamp": timestamp,
        "rule_id": rule_id,
        "decision_step": int(decision_step),
        "evidence": evidence_obj,
        "action": dict(action),
        "effect_step": effect_step if effect_step is None else int(effect_step),
        "effect_delay_steps": (
            effect_delay_steps if effect_delay_steps is None else int(effect_delay_steps)
        ),
    }
    if extra:
        for k, v in extra.items():
            if k not in record:
                record[k] = v
    return record


def write_intervention(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(record)
    if "schema" not in out:
        out["schema"] = SCHEMA_VERSION
    append_jsonl(path, out)
    return out


def patch_intervention_effect(
    path: Path,
    *,
    decision_step: int,
    effect_step: int,
    effect_delay_steps: int,
) -> int:
    """Update matching intervention rows in-place (rewrite file). Returns count patched."""
    if not path.exists():
        return 0
    rows: list[dict[str, Any]] = []
    patched = 0
    for _, obj in iter_jsonl(path):
        if (
            obj.get("decision_step") == decision_step
            and obj.get("event") != "effect_observed"
            and "action" in obj
        ):
            obj = dict(obj)
            obj["effect_step"] = int(effect_step)
            obj["effect_delay_steps"] = int(effect_delay_steps)
            if "schema" not in obj:
                obj["schema"] = SCHEMA_VERSION
            patched += 1
        rows.append(obj)
    path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
        encoding="utf-8",
    )
    return patched


def compare_record(
    *,
    baseline_path: str | Path,
    steered_path: str | Path,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "baseline_path": str(baseline_path),
        "steered_path": str(steered_path),
        **dict(result),
    }


def write_compare(path: Path, record: Mapping[str, Any]) -> Path:
    out = dict(record)
    if "schema" not in out:
        out["schema"] = SCHEMA_VERSION
    write_json(path, out)
    return path


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _check_schema_field(
    obj: Mapping[str, Any],
    *,
    path: Path | None,
    line: int | None,
    strict: bool,
    issues: list[ValidationIssue],
) -> bool:
    """Return True if row should be treated as schema-1; False if legacy accepted."""
    if "schema" not in obj:
        if strict:
            issues.append(ValidationIssue("missing required key 'schema'", path=path, line=line))
            return False
        issues.append(
            ValidationIssue("legacy row (no schema); accepted in compat mode", path=path, line=line, level="warn")
        )
        return False
    schema = obj["schema"]
    if not _is_int(schema):
        issues.append(ValidationIssue(f"schema must be int, got {type(schema).__name__}", path=path, line=line))
        return False
    if schema != SCHEMA_VERSION:
        if strict or schema not in (0, SCHEMA_VERSION):
            issues.append(
                ValidationIssue(
                    f"unsupported schema {schema} (expected {SCHEMA_VERSION})",
                    path=path,
                    line=line,
                )
            )
            return False
        if schema == 0:
            issues.append(
                ValidationIssue("schema 0 (legacy tag); accepted in compat mode", path=path, line=line, level="warn")
            )
            return False
    return True


def validate_metric_row(
    obj: Mapping[str, Any],
    *,
    path: Path | None = None,
    line: int | None = None,
    strict: bool = False,
) -> ValidationResult:
    issues: list[ValidationIssue] = []
    is_v1 = _check_schema_field(obj, path=path, line=line, strict=strict, issues=issues)

    for key, checker, label in (
        ("step", _is_int, "int"),
        ("lr", _is_number, "number"),
        ("wall_ms", _is_number, "number"),
    ):
        if key not in obj:
            issues.append(ValidationIssue(f"missing required key '{key}'", path=path, line=line))
        elif not checker(obj[key]):
            issues.append(
                ValidationIssue(
                    f"'{key}' must be {label}, got {type(obj[key]).__name__}",
                    path=path,
                    line=line,
                )
            )

    has_train = "train_loss" in obj and obj["train_loss"] is not None
    has_loss = "loss" in obj and obj["loss"] is not None
    if not has_train and not has_loss:
        issues.append(
            ValidationIssue("missing train_loss (or toy alias loss)", path=path, line=line)
        )
    else:
        for key in ("train_loss", "loss"):
            if key in obj and obj[key] is not None and not _is_number(obj[key]):
                issues.append(
                    ValidationIssue(
                        f"'{key}' must be number, got {type(obj[key]).__name__}",
                        path=path,
                        line=line,
                    )
                )

    if is_v1:
        for key in ("val_loss", "epoch"):
            if key in obj and obj[key] is not None and not _is_number(obj[key]):
                issues.append(
                    ValidationIssue(
                        f"'{key}' must be number|null, got {type(obj[key]).__name__}",
                        path=path,
                        line=line,
                    )
                )

    ok = not any(i.level == "error" for i in issues)
    return ValidationResult(ok=ok, issues=issues)


def validate_metrics_file(path: Path, *, strict: bool = False) -> ValidationResult:
    result = ValidationResult(ok=True)
    if not path.exists():
        result.ok = False
        result.issues.append(ValidationIssue("file not found", path=path))
        return result
    n = 0
    try:
        for line_no, obj in iter_jsonl(path):
            n += 1
            result.extend(validate_metric_row(obj, path=path, line=line_no, strict=strict))
    except SchemaError as err:
        result.ok = False
        result.issues.append(ValidationIssue(str(err), path=path, line=err.line))
        return result
    if n == 0:
        result.issues.append(ValidationIssue("empty metrics file", path=path, level="warn"))
    return result


def validate_config_snapshot(
    obj: Mapping[str, Any],
    *,
    path: Path | None = None,
    strict: bool = False,
) -> ValidationResult:
    issues: list[ValidationIssue] = []
    _check_schema_field(obj, path=path, line=None, strict=strict, issues=issues)

    required_v1 = ("setting_id", "condition", "seed", "steps", "lr", "started_at")
    required_legacy = ("seed", "steps", "lr", "started_at")
    keys = required_v1 if ("schema" in obj and obj.get("schema") == SCHEMA_VERSION) or strict else required_legacy

    for key in keys:
        if key not in obj:
            # setting_id/condition missing on legacy snapshots → warn unless strict
            if key in ("setting_id", "condition") and not strict:
                issues.append(
                    ValidationIssue(f"missing '{key}' (legacy snapshot)", path=path, level="warn")
                )
            else:
                issues.append(ValidationIssue(f"missing required key '{key}'", path=path))

    for key, checker, label in (
        ("seed", _is_int, "int"),
        ("steps", _is_int, "int"),
        ("lr", _is_number, "number"),
    ):
        if key in obj and not checker(obj[key]):
            issues.append(
                ValidationIssue(
                    f"'{key}' must be {label}, got {type(obj[key]).__name__}",
                    path=path,
                )
            )
    if "started_at" in obj and not isinstance(obj["started_at"], str):
        issues.append(
            ValidationIssue(
                f"'started_at' must be string, got {type(obj['started_at']).__name__}",
                path=path,
            )
        )
    for key in ("setting_id", "condition"):
        if key in obj and not isinstance(obj[key], str):
            issues.append(
                ValidationIssue(
                    f"'{key}' must be string, got {type(obj[key]).__name__}",
                    path=path,
                )
            )

    ok = not any(i.level == "error" for i in issues)
    return ValidationResult(ok=ok, issues=issues)


def validate_config_snapshot_file(path: Path, *, strict: bool = False) -> ValidationResult:
    if not path.exists():
        return ValidationResult(
            ok=False,
            issues=[ValidationIssue("file not found", path=path)],
        )
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        return ValidationResult(
            ok=False,
            issues=[ValidationIssue(f"unreadable JSON ({err})", path=path)],
        )
    if not isinstance(obj, dict):
        return ValidationResult(
            ok=False,
            issues=[ValidationIssue(f"expected object, got {type(obj).__name__}", path=path)],
        )
    return validate_config_snapshot(obj, path=path, strict=strict)


def validate_intervention_row(
    obj: Mapping[str, Any],
    *,
    path: Path | None = None,
    line: int | None = None,
    strict: bool = False,
) -> ValidationResult:
    issues: list[ValidationIssue] = []

    # Allow legacy effect_observed follow-up rows in compat mode.
    if obj.get("event") == "effect_observed":
        if strict:
            issues.append(
                ValidationIssue(
                    "legacy event='effect_observed' row; use patched intervention instead",
                    path=path,
                    line=line,
                )
            )
        else:
            issues.append(
                ValidationIssue(
                    "legacy effect_observed row; accepted in compat mode",
                    path=path,
                    line=line,
                    level="warn",
                )
            )
        ok = not any(i.level == "error" for i in issues)
        return ValidationResult(ok=ok, issues=issues)

    is_v1 = _check_schema_field(obj, path=path, line=line, strict=strict, issues=issues)

    # Legacy shape: evidence_steps + lr_from/lr_to instead of evidence/action
    legacy = "evidence_steps" in obj and "action" not in obj
    if legacy and not strict:
        issues.append(
            ValidationIssue(
                "legacy intervention shape; accepted in compat mode",
                path=path,
                line=line,
                level="warn",
            )
        )
        for key in ("timestamp", "decision_step"):
            if key not in obj:
                issues.append(ValidationIssue(f"missing '{key}'", path=path, line=line))
        ok = not any(i.level == "error" for i in issues)
        return ValidationResult(ok=ok, issues=issues)

    for key in ("timestamp", "rule_id", "decision_step", "evidence", "action"):
        if key not in obj:
            issues.append(ValidationIssue(f"missing required key '{key}'", path=path, line=line))

    if "timestamp" in obj and not isinstance(obj["timestamp"], str):
        issues.append(ValidationIssue("'timestamp' must be string", path=path, line=line))
    if "rule_id" in obj and not isinstance(obj["rule_id"], str):
        issues.append(ValidationIssue("'rule_id' must be string", path=path, line=line))
    if "decision_step" in obj and not _is_int(obj["decision_step"]):
        issues.append(ValidationIssue("'decision_step' must be int", path=path, line=line))
    if "evidence" in obj and not isinstance(obj["evidence"], dict):
        issues.append(ValidationIssue("'evidence' must be object", path=path, line=line))
    if "action" in obj and not isinstance(obj["action"], dict):
        issues.append(ValidationIssue("'action' must be object", path=path, line=line))
    elif "action" in obj and isinstance(obj["action"], dict) and "type" not in obj["action"]:
        issues.append(ValidationIssue("'action.type' required", path=path, line=line))

    for key in ("effect_step", "effect_delay_steps"):
        if key in obj and obj[key] is not None and not _is_int(obj[key]):
            issues.append(ValidationIssue(f"'{key}' must be int|null", path=path, line=line))

    _ = is_v1  # reserved for future tighter checks
    ok = not any(i.level == "error" for i in issues)
    return ValidationResult(ok=ok, issues=issues)


def validate_interventions_file(path: Path, *, strict: bool = False) -> ValidationResult:
    result = ValidationResult(ok=True)
    if not path.exists():
        # Interventions optional for baseline runs
        result.issues.append(
            ValidationIssue("interventions.jsonl absent (ok for baseline)", path=path, level="warn")
        )
        return result
    try:
        for line_no, obj in iter_jsonl(path):
            result.extend(validate_intervention_row(obj, path=path, line=line_no, strict=strict))
    except SchemaError as err:
        result.ok = False
        result.issues.append(ValidationIssue(str(err), path=path, line=err.line))
    return result


def validate_compare_obj(
    obj: Mapping[str, Any],
    *,
    path: Path | None = None,
    strict: bool = False,
) -> ValidationResult:
    issues: list[ValidationIssue] = []
    _check_schema_field(obj, path=path, line=None, strict=strict, issues=issues)
    for key in ("baseline_path", "steered_path", "verdict"):
        if key not in obj:
            issues.append(ValidationIssue(f"missing required key '{key}'", path=path))
    ok = not any(i.level == "error" for i in issues)
    return ValidationResult(ok=ok, issues=issues)


def validate_run_dir(run_dir: Path, *, strict: bool = False, require_interventions: bool = False) -> ValidationResult:
    """Validate metrics + snapshot (+ optional interventions) under a run directory."""
    result = ValidationResult(ok=True)
    metrics = run_dir / "metrics.jsonl"
    snap = run_dir / "config.snapshot.json"
    inter = run_dir / "interventions.jsonl"

    result.extend(validate_metrics_file(metrics, strict=strict))
    if snap.exists():
        result.extend(validate_config_snapshot_file(snap, strict=strict))
    elif strict:
        result.ok = False
        result.issues.append(ValidationIssue("config.snapshot.json required", path=snap))
    else:
        result.issues.append(
            ValidationIssue("config.snapshot.json absent (legacy run)", path=snap, level="warn")
        )
    if require_interventions or inter.exists():
        ir = validate_interventions_file(inter, strict=strict)
        if require_interventions and not inter.exists():
            result.ok = False
            result.issues.append(ValidationIssue("interventions.jsonl required but missing", path=inter))
        else:
            result.extend(ir)
    return result


def schema_ok(run_dir: Path, *, strict: bool = False, require_interventions: bool = False) -> bool:
    return validate_run_dir(
        run_dir, strict=strict, require_interventions=require_interventions
    ).ok


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(
        prog="st validate",
        description="Validate st artifact schema (metrics / snapshot / interventions / run dir)",
    )
    p.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="metrics.jsonl, config.snapshot.json, interventions.jsonl, compare.json, or a run directory",
    )
    p.add_argument("--strict", action="store_true", help="reject legacy (pre-schema) artifacts")
    p.add_argument("--json", action="store_true", help="machine-readable issues")
    args = p.parse_args(argv)

    all_result = ValidationResult(ok=True)
    for path in args.paths:
        path = path.expanduser()
        if not path.is_absolute():
            from st_steer.paths import resolve

            path = resolve(path)

        if path.is_dir():
            r = validate_run_dir(path, strict=args.strict)
        elif path.name == "metrics.jsonl":
            r = validate_metrics_file(path, strict=args.strict)
        elif path.name == "config.snapshot.json":
            r = validate_config_snapshot_file(path, strict=args.strict)
        elif path.name == "interventions.jsonl":
            r = validate_interventions_file(path, strict=args.strict)
        elif path.name == "compare.json" or path.suffix == ".json":
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as err:
                r = ValidationResult(ok=False, issues=[ValidationIssue(f"unreadable ({err})", path=path)])
            else:
                if isinstance(obj, dict) and "verdict" in obj:
                    r = validate_compare_obj(obj, path=path, strict=args.strict)
                else:
                    r = validate_config_snapshot(obj if isinstance(obj, dict) else {}, path=path, strict=args.strict)
                    if not isinstance(obj, dict):
                        r = ValidationResult(
                            ok=False,
                            issues=[ValidationIssue("expected JSON object", path=path)],
                        )
        else:
            # try metrics first
            r = validate_metrics_file(path, strict=args.strict)

        all_result.extend(r)
        if not args.json:
            status = "ok" if r.ok else "FAIL"
            print(f"{status:4}  {path}")
            # Collapse repeated warn messages (legacy rows × N)
            from collections import Counter

            warn_keys: Counter[str] = Counter()
            for issue in r.issues:
                if issue.level == "error":
                    print(f"      {issue.format()}")
                else:
                    # strip line number for grouping
                    msg = issue.message
                    warn_keys[msg] += 1
            for msg, count in warn_keys.items():
                suffix = f"  (×{count})" if count > 1 else ""
                print(f"      warn   {msg}{suffix}")

    if args.json:
        payload = {
            "schema": SCHEMA_VERSION,
            "ok": all_result.ok,
            "issues": [
                {
                    "level": i.level,
                    "message": i.message,
                    "path": str(i.path) if i.path else None,
                    "line": i.line,
                }
                for i in all_result.issues
            ],
        }
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print("validate  " + ("ok" if all_result.ok else "FAILED"))
    return 0 if all_result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
