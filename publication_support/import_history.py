#!/usr/bin/env python3
"""Import verified research bundles, preserving the publication first parent.

Run from a checkout containing .github/gbs-publication.json.  --dry-run creates
and validates a local merge object only; it never contacts or updates a remote.
It does not change HEAD, branch refs, the working tree, or the normal Git index.

Request format (delta may be omitted):
{
  "format": "deziroGBS-publication-request-v1",
  "repository": "qiyuanhuakai/deziroGBS",
  "base": {"manifest_path": "research_archive/iteration9/MANIFEST.json",
           "manifest_bytes": 123, "manifest_sha256": "<64 lowercase hex>"},
  "delta": {"manifest_path": "research_archive/iteration10/MANIFEST.json",
            "manifest_bytes": 123, "manifest_sha256": "<64 lowercase hex>"}
}

The base manifest uses deziroGBS-complete-git-bundle-parts-v1.  The optional
delta manifest uses deziroGBS-incremental-git-bundle-parts-v1 and requires:
prerequisite_commit, source_commit, source_tree, source_branch, bundle_bytes,
bundle_sha256, and parts [{name, bytes, sha256}, ...].  Parts have consecutive
names history.bundle.part000, history.bundle.part001, etc.  If all_tracked_files
is supplied, its complete path/mode/sha/size inventory is also verified.

All inputs and publication overlays come from trigger-commit Git objects,
not mutable working files.  No credential or authentication configuration is
read by this script.  Git uses the standard authentication saved by checkout.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile


REPOSITORY = "qiyuanhuakai/deziroGBS"
REMOTE_URL = "https://github.com/qiyuanhuakai/deziroGBS.git"
REQUEST_PATH = ".github/gbs-publication.json"
SCRIPT_PATH = "publication_support/import_history.py"
BASE_MANIFEST = "research_archive/iteration9/MANIFEST.json"
DELTA_MANIFEST = "research_archive/iteration10/MANIFEST.json"
BASE_COMMIT = "8675497ae2ff16f7f6fbaba4ad6eb0545c1612ff"
BASE_BRANCH = "research/iteration8"
DELTA_BRANCH = "research/iteration10"
OPTIONAL_OVERLAYS = (
    "LEGACY_NATIVE_REBUILD.md",
    "PUBLICATION.md",
    SCRIPT_PATH,
)
SHA1 = re.compile(r"[0-9a-f]{40}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ImportFailure(RuntimeError):
    """A failed integrity or publication precondition; nothing is forced."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ImportFailure(message)


def sha1(value: object, label: str) -> str:
    require(isinstance(value, str) and SHA1.fullmatch(value) is not None,
            f"Invalid SHA-1: {label}")
    return value


def sha256(value: object, label: str) -> str:
    require(isinstance(value, str) and SHA256.fullmatch(value) is not None,
            f"Invalid SHA-256: {label}")
    return value


def positive_bytes(value: object, label: str, *, allow_zero: bool = False) -> int:
    require(type(value) is int and value >= (0 if allow_zero else 1),
            f"Invalid byte count: {label}")
    return value


def safe_path(value: object) -> str:
    require(isinstance(value, str) and bool(value), "Empty or non-string path")
    require(not any(c in value for c in ("\\", "\x00", "\n", "\r", "\t")),
            "Path contains an unsupported separator or control character")
    path = PurePosixPath(value)
    require(not path.is_absolute() and str(path) == value,
            "Path must be a canonical relative POSIX path")
    require(all(p not in ("", ".", "..") and p.lower() != ".git"
                for p in value.split("/")), "Unsafe path component")
    return value


def no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(data: bytes, label: str) -> dict:
    try:
        value = json.loads(data, object_pairs_hook=no_duplicate_keys,
                           parse_constant=lambda value: (_ for _ in ()).throw(
                               ImportFailure("Nonfinite JSON number")))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ImportFailure(f"Invalid JSON: {label}") from exc
    require(isinstance(value, dict), f"Expected JSON object: {label}")
    return value


class Git:
    def __init__(self) -> None:
        self.root = Path(self.run("rev-parse", "--show-toplevel").decode().strip())
        require(self.run("rev-parse", "--show-object-format").strip() == b"sha1",
                "This archive requires a SHA-1 Git object database")

    def run(self, *args: str, data: bytes | None = None,
            env: dict[str, str] | None = None) -> bytes:
        result = subprocess.run(["git", *args], input=data, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, env=env, check=False,
                                cwd=getattr(self, "root", None))
        # Never echo Git stderr: an authentication error can contain a URL.
        if result.returncode:
            raise ImportFailure(f"git {args[0]} failed (exit {result.returncode})")
        return result.stdout

    def object_type(self, oid: str) -> str:
        return self.run("cat-file", "-t", oid).decode().strip()

    def blob(self, oid: str) -> bytes:
        require(self.object_type(oid) == "blob", "Expected a Git blob")
        return self.run("cat-file", "blob", oid)


@dataclass(frozen=True)
class Entry:
    mode: str
    oid: str
    size: int | None
    kind: str


def tree_entries(git: Git, treeish: str) -> dict[str, Entry]:
    entries = {}
    for record in git.run("ls-tree", "-r", "-l", "-z", treeish).split(b"\x00"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, kind, oid, size = metadata.decode("ascii").split()
        path = safe_path(raw_path.decode("utf-8"))
        require(path not in entries, "Duplicate Git tree path")
        entries[path] = Entry(mode, sha1(oid, path),
                              None if size == "-" else int(size), kind)
    return entries


def regular_blob(git: Git, entries: dict[str, Entry], path: str) -> bytes:
    path = safe_path(path)
    require(path in entries, f"Required trigger file is absent: {path}")
    entry = entries[path]
    require(entry.kind == "blob" and entry.mode in ("100644", "100755"),
            f"Required trigger file is not a regular blob: {path}")
    data = git.blob(entry.oid)
    require(len(data) == entry.size, f"Git blob size mismatch: {path}")
    return data


def load_manifest(git: Git, entries: dict[str, Entry], request: dict,
                  expected_path: str) -> dict:
    require(isinstance(request, dict), "Manifest request must be an object")
    require(set(request) == {"manifest_path", "manifest_bytes", "manifest_sha256"},
            "Unexpected or missing manifest request fields")
    require(safe_path(request["manifest_path"]) == expected_path,
            "Manifest path differs from the fixed archive path")
    expected_bytes = positive_bytes(request["manifest_bytes"], expected_path)
    expected_hash = sha256(request["manifest_sha256"], expected_path)
    data = regular_blob(git, entries, expected_path)
    require(len(data) == expected_bytes, "Manifest byte count mismatch")
    require(hashlib.sha256(data).hexdigest() == expected_hash,
            "Manifest SHA-256 mismatch")
    return parse_json(data, expected_path)


def assemble_bundle(git: Git, entries: dict[str, Entry], manifest: dict,
                    manifest_path: str, output: Path) -> dict:
    bundle_size = positive_bytes(manifest.get("bundle_bytes"), "bundle")
    bundle_hash = sha256(manifest.get("bundle_sha256"), "bundle")
    parts = manifest.get("parts")
    require(isinstance(parts, list) and 0 < len(parts) <= 10000,
            "Parts must be a nonempty bounded list")
    digest = hashlib.sha256()
    size = 0
    expected_paths = set()
    parent = str(PurePosixPath(manifest_path).parent)
    with output.open("xb") as stream:
        for index, part in enumerate(parts):
            require(isinstance(part, dict) and set(part) == {"name", "bytes", "sha256"},
                    "Unexpected part record fields")
            expected_name = f"history.bundle.part{index:03d}"
            require(part["name"] == expected_name, "Part name/order mismatch")
            path = safe_path(f"{parent}/{expected_name}")
            expected_paths.add(path)
            expected_size = positive_bytes(part["bytes"], path)
            expected_hash = sha256(part["sha256"], path)
            data = regular_blob(git, entries, path)
            require(len(data) == expected_size, f"Part byte count mismatch: {path}")
            require(hashlib.sha256(data).hexdigest() == expected_hash,
                    f"Part SHA-256 mismatch: {path}")
            stream.write(data)
            digest.update(data)
            size += len(data)
    present_parts = {path for path in entries
                     if path.startswith(parent + "/history.bundle.part")}
    require(present_parts == expected_paths, "Unlisted or missing archive part")
    require(size == bundle_size, "Assembled bundle byte count mismatch")
    require(digest.hexdigest() == bundle_hash, "Assembled bundle SHA-256 mismatch")
    return {"parts": len(parts), "bytes": size, "sha256": bundle_hash}


def bundle_header(path: Path) -> tuple[list[str], dict[str, str]]:
    prerequisites = []
    references = {}
    with path.open("rb") as stream:
        version = stream.readline(256)
        require(version in (b"# v2 git bundle\n", b"# v3 git bundle\n"),
                "Unsupported Git bundle version")
        for _ in range(10000):
            line = stream.readline(8192)
            require(bool(line) and line.endswith(b"\n"), "Malformed bundle header")
            if line == b"\n":
                return prerequisites, references
            text = line[:-1].decode("utf-8")
            if text.startswith("@"):
                require(text == "@object-format=sha1", "Unsupported bundle capability")
                continue
            if text.startswith("-"):
                prerequisites.append(sha1(text[1:].split(" ", 1)[0], "prerequisite"))
                continue
            oid, ref = text.split(" ", 1)
            require(ref == "HEAD" or ref.startswith("refs/"), "Invalid bundle reference")
            require(ref not in references, "Duplicate bundle reference")
            references[ref] = sha1(oid, "bundle reference")
    raise ImportFailure("Bundle header exceeded its record limit")


def import_bundle(git: Git, bundle: Path, branch: str, commit: str,
                  expected_prerequisites: list[str]) -> dict:
    prerequisites, references = bundle_header(bundle)
    require(prerequisites == expected_prerequisites, "Bundle prerequisite mismatch")
    ref = "refs/heads/" + branch
    require(references.get(ref) == commit, "Bundle branch/head mismatch")
    git.run("bundle", "verify", str(bundle))
    # No destination ref and no FETCH_HEAD write: only populate the object store.
    git.run("-c", "gc.auto=0", "fetch", "--no-tags", "--no-write-fetch-head",
            "--no-recurse-submodules", "--no-auto-maintenance", str(bundle), ref)
    require(git.object_type(commit) == "commit", "Imported source is not a commit")
    return {"branch": branch, "commit": commit,
            "prerequisites": prerequisites, "advertised_refs": references}


def verify_inventory(git: Git, commit: str, manifest: dict, *, required: bool) -> int | None:
    inventory = manifest.get("all_tracked_files")
    if inventory is None:
        require(not required, "Base manifest lacks its complete file inventory")
        return None
    require(isinstance(inventory, list), "File inventory must be a list")
    expected = {}
    for record in inventory:
        require(isinstance(record, dict) and set(record) == {"path", "mode", "sha", "size"},
                "Invalid inventory record fields")
        path = safe_path(record["path"])
        require(path not in expected, "Duplicate inventory path")
        mode = record["mode"]
        require(mode in ("100644", "100755", "120000"), "Unsupported inventory mode")
        expected[path] = Entry(mode, sha1(record["sha"], path),
                               positive_bytes(record["size"], path, allow_zero=True), "blob")
    actual = tree_entries(git, commit)
    require(actual == expected, "Imported source tree differs from complete file inventory")
    return len(expected)


def allowed_change(path: str) -> bool:
    return (path.startswith(".github/") or path.startswith("native_builds/")
            or path in OPTIONAL_OVERLAYS)


def build_final_tree(git: Git, trigger: str, source: str,
                     trigger_entries: dict[str, Entry], temporary: Path) -> tuple[str, dict]:
    original = tree_entries(git, source)
    expected = dict(original)
    require(".github" not in original and "native_builds" not in original,
            "Source publication directories cannot be files")
    require(".github" not in trigger_entries and "native_builds" not in trigger_entries,
            "Trigger publication directories cannot be files")
    github_entries = {p: e for p, e in trigger_entries.items() if p.startswith(".github/")}
    require(bool(github_entries), "Trigger .github directory is absent")
    for path in list(expected):
        if path.startswith(".github/"):
            del expected[path]
    expected.update(github_entries)
    native_overlay = {p: e for p, e in trigger_entries.items()
                      if p.startswith("native_builds/")}
    expected.update(native_overlay)
    for path in OPTIONAL_OVERLAYS:
        if path in trigger_entries:
            require(trigger_entries[path].kind == "blob"
                    and trigger_entries[path].mode in ("100644", "100755"),
                    "Optional publication file is not a regular blob")
            expected[path] = trigger_entries[path]
    # A file/directory collision is an error, never a silent deletion of source.
    for path in expected:
        for parent in PurePosixPath(path).parents:
            require(str(parent) not in expected, "Overlay creates a file/directory collision")
    changed = sorted(p for p in original.keys() | expected.keys()
                     if original.get(p) != expected.get(p))
    require(all(allowed_change(p) for p in changed), "Change outside permitted publication paths")
    require(not any(p.startswith("research_archive/iteration9/")
                    or p.startswith("research_archive/iteration10/") for p in expected),
            "Archive transport files unexpectedly occur in the research source tree")
    environment = os.environ.copy()
    environment["GIT_INDEX_FILE"] = str(temporary / "publication.index")
    git.run("read-tree", source, env=environment)
    removals = b"".join(("0 " + "0" * 40 + "\t" + p + "\0").encode()
                        for p in changed if p in original)
    additions = b"".join((f"{expected[p].mode} {expected[p].oid}\t{p}\0").encode()
                         for p in changed if p in expected)
    if removals:
        git.run("update-index", "-z", "--index-info", data=removals, env=environment)
    if additions:
        git.run("update-index", "-z", "--index-info", data=additions, env=environment)
    final_tree = sha1(git.run("write-tree", env=environment).decode().strip(), "final tree")
    require(tree_entries(git, final_tree) == expected, "Final index/tree content mismatch")
    require(git.run("rev-parse", f"{trigger}:.github").strip()
            == git.run("rev-parse", f"{final_tree}:.github").strip(),
            "Final .github directory is not exactly the trigger directory")
    preserved_native = sum(p.startswith("native_builds/") and p not in native_overlay
                           for p in original)
    return final_tree, {"source_files": len(original), "final_files": len(expected),
                        "changed_paths": changed, "github_exact": True,
                        "trigger_native_files": len(native_overlay),
                        "source_only_native_files_preserved": preserved_native,
                        "all_other_source_paths_unchanged": True}


def create_merge(git: Git, tree: str, trigger: str, source: str) -> str:
    require(source != trigger, "Trigger and source must be distinct commits")
    environment = os.environ.copy()
    environment.update({"GIT_AUTHOR_NAME": "github-actions[bot]",
                        "GIT_AUTHOR_EMAIL": "41898282+github-actions[bot]@users.noreply.github.com",
                        "GIT_COMMITTER_NAME": "github-actions[bot]",
                        "GIT_COMMITTER_EMAIL": "41898282+github-actions[bot]@users.noreply.github.com"})
    message = ("Publish verified complete GBS research history\n\n"
               f"Publication trigger: {trigger}\nResearch source: {source}\n"
               "Preserve publication history and trigger .github tree.\n")
    merge = sha1(git.run("commit-tree", tree, "-p", trigger, "-p", source,
                         data=message.encode(), env=environment).decode().strip(), "merge")
    headers = git.run("cat-file", "commit", merge).split(b"\n\n", 1)[0].decode().splitlines()
    require([line[7:] for line in headers if line.startswith("parent ")] == [trigger, source],
            "Merge parent order mismatch")
    require(headers[0] == "tree " + tree, "Merge tree mismatch")
    git.run("merge-base", "--is-ancestor", trigger, merge)
    git.run("merge-base", "--is-ancestor", source, merge)
    return merge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate locally; never push")
    parser.add_argument("--trigger", help="Local trigger commit (only with --dry-run)")
    arguments = parser.parse_args()
    require(arguments.dry_run or arguments.trigger is None,
            "--trigger is only permitted with --dry-run")
    git = Git()
    head = sha1(git.run("rev-parse", "HEAD").decode().strip(), "checkout HEAD")
    trigger = arguments.trigger or os.environ.get("GITHUB_SHA") or head
    trigger = sha1(trigger, "trigger")
    require(git.object_type(trigger) == "commit", "Trigger is not a commit")
    if not arguments.dry_run:
        require(os.environ.get("GITHUB_ACTIONS") == "true", "Publishing requires GitHub Actions")
        require(os.environ.get("GITHUB_REPOSITORY") == REPOSITORY, "Wrong Actions repository")
        require(os.environ.get("GITHUB_EVENT_NAME") == "push", "Only push events may publish")
        require(os.environ.get("GITHUB_REF") == "refs/heads/main", "Only main may publish")
        require(os.environ.get("GITHUB_SHA") == trigger and head == trigger,
                "Checkout HEAD must equal GITHUB_SHA")
    trigger_entries = tree_entries(git, trigger)
    executing_script = Path(__file__).read_bytes()
    require(executing_script == regular_blob(git, trigger_entries, SCRIPT_PATH),
            "Executing script differs from the trigger-commit script")
    request_bytes = regular_blob(git, trigger_entries, REQUEST_PATH)
    request = parse_json(request_bytes, REQUEST_PATH)
    require(set(request) in ({"format", "repository", "base"},
                            {"format", "repository", "base", "delta"}),
            "Unexpected publication request fields")
    require(request["format"] == "deziroGBS-publication-request-v1", "Unknown request format")
    require(request["repository"] == REPOSITORY, "Publication request names the wrong repository")
    base = load_manifest(git, trigger_entries, request["base"], BASE_MANIFEST)
    require(base.get("format") == "deziroGBS-complete-git-bundle-parts-v1", "Unknown base format")
    require(base.get("original_commit") == BASE_COMMIT, "Wrong base source commit")
    require(base.get("original_branch") == BASE_BRANCH, "Wrong base source branch")
    audit = {"mode": "dry-run" if arguments.dry_run else "publish",
             "repository": REPOSITORY, "trigger_commit": trigger,
             "request_sha256": hashlib.sha256(request_bytes).hexdigest()}
    with tempfile.TemporaryDirectory(prefix="gbs-history-import-") as temporary_name:
        temporary = Path(temporary_name)
        base_bundle = temporary / "base.bundle"
        audit["base"] = assemble_bundle(git, trigger_entries, base, BASE_MANIFEST, base_bundle)
        audit["base"].update(import_bundle(git, base_bundle, BASE_BRANCH, BASE_COMMIT, []))
        audit["base"]["inventory_files"] = verify_inventory(git, BASE_COMMIT, base, required=True)
        source = BASE_COMMIT
        if "delta" in request:
            delta = load_manifest(git, trigger_entries, request["delta"], DELTA_MANIFEST)
            require(delta.get("format") == "deziroGBS-incremental-git-bundle-parts-v1",
                    "Unknown delta format")
            require(delta.get("prerequisite_commit") == BASE_COMMIT, "Wrong delta prerequisite")
            require(delta.get("source_branch") == DELTA_BRANCH, "Wrong delta source branch")
            source = sha1(delta.get("source_commit"), "delta source commit")
            expected_tree = sha1(delta.get("source_tree"), "delta source tree")
            require(source != BASE_COMMIT, "Delta source cannot equal its prerequisite")
            delta_bundle = temporary / "delta.bundle"
            audit["delta"] = assemble_bundle(git, trigger_entries, delta, DELTA_MANIFEST, delta_bundle)
            audit["delta"].update(import_bundle(git, delta_bundle, DELTA_BRANCH, source, [BASE_COMMIT]))
            git.run("merge-base", "--is-ancestor", BASE_COMMIT, source)
            actual_tree = git.run("rev-parse", f"{source}^{{tree}}").decode().strip()
            require(actual_tree == expected_tree, "Delta source tree SHA-1 mismatch")
            audit["delta"]["inventory_files"] = verify_inventory(git, source, delta, required=False)
        source_tree = sha1(git.run("rev-parse", f"{source}^{{tree}}").decode().strip(), "source tree")
        final_tree, tree_audit = build_final_tree(git, trigger, source, trigger_entries, temporary)
        merge = create_merge(git, final_tree, trigger, source)
        audit.update({"source_commit": source, "source_tree": source_tree,
                      "final_tree": final_tree, "merge_commit": merge,
                      "parents": [trigger, source], "tree_validation": tree_audit,
                      "pushed": False})
        # Check immediately before ordinary push. A later race is also rejected
        # by Git's normal non-fast-forward protection; no force option exists.
        if not arguments.dry_run:
            remote = git.run("ls-remote", "--exit-code", REMOTE_URL, "refs/heads/main")
            require(remote.decode().strip() == trigger + "\trefs/heads/main",
                    "Remote main advanced or changed; refusing to overwrite progress")
            git.run("merge-base", "--is-ancestor", trigger, merge)
            git.run("push", "--porcelain", REMOTE_URL, merge + ":refs/heads/main")
            audit["pushed"] = True
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (ImportFailure, OSError, UnicodeError, KeyError, TypeError, ValueError) as error:
        print("History import stopped: " + str(error), file=sys.stderr)
        sys.exit(1)
