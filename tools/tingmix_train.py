#!/usr/bin/env python3
"""tingmix-train: host job manager for long TingMix training runs (Docker, independent of any shell/agent session).

  tingmix-train launch <exp> [--dry-run]     start <exp> from $TINGMIX_HOME/experiments/<exp>.yaml
  tingmix-train status [<exp>] [--json]      container / trainer status, last step and eval, GPU memory
  tingmix-train logs <exp> [-n N] [-f]       trainer log (kept in the run dir, survives the container)
  tingmix-train resume <exp> [--new-code] [--force]
  tingmix-train stop <exp>                   SIGTERM -> trainer checkpoints and exits (state PAUSED)
  tingmix-train reconcile                    supervisor tick (systemd timer): restart runs that should be running

Each run is ONE detached container `tingmix-<exp>` (fixed name: Docker refuses a second one) with
--restart=on-failure:N, running a frozen copy of the code (`git archive` of a commit) from the run dir. Checkpoints,
config, logs and status live in the run dir on the host. The trainer itself takes a run-dir lock, exits 0 on permanent
errors (status FAILED, no restart) and non-zero on transient ones (Docker restarts it, it resumes from resume.pt).
`reconcile` covers what Docker's policy does not: host/daemon restarts and exhausted restart budgets (bounded).

Experiment file: trainer options (training/large/train.py) plus a `job:` block:
  job: {gpu: 7, restarts: 3, memory: 96g, cpus: 32, image: <docker image>}
Host settings come from the environment (or $TINGMIX_HOME/host.env, KEY=VALUE lines):
  TINGMIX_HOME (runs/, experiments/, site/, hf/), TINGMIX_STATE (RUNNING_JOBS.json), TINGMIX_IMAGE,
  TINGMIX_GPU_RENDER ("7=renderD160,4=renderD168"), TINGMIX_RO_MOUNTS (comma list of read-only host dirs)."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAX_RECONCILE_RESTARTS = 5


def env():
    home = Path(os.environ.get("TINGMIX_HOME", Path.home() / "tingmix-work"))
    hostenv = home / "host.env"
    e = {}
    if hostenv.exists():
        for line in hostenv.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                e[k.strip()] = v.strip()
    e.update({k: v for k, v in os.environ.items() if k.startswith("TINGMIX_")})
    e.setdefault("TINGMIX_HOME", str(home))
    e.setdefault("TINGMIX_STATE", str(Path.home() / ".local/state/tingmix"))
    return e


E = env()
HOME, STATE = Path(E["TINGMIX_HOME"]), Path(E["TINGMIX_STATE"])
JOBS = STATE / "RUNNING_JOBS.json"


def sh(*c, check=False):
    r = subprocess.run(list(c), capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f"{' '.join(c[:3])}...: {r.stderr.strip()[-500:]}")
    return r


class Jobs:
    """RUNNING_JOBS.json under an exclusive file lock (manager and timer may run concurrently)."""

    def __enter__(self):
        STATE.mkdir(parents=True, exist_ok=True)
        self.lk = open(STATE / ".jobs.lock", "w")
        fcntl.flock(self.lk, fcntl.LOCK_EX)
        self.d = json.loads(JOBS.read_text()) if JOBS.exists() else {"jobs": []}
        self.d.setdefault("jobs", [])
        return self

    def get(self, exp):
        return next((j for j in self.d["jobs"] if j["exp"] == exp), None)

    def __exit__(self, *exc):
        self.d["updated"] = time.strftime("%F %T")
        tmp = JOBS.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, indent=1))
        os.replace(tmp, JOBS)
        self.lk.close()


def load_exp(exp):
    import yaml

    p = HOME / "experiments" / f"{exp}.yaml"
    if not p.exists():
        raise SystemExit(f"no experiment file {p}")
    cfg = yaml.safe_load(p.read_text()) or {}
    job = cfg.pop("job", {})
    if "gpu" not in job:
        raise SystemExit(f"{p}: job.gpu is required")
    cfg.setdefault("out", str(HOME / "runs" / exp))
    return cfg, job, p


def cname(exp):
    return f"tingmix-{exp}"


def cstate(exp):
    r = sh("docker", "inspect", "-f", "{{.State.Status}} {{.State.ExitCode}} {{.RestartCount}} {{.State.StartedAt}}", cname(exp))
    if r.returncode:
        return None
    s, code, rc, started = r.stdout.split()
    return {"status": s, "exit": int(code), "restarts": int(rc), "started": started}


def run_status(run):
    p = Path(run) / "status.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def render_map():
    m = E.get("TINGMIX_GPU_RENDER", "4=renderD168,5=renderD184,6=renderD176,7=renderD160")
    return dict(x.split("=") for x in m.split(","))


def snapshot_code(run: Path, commit="HEAD"):
    """Freeze the code of a commit into <run>/code (the trainer never runs from the live working tree)."""
    dirty = sh("git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no", "--", "src", "training", "tools").stdout.strip()
    if dirty and commit == "HEAD":
        print(f"warning: uncommitted changes are NOT included (snapshot = {commit}):\n{dirty}", file=sys.stderr)
    sha = sh("git", "-C", str(REPO), "rev-parse", commit, check=True).stdout.strip()
    code = run / "code"
    if code.exists():
        shutil.rmtree(code)
    code.mkdir(parents=True)
    git = subprocess.Popen(["git", "-C", str(REPO), "archive", "--format=tar", sha, "src", "training", "tools"], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(code)], stdin=git.stdout, check=True)
    if git.wait():
        raise SystemExit("git archive failed")
    (code / "COMMIT").write_text(sha + "\n")
    return sha


def docker_create(exp, cfg, job, sha, allow_code_change=False):
    import yaml

    run = Path(cfg["out"])
    tcfg = dict(cfg)
    if allow_code_change:
        tcfg["allow_code_change"] = True
    (run / "trainer.yaml").write_text(yaml.safe_dump(tcfg, allow_unicode=True, sort_keys=False))
    image = job.get("image") or E.get("TINGMIX_IMAGE")
    if not image:
        raise SystemExit("set TINGMIX_IMAGE (host.env) or job.image")
    render = render_map()[str(job["gpu"])]
    code, site = run / "code", HOME / "site"
    mounts = [f"{HOME}:{HOME}"]
    mounts += [f"{d}:{d}:ro" for d in E.get("TINGMIX_RO_MOUNTS", "").split(",") if d]
    uid, gid = os.getuid(), os.getgid()
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        cname(exp),
        "--restart",
        f"on-failure:{job.get('restarts', 3)}",
        "--label",
        "tingmix=1",
        "--label",
        f"tingmix.exp={exp}",
        "--label",
        f"tingmix.gpu={job['gpu']}",
        "--label",
        f"tingmix.commit={sha}",
        "--device=/dev/kfd",
        f"--device=/dev/dri/{render}",
        "--group-add",
        "video",
        "--group-add",
        str(job.get("render_gid", 993)),
        "--user",
        f"{uid}:{gid}",
        "--ipc=host",
        "--shm-size=32g",
        "--memory",
        str(job.get("memory", "96g")),
        "--cpus",
        str(job.get("cpus", 32)),
        "-e",
        f"HOME={HOME}/home",
        "-e",
        "USER=tingmix",
        "-e",
        "LOGNAME=tingmix",
        "-e",
        f"HF_HOME={HOME}/hf",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        "OMP_NUM_THREADS=4",
        "-e",
        f"PYTHONPATH={code}/src:{code}:{site}",
        "-e",
        f"TINGMIX_COMMIT={sha}",
        "-e",
        f"TINGMIX_EXP={exp}",
        "-e",
        "PYTORCH_HIP_ALLOC_CONF=expandable_segments:True",
        # per-run MIOpen db/cache: a shared $HOME/.config/miopen races between containers (rename of *.ufdb.txt.temp)
        "-e",
        f"MIOPEN_USER_DB_PATH={run}/.miopen",
        "-e",
        f"MIOPEN_CUSTOM_CACHE_DIR={run}/.miopen",
        "-w",
        str(code),
    ]
    for m in mounts:
        cmd += ["-v", m]
    cmd += [image, "bash", "-c", f"exec python -m training.large.train --config {run}/trainer.yaml >> {run}/train.log 2>&1"]
    sh(*cmd, check=True)


def launch(a):
    cfg, job, src = load_exp(a.exp)
    run = Path(cfg["out"])
    if cstate(a.exp):
        raise SystemExit(f"container {cname(a.exp)} already exists; use status/resume/stop")
    if (run / "resume.pt").exists() and not a.force:
        raise SystemExit(f"{run} already has a checkpoint; use `resume` (or --force to start a new container on it)")
    if a.dry_run:
        print(f"would launch {a.exp} on GPU {job['gpu']} -> {run}")
        return
    run.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, run / "experiment.yaml")
    sha = snapshot_code(run, a.commit)
    docker_create(a.exp, cfg, job, sha)
    with Jobs() as J:
        J.d["jobs"] = [j for j in J.d["jobs"] if j["exp"] != a.exp]
        J.d["jobs"].append(
            {
                "exp": a.exp,
                "container": cname(a.exp),
                "gpu": job["gpu"],
                "run": str(run),
                "commit": sha,
                "launched": time.strftime("%F %T"),
                "desired": "running",
                "reconcile_restarts": 0,
            }
        )
    print(f"launched {cname(a.exp)} on GPU {job['gpu']} (commit {sha[:10]}); log: {run}/train.log")


def resume(a):
    cfg, job, _ = load_exp(a.exp)
    run = Path(cfg["out"])
    st, cs = run_status(run), cstate(a.exp)
    if st.get("state") == "FAILED" and st.get("permanent") and not a.force:
        raise SystemExit(f"{a.exp} failed permanently: {st.get('reason')}\nfix the cause, then `resume --force` (or --new-code)")
    if st.get("state") == "DONE" and not a.force:
        raise SystemExit(f"{a.exp} is DONE ({st.get('reason')}); raise the step cap / budget and use --force to continue")
    if a.new_code:  # deliberate code change: new snapshot, trainer accepts the different code hash
        if cs:
            sh("docker", "stop", "-t", "300", cname(a.exp))
            sh("docker", "rm", cname(a.exp), check=True)
        sha = snapshot_code(run, a.commit)
        docker_create(a.exp, cfg, job, sha, allow_code_change=True)
        print(f"recreated {cname(a.exp)} with new code {sha[:10]}")
    elif cs and cs["status"] == "running":
        print(f"{cname(a.exp)} is already running")
    elif cs:
        sh("docker", "start", cname(a.exp), check=True)
        print(f"started {cname(a.exp)} (same container, same frozen code)")
    else:  # container gone: recreate on the SAME frozen code snapshot
        sha = (run / "code" / "COMMIT").read_text().strip()
        docker_create(a.exp, cfg, job, sha)
        print(f"recreated {cname(a.exp)} on frozen code {sha[:10]}")
    with Jobs() as J:
        j = J.get(a.exp)
        if j:
            j.update(
                desired="running",
                reconcile_restarts=0,
                resumed=time.strftime("%F %T"),
                commit=(run / "code" / "COMMIT").read_text().strip(),
            )


def stop(a):
    with Jobs() as J:
        j = J.get(a.exp)
        if j:
            j["desired"] = "stopped"
    r = sh("docker", "stop", "-t", "300", cname(a.exp))
    print(r.stdout.strip() or r.stderr.strip())
    print(f"status: {run_status(load_exp(a.exp)[0]['out']).get('state')}")


def last_lines(run, kinds=("train", "eval")):
    out = {}
    p = Path(run) / "train_log.jsonl"
    if p.exists():
        with open(p, "rb") as f:
            f.seek(max(0, f.seek(0, 2) - 200_000))
            for line in f.read().decode(errors="replace").splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("type") in kinds:
                    out[r["type"]] = r
    return out


def free_gib(gpu):
    r = sh("amd-smi", "metric", "-g", str(gpu), "-m")
    for tok in r.stdout.split("\n"):
        if "FREE_VRAM" in tok:
            return round(int(tok.split(":")[1].split()[0]) / 1024, 1)
    return None


def status(a):
    jobs = json.loads(JOBS.read_text())["jobs"] if JOBS.exists() else []
    rows = []
    for j in jobs:
        if a.exp and j["exp"] != a.exp:
            continue
        ll = last_lines(j["run"])
        rows.append(
            {
                **j,
                "container_state": cstate(j["exp"]),
                "trainer": run_status(j["run"]),
                "last_train": ll.get("train"),
                "last_eval": ll.get("eval"),
                "gpu_free_gib": free_gib(j["gpu"]),
            }
        )
    if a.json:
        print(json.dumps(rows, indent=1, ensure_ascii=False))
        return
    if not rows:
        print("no jobs")
    for r in rows:
        c, t, lt, le = r["container_state"] or {}, r["trainer"], r["last_train"] or {}, r["last_eval"] or {}
        print(f"== {r['exp']}  GPU{r['gpu']} (free {r['gpu_free_gib']} GiB)  desired={r['desired']}  commit {r['commit'][:10]}")
        print(
            f"   container: {c.get('status', 'MISSING')} exit={c.get('exit')} docker_restarts={c.get('restarts')} reconcile_restarts={r['reconcile_restarts']}"
        )
        print(f"   trainer:   {t.get('state')} {t.get('reason', '')} ({t.get('time', '')})")
        if lt:
            print(
                f"   train:     step {lt['step']} ep {lt['epoch']} loss {lt['loss']} lr {lt['lr']} {lt.get('audio_h_per_h')} audio-h/h, train {lt.get('train_h')} h"
            )
        if le:
            print(f"   eval:      {json.dumps({k: v for k, v in le.items() if k not in ('type',)}, ensure_ascii=False)}")


def logs(a):
    run = Path(load_exp(a.exp)[0]["out"])
    p = run / "train.log"
    if not p.exists():
        raise SystemExit(f"no log yet: {p}")
    os.execvp("tail", ["tail", "-n", str(a.n)] + (["-F"] if a.f else []) + [str(p)])


def reconcile(a):
    """Restart runs that should be running but are not (host reboot, daemon restart, exhausted Docker retries)."""
    with Jobs() as J:
        for j in J.d["jobs"]:
            cs, st = cstate(j["exp"]), run_status(j["run"])
            j["last_seen"] = {"container": cs and cs["status"], "trainer": st.get("state"), "at": time.strftime("%F %T")}
            if j["desired"] != "running" or (cs and cs["status"] in ("running", "restarting", "created")):
                continue
            if st.get("state") == "DONE" or (st.get("state") == "FAILED" and st.get("permanent")):
                j["desired"] = st["state"].lower()
                continue
            if j["reconcile_restarts"] >= MAX_RECONCILE_RESTARTS:
                j["desired"] = "failed"
                j["note"] = "reconcile restart budget exhausted; inspect logs, then `tingmix-train resume --force`"
                continue
            j["reconcile_restarts"] += 1
            if cs:
                r = sh("docker", "start", j["container"])
            else:
                cfg, job, _ = load_exp(j["exp"])
                docker_create(j["exp"], cfg, job, j["commit"])
                r = None
            print(f"reconcile: restarted {j['exp']} ({'start' if cs else 'recreate'}) {r.stderr.strip() if r else ''}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="tingmix-train", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("launch")
    x.add_argument("exp")
    x.add_argument("--commit", default="HEAD")
    x.add_argument("--force", action="store_true")
    x.add_argument("--dry-run", action="store_true")
    x = s.add_parser("status")
    x.add_argument("exp", nargs="?")
    x.add_argument("--json", action="store_true")
    x = s.add_parser("logs")
    x.add_argument("exp")
    x.add_argument("-n", type=int, default=40)
    x.add_argument("-f", action="store_true")
    x = s.add_parser("resume")
    x.add_argument("exp")
    x.add_argument("--new-code", action="store_true", help="re-snapshot code at --commit and resume with --allow-code-change")
    x.add_argument("--commit", default="HEAD")
    x.add_argument("--force", action="store_true")
    x = s.add_parser("stop")
    x.add_argument("exp")
    s.add_parser("reconcile")
    a = p.parse_args(argv)
    {"launch": launch, "status": status, "logs": logs, "resume": resume, "stop": stop, "reconcile": reconcile}[a.cmd](a)


if __name__ == "__main__":
    main()
