"""The default worker count follows the CPUs the process may use.

With num_threads=0 a block decode starts one worker per usable CPU. On a
cluster, a job's CPUs are usually a subset of the machine's (a SLURM cpuset,
taskset, or a container's cgroup quota), so counting the machine's cores would
oversubscribe the job. On Linux the affinity mask is read on every decode and
the tightest cgroup quota once per process; the quota lookup takes mount points
and roots from mountinfo, which the cgroup tests below fake.
"""

import os
import shutil
import subprocess
import sys

import pytest
from lazybgen.reader import _cgroup_cpu_limit_from, _default_num_threads


def test_default_is_at_least_one_and_at_most_the_machine():
    n = _default_num_threads()
    assert 1 <= n <= (os.cpu_count() or n)


@pytest.mark.skipif(
    not hasattr(os, "sched_getaffinity") or shutil.which("taskset") is None,
    reason="needs Linux and taskset",
)
def test_default_respects_a_narrowed_affinity_mask():
    cpu = sorted(os.sched_getaffinity(0))[0]
    code = "from lazybgen.reader import _default_num_threads as f; print(f())"
    out = subprocess.run(
        ["taskset", "-c", str(cpu), sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "1"


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="Linux affinity API")
def test_default_follows_an_affinity_change_after_first_use():
    """A mask narrowed after the default was first computed is still honored."""
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("needs two or more CPUs")
    code = (
        "import os; from lazybgen.reader import _default_num_threads as f; f(); "
        "os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[0]}); print(f())"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "1"


def _cgroup_tree(tmp_path, mounts, cgroup_lines, files):
    """Write a fake mountinfo, /proc/self/cgroup and cgroup files under tmp_path."""
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "".join(
            f"{30 + i} 1 0:{30 + i} {root} {point} rw,nosuid - {fstype} cgroup rw{',' + opts if opts else ''}\n"
            for i, (fstype, root, point, opts) in enumerate(mounts)
        )
    )
    cgroup = tmp_path / "cgroup"
    cgroup.write_text("".join(line + "\n" for line in cgroup_lines))
    fs = tmp_path / "fs"
    for rel, text in files.items():
        f = fs / rel.lstrip("/")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    return str(mountinfo), str(cgroup), str(fs)


def test_cgroup_v2_quota_on_a_nested_container_root(tmp_path):
    """The mount is rooted at /job, so /job/step lives at <mount>/step, not <mount>/job/step."""
    args = _cgroup_tree(
        tmp_path,
        [("cgroup2", "/job", "/sys/fs/cgroup", "")],
        ["0::/job/step"],
        {"/sys/fs/cgroup/cpu.max": "max 100000\n", "/sys/fs/cgroup/step/cpu.max": "200000 100000\n"},
    )
    assert _cgroup_cpu_limit_from(*args) == 2


def test_cgroup_v2_takes_the_tightest_ancestor_and_rounds_up(tmp_path):
    args = _cgroup_tree(
        tmp_path,
        [("cgroup2", "/", "/sys/fs/cgroup", "")],
        ["0::/a/b"],
        {
            "/sys/fs/cgroup/a/cpu.max": "250000 100000\n",  # 2.5 CPUs -> 3
            "/sys/fs/cgroup/a/b/cpu.max": "max 100000\n",
        },
    )
    assert _cgroup_cpu_limit_from(*args) == 3


def test_cgroup_v1_controller_order_need_not_match_the_mount(tmp_path):
    """/proc/self/cgroup may list cpuacct,cpu for a hierarchy mounted at cpu,cpuacct."""
    args = _cgroup_tree(
        tmp_path,
        [("cgroup", "/", "/sys/fs/cgroup/cpu,cpuacct", "cpu,cpuacct")],
        ["4:cpuacct,cpu:/slurm/job1"],
        {
            "/sys/fs/cgroup/cpu,cpuacct/slurm/job1/cpu.cfs_quota_us": "400000\n",
            "/sys/fs/cgroup/cpu,cpuacct/slurm/job1/cpu.cfs_period_us": "100000\n",
        },
    )
    assert _cgroup_cpu_limit_from(*args) == 4


def test_no_quota_means_no_limit(tmp_path):
    args = _cgroup_tree(
        tmp_path,
        [("cgroup2", "/", "/sys/fs/cgroup", ""), ("cgroup", "/", "/sys/fs/cgroup/memory", "memory")],
        ["0::/user.slice", "5:memory:/user.slice"],
        {"/sys/fs/cgroup/user.slice/cpu.max": "max 100000\n"},
    )
    assert _cgroup_cpu_limit_from(*args) == 0


def test_an_unrelated_cgroup_mount_does_not_cap_the_process(tmp_path):
    """A second cgroup2 mount of some other subtree says nothing about this process.

    Here the process sits in an unlimited /user.slice under the root mount,
    while a monitoring agent has bind-mounted a two-CPU container's cgroup
    elsewhere. Only a mount whose root contains the process's cgroup counts.
    """
    args = _cgroup_tree(
        tmp_path,
        [("cgroup2", "/", "/sys/fs/cgroup", ""), ("cgroup2", "/docker/abc", "/mnt/agent/abc", "")],
        ["0::/user.slice"],
        {
            "/sys/fs/cgroup/user.slice/cpu.max": "max 100000\n",
            "/mnt/agent/abc/cpu.max": "200000 100000\n",
        },
    )
    assert _cgroup_cpu_limit_from(*args) == 0


def test_a_namespaced_path_no_mount_resolves_falls_back_to_the_mount_point(tmp_path):
    """Inside a cgroup namespace the path may match no mount root; the mount point is then read."""
    args = _cgroup_tree(
        tmp_path,
        [("cgroup2", "/docker/abc", "/sys/fs/cgroup", "")],
        ["0::/"],
        {"/sys/fs/cgroup/cpu.max": "300000 100000\n"},
    )
    assert _cgroup_cpu_limit_from(*args) == 3
