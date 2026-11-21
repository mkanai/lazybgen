"""The default worker count follows the CPUs the process may use.

With num_threads=0 a block decode starts one worker per usable CPU. On a
cluster, a job's CPUs are usually a subset of the machine's (a SLURM cpuset,
taskset, or a container's cgroup quota), so counting the machine's cores would
oversubscribe the job.
"""

import os
import shutil
import subprocess
import sys

import pytest
from lazybgen.reader import _default_num_threads


def test_default_is_at_least_one_and_at_most_the_machine():
    n = _default_num_threads()
    assert 1 <= n <= (os.cpu_count() or n)


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="Linux affinity API")
def test_default_respects_this_process_affinity():
    assert _default_num_threads() <= len(os.sched_getaffinity(0))


@pytest.mark.skipif(
    not hasattr(os, "sched_getaffinity") or shutil.which("taskset") is None,
    reason="needs Linux and taskset",
)
def test_default_respects_a_narrowed_affinity_mask():
    cpu = sorted(os.sched_getaffinity(0))[0]
    out = subprocess.run(
        ["taskset", "-c", str(cpu), sys.executable, "-c",
         "from lazybgen.reader import _default_num_threads as f; print(f())"],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "1"
