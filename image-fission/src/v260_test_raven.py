"""v260 单图测试 — raven (走 build_workflow, 修正后)
"""
import sys, time
sys.path.insert(0, 'src')
from v260_api_call import build_workflow, post_workflow, poll_done, collect_output
from v260_api_call import POSITIVE_PREFIX, POSITIVE_SUFFIX

prompt_pos = POSITIVE_PREFIX + "raven with outspread wings, dark plumage" + POSITIVE_SUFFIX
wf = build_workflow(prompt_pos, 42)
wf["9"]["inputs"]["filename_prefix"] = "v260_test_raven"

print("SUBMIT v260_test_raven via build_workflow...")
pid = post_workflow("test_raven_42", wf)
print("pid=", pid, ", polling...")
e = poll_done(pid, timeout_s=900)
print("status:", e.get("status", {}))
for f in collect_output(e):
    print("OUT:", f)
