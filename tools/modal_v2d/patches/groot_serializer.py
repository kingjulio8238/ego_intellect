"""Applied at image build (tools/modal_v2d/images.py). Verified on Vast 2026-11-17:
without it the closed-loop eval fails "Video key 'front' must be a numpy array. Got dict".

Swap the n1.7-release GR00T server's MsgSerializer for the one on GR00T main, which the
repo's closed-loop client (groot_finetune/closed_loop/policy_client.py) was ported from:
main speaks msgpack_numpy and still decodes the older __ndarray_class__ format."""
import re, sys, urllib.request
path = (sys.argv[1] if len(sys.argv) > 1 else "/opt/Isaac-GR00T") + "/gr00t/policy/server_client.py"
main = urllib.request.urlopen("https://raw.githubusercontent.com/NVIDIA/Isaac-GR00T/3b49067943eb1ab2eb2203b93d461c4390c78da3/gr00t/policy/server_client.py").read().decode()
def extract(src):
    start = src.index("class MsgSerializer")
    m = re.search(r"\n(@dataclass|class |def )", src[start + 10:])
    return start, start + 10 + m.start() + 1
s, e = extract(main); new_cls = main[s:e]
cur = open(path).read()
if "msgpack_numpy" in cur.split("class MsgSerializer")[1][:3000]:
    print("already patched"); sys.exit(0)
cs, ce = extract(cur)
cur = cur[:cs] + new_cls + cur[ce:]
header = "import functools\nimport msgpack_numpy as mnp\nfrom gr00t.data.utils import to_json_serializable\n"
cur = cur.replace("import msgpack\n", "import msgpack\n" + header, 1)
open(path, "w").write(cur)
print("patched", path)
