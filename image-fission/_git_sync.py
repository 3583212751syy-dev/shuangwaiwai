# -*- coding: utf-8 -*-
"""自动把图裂变代码/成品同步到 git 仓库（shuangwaiwai master）+ 本机 E 盘双存。
- 代码源：E:/Desktop/双接口/image-fission/_*.py  → D 盘主仓库 image-fission/_*.py
- 成品源：E:/Desktop/双接口/image-fission/outputs/_ipa_lock/ → D 盘主仓库 outputs/_ipa_lock/
- commit + push 到 shuangwaiwai master（SSH key 已配好）

用法:
  python _git_sync.py --dry-run          # 只看会同步什么
  python _git_sync.py --commit "msg"     # 同步+提交+推送
  python _git_sync.py --commit "msg" --no-push
"""
import argparse, os, shutil, subprocess, sys, glob, datetime

E_ROOT = r"E:/Desktop/双接口/image-fission"
D_ROOT = r"D:/Users/Administrator/Desktop/双接口"

def run(cmd, cwd=None, capture=True):
    print("[RUN]", " ".join(cmd) if isinstance(cmd, list) else cmd, file=sys.stderr)
    r = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str),
                       capture_output=capture, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print("[ERR]", (r.stderr or r.stdout or "")[-600:], file=sys.stderr)
    return r

def sync_paths(dry=False):
    """返回 (src,dst) 列表：E盘引擎py + E盘成品。"""
    pairs = []
    # 只同步本次任务相关的新引擎/校验脚本（避免全量 665 个文件）
    patterns = ["_fz_ipa_lock.py", "_zai_decompose*.py", "_zai_pxmeasure.py",
                "_zai_spec*.py", "_zai_batch_driver.py", "_zai_bg_restore.py", "_git_sync.py"]
    for pat in patterns:
        for src in glob.glob(os.path.join(E_ROOT, pat)):
            dst = os.path.join(D_ROOT, "image-fission", os.path.basename(src))
            pairs.append((src, dst))
    # D 盘的 _zai_* 模块（本轮新增的拆解/校验引擎）也纳入同步
    for pat in ["_zai_decompose*.py", "_zai_pxmeasure.py", "_zai_spec*.py",
                "_zai_batch_driver.py", "_zai_bg_restore.py"]:
        for src in glob.glob(os.path.join(D_ROOT, "image-fission", pat)):
            dst = os.path.join(E_ROOT, os.path.basename(src))  # D -> E 也存一份
            if not os.path.exists(dst):
                pairs.append((src, dst))
    # 成品目录（图像）
    src_out = os.path.join(E_ROOT, "outputs", "_ipa_lock")
    dst_out = os.path.join(D_ROOT, "outputs", "_ipa_lock")
    if os.path.isdir(src_out):
        for f in glob.glob(os.path.join(src_out, "*.jpg")):
            pairs.append((f, os.path.join(dst_out, os.path.basename(f))))
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--commit", default="")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    pairs = sync_paths()
    print(f"[sync] {len(pairs)} files", file=sys.stderr)
    n = 0
    for src, dst in pairs:
        if not os.path.exists(src):
            continue
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src) \
                and os.path.getmtime(dst) >= os.path.getmtime(src):
            continue
        print(f"  {src} -> {dst}", file=sys.stderr)
        n += 1
        if not args.dry_run:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
    print(f"[sync] copied {n} files (E盘保留原件)", file=sys.stderr)
    if args.dry_run or not args.commit:
        return

    # git add 只加本次相关路径
    addpaths = ["image-fission/_fz_ipa_lock.py", "image-fission/_zai_*",
                "image-fission/_git_sync.py", "outputs/_ipa_lock/"]
    run(["git", "add", "-f"] + addpaths, cwd=D_ROOT, capture=False)
    r = run(["git", "diff", "--cached", "--name-only"], cwd=D_ROOT)
    staged = (r.stdout or "").strip().splitlines()
    print(f"[git] staged {len(staged)} files", file=sys.stderr)
    if not staged:
        print("[git] nothing to commit", file=sys.stderr)
        return
    msg = args.commit or ("auto: fission sync " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    r = run(["git", "commit", "-m", msg], cwd=D_ROOT, capture=False)
    if args.no_push or r.returncode != 0:
        print("[DONE] (no push)", file=sys.stderr)
        return

    # 鲁棒推送：先 fetch；若远端领先（分叉），用 patch 应用到远端基线的新分支，绝不 force
    run(["git", "fetch", "shuangwaiwai", "master"], cwd=D_ROOT, capture=False)
    ahead = run(["git", "rev-list", "--count", "HEAD..shuangwaiwai/master"], cwd=D_ROOT).stdout.strip()
    cur_branch = run(["git", "branch", "--show-current"], cwd=D_ROOT).stdout.strip()
    if ahead == "0":
        run(["git", "push", "shuangwaiwai", "HEAD:master"], cwd=D_ROOT, capture=False)
    else:
        print(f"[git] remote ahead by {ahead} commits → patch-based safe push", file=sys.stderr)
        sha = run(["git", "rev-parse", "HEAD"], cwd=D_ROOT).stdout.strip()
        sym = "_sync_tmp_%s" % sha[:8]
        patch = os.path.join(os.environ.get("TEMP", "/tmp"), sym + ".patch")
        with open(patch, "w", encoding="utf-8") as f:
            f.write(run(["git", "format-patch", "-1", sha, "--stdout"], cwd=D_ROOT).stdout)
        run(["git", "checkout", "-b", sym, "shuangwaiwai/master"], cwd=D_ROOT, capture=False)
        r2 = run(["git", "am", patch], cwd=D_ROOT, capture=False)
        if r2.returncode == 0:
            run(["git", "push", "shuangwaiwai", sym + ":master"], cwd=D_ROOT, capture=False)
        else:
            run(["git", "am", "--abort"], cwd=D_ROOT, capture=False)
            print("[git] patch conflict — left remote untouched", file=sys.stderr)
        run(["git", "checkout", cur_branch], cwd=D_ROOT, capture=False)
        try:
            os.remove(patch)
        except Exception:
            pass
    print("[DONE]", file=sys.stderr)

if __name__ == "__main__":
    main()
