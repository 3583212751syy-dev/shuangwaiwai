import re, os
path = r"E:\Desktop\茶叶\参考链接.xls"
data = open(path, "rb").read()
print("FILE SIZE:", len(data))
# 1) extract URLs
urls = re.findall(rb'https?://[^\x00-\x1f\s"\'<>]+', data)
seen = set()
print("=== URLs found ===")
for u in urls:
    s = u.decode('latin-1', 'ignore')
    if s not in seen:
        seen.add(s)
        print(s)
# 2) extract readable text (utf-16le + latin1)
print("=== readable strings (utf16) ===")
try:
    txt = data.decode('utf-16-le', 'ignore')
    strings = re.findall(r'[\u4e00-\u9fffA-Za-z0-9\-_./:]{3,}', txt)
    out = []
    for s in strings:
        if s not in out:
            out.append(s)
    print(" | ".join(out[:300]))
except Exception as e:
    print("utf16 err", e)
print("=== readable strings (latin1) ===")
try:
    txt2 = data.decode('latin-1', 'ignore')
    strings2 = re.findall(r'[\u4e00-\u9fffA-Za-z0-9\-_./:@]{3,}', txt2)
    out2 = []
    for s in strings2:
        if s not in out2:
            out2.append(s)
    print(" | ".join(out2[:300]))
except Exception as e:
    print("latin1 err", e)
