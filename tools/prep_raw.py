# 批量把原图 resize 成 256x256 RGB raw，写入单一二进制 + 索引，供 node 快速读取
import json,os,sys
from PIL import Image
W=r'D:/GPT/2026-11-21/new-chat-2/work'
meta=json.load(open(W+'/deploy-product-library/metadata.json',encoding='utf-8'))
orig=W+'/full-extraction/originals'
out=W+'/_all_raw.bin';idx={}
SIZE=256
with open(out,'wb') as f:
 for m in meta:
  fn=m['filename']
  try:
   im=Image.open(os.path.join(orig,fn)).convert('RGB').resize((SIZE,SIZE),Image.LANCZOS)
   idx[fn]={'off':f.tell(),'len':SIZE*SIZE*3}
   f.write(im.tobytes())
  except Exception as e:
   idx[fn]={'error':str(e)}
json.dump(idx,open(W+'/_all_raw_idx.json','w'))
print('prepared',len(idx),'bytes',os.path.getsize(out))
