"""Extract original workbook images and reconcile every drawing occurrence.
No credentials, network calls, image re-encoding, or automatic packaging deletion.
Usage: python extract_workbook.py input.xlsx output_directory
Requires Pillow for decoding validation; original bytes remain unchanged.
"""
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from collections import Counter
import argparse, csv, hashlib, json, posixpath
import xml.etree.ElementTree as ET
from PIL import Image

NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
def sha(data): return hashlib.sha256(data).hexdigest()
def column(n):
    result=''
    while n: n,k=divmod(n-1,26);result=chr(65+k)+result
    return result
def resolve(part,target):
    return posixpath.normpath(posixpath.join(posixpath.dirname(part),target)).lstrip('/')
def relations(z,part):
    p=PurePosixPath(part);name=str(p.parent/'_rels'/(p.name+'.rels'))
    if name not in z.namelist(): return {}
    return {x.get('Id'):x for x in ET.fromstring(z.read(name))}
def extract(source,out):
    out.mkdir(parents=True,exist_ok=True);originals=out/'originals';originals.mkdir(exist_ok=True)
    inventory=[];occurrences=[];missing=[]
    source_hash=hashlib.file_digest(source.open('rb'),'sha256').hexdigest()
    with ZipFile(source) as z:
        bad=z.testzip()
        if bad: raise ValueError('ZIP CRC failed: '+bad)
        media=[n for n in z.namelist() if n.startswith('xl/media/') and not n.endswith('/')]
        for name in sorted(media):
            data=z.read(name);dest=originals/Path(name).name
            if dest.exists() and sha(dest.read_bytes())!=sha(data):raise ValueError('Output collision: '+str(dest))
            dest.write_bytes(data)
            with Image.open(dest) as image:
                width,height=image.size;fmt=image.format;image.verify()
            if sha(dest.read_bytes())!=sha(data):raise ValueError('Byte verification failed: '+name)
            inventory.append({'archive_path':name,'file':'originals/'+dest.name,'bytes':len(data),
                              'sha256':sha(data),'width':width,'height':height,'format':fmt,
                              'decision':'unreviewed','category':'unreviewed'})
        media_set=set(media);wb='xl/workbook.xml';wr=relations(z,wb);sheets=[]
        for sheet in ET.fromstring(z.read(wb)).findall('s:sheets/s:sheet',NS):
            title=sheet.get('name');sheets.append(title)
            part=resolve(wb,wr[sheet.get('{'+NS['r']+'}id')].get('Target'))
            sr=relations(z,part)
            for drawing in ET.fromstring(z.read(part)).findall('s:drawing',NS):
                dp=resolve(part,sr[drawing.get('{'+NS['r']+'}id')].get('Target'));dr=relations(z,dp)
                for idx,anchor in enumerate(ET.fromstring(z.read(dp)),1):
                    start=anchor.find('{*}from');pic=anchor.find('{*}pic')
                    if pic is None:continue
                    row=int(start.find('{*}row').text)+1 if start is not None else None
                    col=int(start.find('{*}col').text)+1 if start is not None else None
                    base={'sheet':title,'drawing':dp,'anchor':idx,'row':row,'column':col,
                          'cell':column(col)+str(row) if row and col else None}
                    blip=pic.find('.//a:blip',NS)
                    rid=blip.get('{'+NS['r']+'}embed') if blip is not None else None
                    rel=dr.get(rid)
                    if rel is None or rel.get('TargetMode')=='External':
                        missing.append({**base,'reason':'No embedded image relationship'});continue
                    target=resolve(dp,rel.get('Target'))
                    if target not in media_set:missing.append({**base,'reason':'Missing embedded target','target':target});continue
                    occurrences.append({**base,'archive_path':target,'file':'originals/'+Path(target).name})
        counts=Counter(x['archive_path'] for x in occurrences)
        for item in inventory:item['occurrences']=counts[item['archive_path']]
        summary={'source_filename':source.name,'source_bytes':source.stat().st_size,'source_sha256':source_hash,
                 'sheets':sheets,'embedded_media_files':len(inventory),'unique_image_hashes':len({x['sha256'] for x in inventory}),
                 'drawing_occurrences':len(occurrences),'missing_picture_positions':len(missing),
                 'unreferenced_media_files':sum(x['occurrences']==0 for x in inventory),
                 'verified_images':len(inventory),'zip_crc':'PASS','byte_hash_check':'PASS',
                 'scope':'This saved workbook only; not proof of coverage of the original online workbook.',
                 'filtering_status':'Not classified or filtered. All originals preserved.'}
    for name,data in [('summary',summary),('inventory',inventory),('occurrences',occurrences),('missing',missing)]:
        (out/(name+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    for name,data in [('inventory',inventory),('occurrences',occurrences)]:
        with (out/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]) if data else []);w.writeheader();w.writerows(data)
    print(json.dumps(summary,ensure_ascii=True))
    if missing:raise SystemExit('Extraction incomplete; inspect missing.json')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();extract(a.source,a.output)
