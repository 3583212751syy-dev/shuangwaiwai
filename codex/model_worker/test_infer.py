import requests
import base64

URL = 'http://localhost:7860/infer_enhanced'
with open('../tests/sample.jpg', 'rb') as f:
    files = {'image': ('sample.jpg', f, 'image/jpeg')}
    resp = requests.post(URL, files=files, data={'variants': 2})
    print('status', resp.status_code)
    print(resp.text)
    j = resp.json()
    if j.get('success'):
        for out in j['outputs']:
            b64 = out.get('b64')
            if b64:
                data = base64.b64decode(b64)
                fname = out.get('filename')
                with open(fname, 'wb') as o:
                    o.write(data)
                print('wrote', fname)
