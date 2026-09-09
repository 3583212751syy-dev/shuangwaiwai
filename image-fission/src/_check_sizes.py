from PIL import Image
for p in ['jobs/v319v_bat_style/source_bacardi.jpg', 'jobs/v319v_eagle/source.jpg', 'jobs/v319v_denim/source.jpg']:
    img = Image.open('E:/Desktop/双接口/image-fission/' + p)
    print(f'{p}: {img.size}')
