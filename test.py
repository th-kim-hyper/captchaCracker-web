from PIL import Image
import ImageProcess as imgp
import time
from io import BytesIO
import aiofiles

def getData():
    img = imgp.imageLoadFromUrl("https://safind.scourt.go.kr/sf/captchaImg?t=image")
    img = imgp.setBGColor(img)
    img = img.crop((1, 1, img.width - 1, 50))
    time_stamp = int(time.time())
    save_path = f"download/supreme_court/{time_stamp}.png"
    img.save(save_path)
    img.close()
    img = None

for i in range(20):
    print(f"#### Start >> {i+1}/20 ####")
    getData()
    print(f"#### << Done {i+1}/20 ####")
    time.sleep(1)
