from PIL import Image
import requests
from io import BytesIO

def imageLoadFromUrl(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def setBGColor(img, fill_color = (255,255,255)):
    bg_white = Image.new("RGBA", img.size, fill_color)
    bg_white.paste(img, (0, 0), img)
    bg_white = bg_white.convert("L")
    return bg_white

def getSCourtImage(img_path):
    img = Image.open(img_path)
    img = setBGColor(img)
    return img
