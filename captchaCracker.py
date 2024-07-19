import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import time
from hyper import CaptchaType, Hyper

argv = sys.argv
exec = os.path.basename(argv[0])

# captcha_type = CaptchaType.GOV24
# weights_only = True
# image_path = "C:\\python\\captchaCracker\\images\\gov24\\pred\\716047.png"
# hyper = Hyper(captcha_type=captcha_type, weights_only=weights_only, quiet_out=True)
# pred = hyper.predict(image_path)
# hyper.quiet(False)
# print(pred)
# sys.exit(0)

if len(sys.argv) < 3:
    print("Usage: " + exec + " supreme_court|gov24|nh_web_mail IMAGE_PATH")
    sys.exit(-1)

captcha_type = CaptchaType(argv[1])
weights_only = True
image_path = argv[2]

if("__main__" == __name__):
    hyper = Hyper(captcha_type=captcha_type, weights_only=weights_only, quiet_out=True)

    # from PIL import Image
    # img = Image.open(image_path)

    # fill_color = (255,255,255)  # your new background color
    # img = img.convert("RGBA")   # it had mode P after DL it from OP
    # if img.mode in ('RGBA', 'LA'):
    #     background = Image.new(img.mode[:-1], img.size, fill_color)
    #     background.paste(img, img.split()[-1]) # omit transparency
    #     img = background

    # new_image_path = "./whitebg.png"
    # img.save(new_image_path)

    temp_image_path = hyper.setBG(image_path)
    pred = hyper.predict(temp_image_path)
    hyper.quiet(False)
    print(pred)

    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)

    sys.exit(0)

else:
    print("module imported")
