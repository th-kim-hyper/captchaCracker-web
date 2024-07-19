from hyper import CaptchaType, Hyper

captcha_type = CaptchaType.SUPREME_COURT
weights_only = True

Hyper(captcha_type, weights_only).validate_model()
