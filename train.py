from hyper import CaptchaType, Hyper

captcha_type = CaptchaType.SUPREME_COURT
epochs = 100
earlystopping = True
early_stopping_patience = 8
save_weights = True
save_model = True

Hyper(captcha_type).train_model(
    epochs=100,
    earlystopping=earlystopping,
    early_stopping_patience=early_stopping_patience,
    save_weights=save_weights,
    save_model=save_model)
