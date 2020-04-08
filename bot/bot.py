from telegram import Bot
from telegram import Update
from telegram.ext import MessageHandler, Updater, Filters
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf

# this is unsafe!
TOKEN = '1166179420:AAF8yULcvGCv-XS4-djsJ8yVlbpNFb1prL0'

CROP_SIZE_H = 32
CROP_SIZE_W = 32

cnn_model = None


def load_image_into_numpy( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="float64" )
    return data


def ping(bot, update):
    return 'pong'


def resize():
    img = Image.open('input_photo.jpg')
    img = img.resize((CROP_SIZE_W, CROP_SIZE_H), Image.ANTIALIAS)
    img.save('output_photo.jpg')


def predict():
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255
    )
    my_img = load_image_into_numpy('output_photo.jpg')
    i = np.expand_dims(my_img, 0)
    predicted = cnn_model.predict(image_generator.flow(i))
    return (predicted.argmax(), predicted.max())


def message_handler(bot: Bot, update: Update):
    user = update.effective_user
    repost = update.message
    print(repost.photo)
    photo = bot.getFile(update.message.photo[-1].file_id)
    photo.download('input_photo.jpg')

    resize()
    number, percentage = predict()

    # bot.send_photo(
    #     chat_id=update.effective_message.chat_id,
    #     photo=open('../dl_4/output_photo.jpg', 'rb')
    # )

    percents = round(percentage * 100, 4)

    message = "With {0}% it's number - {1}".format(percents, number)
    bot.send_message(
        chat_id=update.effective_message.chat_id,
        text=message
    )


def main():
    bot = Bot(
        token=TOKEN
    )
    updater = Updater(
        bot=bot
    )

    handler = MessageHandler(Filters.all, message_handler)
    updater.dispatcher.add_handler(handler)

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    cnn_model = load_model('../dl_4/model.h5')
    cnn_model.summary()
    main()
