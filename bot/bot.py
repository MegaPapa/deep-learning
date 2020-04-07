from telegram import Bot
from telegram import Update
from telegram.ext import MessageHandler, Updater, Filters
from PIL import Image

# this is unsafe!
TOKEN = '1166179420:AAF8yULcvGCv-XS4-djsJ8yVlbpNFb1prL0'

CROP_SIZE_H = 32
CROP_SIZE_W = 32


def ping(bot, update):
    return 'pong'


def message_handler(bot: Bot, update: Update):
    user = update.effective_user
    repost = update.message
    print(repost.photo)
    photo = bot.getFile(update.message.photo[-1].file_id)
    photo.download('input_photo.jpg')

    img = Image.open('input_photo.jpg')
    img = img.resize((CROP_SIZE_W, CROP_SIZE_H), Image.ANTIALIAS)
    img.save('output_photo.jpg')

    bot.send_photo(
        chat_id=update.effective_message.chat_id,
        photo=open('output_photo.jpg', 'rb')
    )

    message = 'Hello ' + user.first_name
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
    main()
