from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, Job, JobQueue
import albumentations as A
import nibabel as nib
import numpy as np
import random
import onnxruntime
import torch
from google_drive_downloader import GoogleDriveDownloader as gdd
import os
from PIL import Image
from sys import argv

img_size = 256

transform = A.Compose([
        A.Resize(img_size, img_size),
    ])


def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Привет! Я помогу тебе узнать процент области и процент заражения легких от COVID-19. Для получения результата отправь изображение или файл в формате .nii.gz')


def text_handler(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Привет! Я помогу тебе узнать процент области и процент заражения легких от COVID-19. Для получения результата отправь изображение или файл в формате .nii.gz')


def get_niigz(path: str):
    img = nib.load(path)
    a = np.array(img.dataobj)
    a = np.interp(a, (-2048, 2048), (0, 1))
    return a[:,:,0:32]


def get_percent(img):
    ort_session = onnxruntime.InferenceSession("full_connected.onnx")
    ort_inputs = {'input': img.detach().cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    pred = ort_outs[0]
    return pred


def get_regions(img) -> str:
    ort_session = onnxruntime.InferenceSession("super_resolution.onnx")
    ort_inputs = {'input': img.detach().cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    pred = np.squeeze(ort_outs[0])

    min = np.min(pred)
    max = np.max(pred)

    pred -= min
    pred = pred / (max - min) * 255

    data = Image.fromarray(np.uint8(pred))
    filename = 'result' + str(random.randint(1, 1000000)) + '.jpeg'
    data.save(filename)

    return filename


def image_handler(update: Update, context: CallbackContext) -> None:
    file = context.bot.getFile(update.message.photo[-1].file_id)
    niigz_file = context.bot.get_file(file)

    update.message.reply_text('Файл получен, ожидайте результат...')

    filename = 'data' + str(random.randint(1, 1000000)) + '.nii.gz'
    niigz_file.download(filename)

    image = np.array(Image.open(filename))
    image = np.mean(image, axis=2)
    image = np.array([image for _ in range(32)])
    image = np.moveaxis(image, 0, 1)
    image = np.moveaxis(image, 1, 2)
    image = np.interp(image, (0, 256), (0, 1))

    transformed = transform(image=image)
    image = np.moveaxis(transformed['image'], 2, 0)
    image = np.expand_dims(image, axis=0)

    result_path = get_regions(torch.Tensor(image))

    update.message.reply_text('Процент заражения ' + str(int(get_percent(torch.Tensor(image)) * 25)) + '%')
    context.bot.send_photo(update.message.chat_id, open(result_path, 'rb'))

    os.remove(filename)
    os.remove(result_path)


def document_handler(update: Update, context: CallbackContext) -> None:
    file = context.bot.getFile(update.message.document.file_id)
    niigz_file = context.bot.get_file(file)

    update.message.reply_text('Файл получен, ожидайте результат...')

    filename = 'data' + str(random.randint(1, 1000000)) + '.nii.gz'
    niigz_file.download(filename)

    image = get_niigz(filename)
    transformed = transform(image=image)
    image = np.moveaxis(transformed['image'], 2, 0)
    image = np.expand_dims(image, axis = 0)

    result_path = get_regions(torch.Tensor(image))

    update.message.reply_text('Процент заражения ' + str(int(get_percent(torch.Tensor(image)) * 25)) + '%')
    context.bot.send_photo(update.message.chat_id, open(result_path, 'rb'))

    os.remove(filename)
    os.remove(result_path)


if __name__ == '__main__':
    if not os.path.exists('full_connected.onnx'):
        gdd.download_file_from_google_drive(file_id='12vfGKEKHO4gTUDbt_t2Fz7WOkWx2ljme',
                                            dest_path='./full_connected.onnx',
                                            unzip=False)
        gdd.download_file_from_google_drive(file_id='1eccY-xkPRFHiXkxV2_BkwM1-TqeVWeov',
                                            dest_path='./super_resolution.onnx',
                                            unzip=False)
        print('hey')
    name, token = argv
    updater = Updater(token)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler('start', start))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, text_handler))
    dispatcher.add_handler(MessageHandler(Filters.photo, image_handler))
    dispatcher.add_handler(MessageHandler(Filters.document, document_handler))

    updater.start_polling()
    updater.idle()
