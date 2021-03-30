import os
import json
import argparse
import requests
from PIL import Image
from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL


"""
ref: https://github.com/johancc/ImageNetDownloader/blob/master/downloader.py

in cmd: if the shell use wordsplitting that makes each word in your command invocation a separate word,
use double quotation marks instead of single quotation marks
"""


def get_url(wnid):
    return f'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={wnid}'


def get_image(img_url, class_folder):
    if len(img_url) <= 1:
        print("Try another valid URL!")
        return

    try:
        img_resp = requests.get(img_url, timeout=1)
    except (ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL) as e:
        print(e)
        # img_resp = 'NO RESPONSE'
        return

    if not 'content-type' in img_resp.headers:
        print("MISSING CONTENT")
        return
    if not 'image' in img_resp.headers['content-type']:
        print("The url doesn't have any image!")
        return
    if len(img_resp.content) < 1000:
        return

    img_name = img_url.split('/')[-1]
    img_name = img_name.split('?')[0]

    if len(img_name) <= 1:
        print("MISSING Image Name!")
        return
    if not 'flickr' in img_url:
        print("Missing non-flickr images!")
        return

    img_file_path = os.path.join(class_folder, img_name)

    with open(img_file_path, 'wb') as img_f:
        img_f.write(img_resp.content)

    im = Image.open(img_file_path)

    if im.mode != 'RGB':
        im = im.convert(mode='RGB')

    im_resized = im.resize((64, 64), Image.BOX)
    im_resized.save(img_file_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='HW02 Task1')
    parser.add_argument('--subclass_list', nargs='*', type=str, required=True)
    parser.add_argument('--images_per_subclass', type=int, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--main_class', type=str, required=True)
    parser.add_argument('--imagenet_info_json', type=str, required=True)
    args, args_other = parser.parse_known_args()

    current_folder = os.path.dirname(os.path.realpath(__file__))

    class_info_json_filename = 'imagenet_class_info.json'
    class_info_json_filepath = os.path.join(current_folder, class_info_json_filename)

    with open(class_info_json_filepath) as f:
        imagenet_class_info = json.load(f)

    # add \Train folder
    if not os.path.isdir(args.data_root):
        os.mkdir(args.data_root)

    # temp_class_name = args.main_class[1:-1]
    # add class_name (eg. cat/dog/...) to Train/ folder
    class_images_folder = os.path.join(args.data_root, args.main_class[1:-1])
    if not os.path.isdir(class_images_folder):
        os.mkdir(class_images_folder)

    for id, attr in imagenet_class_info.items():
        # print(attr['class_name'])
        temp_subclass_name = attr['class_name']

        if temp_subclass_name in args.subclass_list:

            class_folder = os.path.join(class_images_folder, temp_subclass_name)
            if not os.path.exists(class_folder):
                os.mkdir(class_folder)

            url_urls = get_url(id)
            resp = requests.get(url_urls)

            urls = [url.decode('utf-8') for url in resp.content.splitlines()]
            for url in urls:
                # check number of files in each subclass is 200
                if len(os.walk(class_folder).__next__()[2]) < args.images_per_subclass:
                    get_image(url, class_folder)
                else:
                    break
        else:
            pass
