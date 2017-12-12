import os
from multiprocessing import Process
from urllib.request import Request, urlopen, urlretrieve

from numpy import random


def download_image(p_id, images_dir, print_file_names, images_amount):
    while True:
        try:
            if len(os.listdir(images_dir)) >= images_amount:
                break

            # Randomize the URL so that we don't get cached URL redirects
            req = Request('https://source.unsplash.com/random?sig=%s' % random.randint(1, 30000))
            res = urlopen(req)
            url = res.geturl()
            file_name = url.split('/')[-1].split('#')[0].split('?')[0] + '.jpeg'

            file_path = os.path.join(images_dir, file_name)

            if not os.path.exists(file_path):
                if print_file_names:
                    print('%s is downloading %s' % (p_id + 1, file_name))

                urlretrieve(url, file_path)
        except:
            pass


def download_random_unsplash_images(images_dir, parallel_processes_amount=1, print_file_names=False,
                                    images_amount=1):
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    for p_id in range(parallel_processes_amount):
        p = Process(target=download_image, args=(p_id, images_dir, print_file_names, images_amount))
        p.start()


if __name__ == '__main__':
    images_dir = 'images'

    download_random_unsplash_images(images_dir=images_dir,
                                    parallel_processes_amount=5,
                                    images_amount=30000)
