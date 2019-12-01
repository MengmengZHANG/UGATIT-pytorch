from PIL import Image
import os, sys


def resize(source_dir, target_dir):
    dirs = os.listdir( source_dir )
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for item in dirs:
        if os.path.isfile(os.path.join(source_dir,item)):
            im = Image.open(os.path.join(source_dir,item))
            imResize = im.resize((64,64), Image.ANTIALIAS)
            print (os.path.join(target_dir,item))
            imResize.save(os.path.join(target_dir,item), 'JPEG', quality=90)

resize('../dataset/selfie2anime/testA', '../dataset/selfie2anime_64_64/testA')
resize('../dataset/selfie2anime/testB', '../dataset/selfie2anime_64_64/testB')
resize('../dataset/selfie2anime/trainA', '../dataset/selfie2anime_64_64/trainA')
resize('../dataset/selfie2anime/trainB', '../dataset/selfie2anime_64_64/trainB')