from frechet_kernel_Inception_distance import *
from inception_score import *
from glob import glob
import os

def inception_score(fake_path='./fake', batch_size=1) :
    filenames = glob(os.path.join(fake_path, '*.*'))
    images = [get_images(filename) for filename in filenames]
    images = np.transpose(images, axes=[0, 3, 1, 2])

    # A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
    BATCH_SIZE = batch_size

    # Run images through Inception.
    inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])

    logits = inception_logits(inception_images)

    IS = get_inception_score(BATCH_SIZE, images, inception_images, logits, splits=10)
    print ('is finish:', IS)
    return IS


def frechet_inception_distance(real_target_path='./real_target', fake_path='./fake') :
    filenames = glob(os.path.join(real_target_path, '*.*'))
    real_images = [get_images(filename) for filename in filenames]
    real_images = np.transpose(real_images, axes=[0, 3, 1, 2])

    filenames = glob(os.path.join(fake_path, '*.*'))
    fake_images = [get_images(filename) for filename in filenames]
    fake_images = np.transpose(fake_images, axes=[0, 3, 1, 2])

    # A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
    BATCH_SIZE = 1

    # Run images through Inception.
    inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])
    real_activation = tf.placeholder(tf.float32, [None, None], name='activations1')
    fake_activation = tf.placeholder(tf.float32, [None, None], name='activations2')

    fcd = frechet_classifier_distance_from_activations(real_activation, fake_activation)
    activations = inception_activations(inception_images)

    FID = get_fid(fcd, BATCH_SIZE, real_images, fake_images, inception_images, real_activation, fake_activation, activations)

    print('fid finish')
    print("FID : ", FID / 100)

def kernel_inception_distance(real_target_path='./real_target', fake_path='./fake', batch_size=1) :
    filenames = glob(os.path.join(real_target_path, '*.*'))
    real_images = [get_images(filename) for filename in filenames]
    real_images = np.transpose(real_images, axes=[0, 3, 1, 2])

    filenames = glob(os.path.join(fake_path, '*.*'))
    fake_images = [get_images(filename) for filename in filenames]
    fake_images = np.transpose(fake_images, axes=[0, 3, 1, 2])

    # A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
    BATCH_SIZE = batch_size

    # Run images through Inception.
    inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])
    real_activation = tf.placeholder(tf.float32, [None, None], name='activations1')
    fake_activation = tf.placeholder(tf.float32, [None, None], name='activations2')

    kcd_mean, kcd_stddev = kernel_classifier_distance_and_std_from_activations(real_activation, fake_activation, max_block_size=500)
    activations = inception_activations(inception_images)

    KID_mean = get_kid(kcd_mean, BATCH_SIZE, real_images, fake_images, inception_images, real_activation, fake_activation, activations)
    KID_stddev = get_kid(kcd_stddev, BATCH_SIZE, real_images, fake_images, inception_images, real_activation, fake_activation, activations)

    print('kid finish:', KID_mean * 100, KID_stddev * 100)

    return KID_mean * 100, KID_stddev * 100

def mean_kernel_inception_distance(real_source_path='./real_source', real_target_path='./real_target', fake_path='./fake') :
    source_alpha = 0.98
    target_alpha = 1 - source_alpha
    
    filenames = glob(os.path.join('./real_source', '*.*'))
    real_source_images = [get_images(filename) for filename in filenames]
    real_source_images = np.transpose(real_source_images, axes=[0, 3, 1, 2])

    filenames = glob(os.path.join('./real_target', '*.*'))
    real_target_images = [get_images(filename) for filename in filenames]
    real_target_images = np.transpose(real_target_images, axes=[0, 3, 1, 2])

    filenames = glob(os.path.join('./fake', '*.*'))
    fake_images = [get_images(filename) for filename in filenames]
    fake_images = np.transpose(fake_images, axes=[0, 3, 1, 2])

    # A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
    BATCH_SIZE = 1

    # Run images through Inception.
    inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])
    real_activation = tf.placeholder(tf.float32, [None, None], name='activations1')
    fake_activation = tf.placeholder(tf.float32, [None, None], name='activations2')

    fcd = frechet_classifier_distance_from_activations(real_activation, fake_activation)
    kcd_mean, kcd_stddev = kernel_classifier_distance_and_std_from_activations(real_activation, fake_activation,
                                                                               max_block_size=500)
    activations = inception_activations(inception_images)

    FID = get_fid(fcd, BATCH_SIZE, real_target_images, fake_images, inception_images, real_activation, fake_activation, activations)
    KID_mean = get_kid(kcd_mean, BATCH_SIZE, real_target_images, fake_images, inception_images, real_activation, fake_activation, activations)
    KID_stddev = get_kid(kcd_stddev, BATCH_SIZE, real_target_images, fake_images, inception_images, real_activation, fake_activation, activations)

    mean_FID = get_fid(fcd, BATCH_SIZE, real_source_images, fake_images, inception_images, real_activation, fake_activation, activations)
    mean_KID_mean = get_kid(kcd_mean, BATCH_SIZE, real_source_images, fake_images, inception_images, real_activation, fake_activation, activations)
    mean_KID_stddev = get_kid(kcd_stddev, BATCH_SIZE, real_source_images, fake_images, inception_images, real_activation, fake_activation, activations)

    mean_FID = (target_alpha * FID + source_alpha * mean_FID) / 2.0
    mean_KID_mean = (target_alpha * KID_mean + source_alpha * mean_KID_mean) / 2.0
    mean_KID_stddev = (target_alpha * KID_stddev + source_alpha * mean_KID_stddev) / 2.0

    # mean_FID = (2 * FID * mean_FID) / (FID + mean_FID)
    # mean_KID_mean = (2 * KID_mean * mean_KID_mean) / (KID_mean + mean_KID_mean)
    # mean_KID_stddev = (2 * KID_stddev * mean_KID_stddev) / (KID_stddev + mean_KID_stddev)

    print()

    print("mean_FID : ", mean_FID / 100)
    print("mean_KID_mean : ", mean_KID_mean * 100)
    print("mean_KID_stddev : ", mean_KID_stddev * 100)

def kid_kid(target_path, fake_path, batch_size):
    return kernel_inception_distance(target_path, fake_path, batch_size)

def kid_is(fake_path, batch_size):
    return inception_score(fake_path, batch_size)

if __name__ == "__main__":
    inception_score()
    frechet_inception_distance()
    kernel_inception_distance()
    # mean_kernel_inception_distance()
