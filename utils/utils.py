import numpy as np
import matplotlib.pyplot as plt
import torchvision
import os

def imshow_grid(img):
    img = torchvision.utils.make_grid(img.cpu().detach())
    img_numpy = img.numpy()
    print(img_numpy.shape)
    plt.figure(figsize=(10, 20))
    plt.imshow(np.transpose(img_numpy, (1, 2, 0)))
    plt.show()

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]

def compare_images(real_img, generated_img, i, score, reverse=False, threshold=50):
    real_img = np.transpose(real_img.cpu().data.numpy().squeeze(), (1, 2, 0))
    real_img = real_img.reshape(300, 300, 3)
    generated_img = np.transpose(generated_img.cpu().data.numpy().squeeze(), (1, 2, 0))
    generated_img = generated_img.reshape(300, 300, 3)

    negative = np.zeros_like(real_img)

    if not reverse:
        diff_img = real_img - generated_img
    else:
        diff_img = generated_img - real_img

    diff_img[diff_img <= threshold] = 0

    anomaly_img = [np.zeros(shape=(300, 300, 3)), np.zeros(shape=(300, 300, 3)), np.zeros(shape=(300, 300, 3))]
    anomaly_img[0] = (real_img - diff_img) * 255
    anomaly_img[1] = (real_img - diff_img) * 255
    anomaly_img[2] = (real_img - diff_img) * 255
    anomaly_img[0] = anomaly_img[0] + diff_img

    anomaly_img = [anomaly_img[0].astype(np.uint8), anomaly_img[1].astype(np.uint8), anomaly_img[2].astype(np.uint8)]

    fig, plots = plt.subplots(1, 4)

    fig.suptitle(f'Anomaly - (anomaly score: {score:.4})')
    
    fig.set_figwidth(20)
    fig.set_tight_layout(True)
    plots = plots.reshape(-1)
    plots[0].imshow(real_img, cmap='bone', label='real')
    plots[1].imshow(generated_img, cmap='bone')
    plots[2].imshow(diff_img, cmap='bone')
    plots[3].imshow(anomaly_img[0], cmap='bone')

    
    plots[0].set_title('real')
    plots[1].set_title('generated')
    plots[2].set_title('difference')
    plots[3].set_title('Anomaly Detection')
    plt.show()