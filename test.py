from IFM.ifm import information_flow_matting
from utils.utils import save_image, show_image
import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':

    file_name = 'plasticbag.png'
    trimap_dir = 'Trimap1'

    # matting
    alpha_matte = information_flow_matting('./data/input_lowres/' + file_name,
                                           './data/trimap_lowres/' + trimap_dir + '/' + file_name,
                                           (file_name != 'net.png' and file_name != 'plasticbag.png'))  # 人工区分高透明度
    # save
    save_image(alpha_matte, './out/test/' + trimap_dir + '/', file_name, True)
    alpha_matte_3d = (alpha_matte.reshape(alpha_matte.shape[0], alpha_matte.shape[1], 1)
                      * np.asarray([255, 255, 255]).T).astype(np.int)
    alpha_matte_2d = alpha_matte_3d[:, :, 0]
    np.savetxt('./out/test/' + trimap_dir + '/' + file_name + '.txt', alpha_matte_2d)

    # read again
    image = plt.imread('./out/test/' + trimap_dir + '/' + file_name)
    np.savetxt('./out/test/' + trimap_dir + '/' + file_name + 'again.txt', (image * 255).astype(int))

    error = (image * 255).astype(int) - alpha_matte_2d
    error = error[error != 0]
    print(error.size)
    # show
    # show_image(alpha_matte)
