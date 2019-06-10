import time

from IFM.ifm import information_flow_matting
from utils.utils import save_image, show_image

if __name__ == '__main__':
    file_name = 'pineapple.png'
    trimap_sub_dir = 'Trimap1'

    input_dir = './data/input_lowres/'
    trimap_dir = './data/trimap_lowres/'
    out_dir = './out/test/'

    time_start = time.time()
    # matting
    alpha_matte = information_flow_matting(input_dir + file_name,
                                           trimap_dir + trimap_sub_dir + '/' + file_name,
                                           (file_name != 'net.png' and file_name != 'plasticbag.png'))  # 人工区分高透明度
    time_end = time.time()
    print('cost: {:.2f}s'.format(time_end - time_start))
    # save
    save_image(alpha_matte, out_dir + trimap_sub_dir + '/', file_name, True)

    # show
    # show_image(alpha_matte)
