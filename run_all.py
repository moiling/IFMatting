import os
import time

from IFM.ifm import information_flow_matting
from utils.utils import save_image, show_image

if __name__ == '__main__':
    input_dir = './data/input_lowres/'
    trimap_dir = './data/trimap_lowres/'
    out_dir = './out/demo_no_trim/'
    cost_time = {}

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            for trimap_sub_dir in ['Trimap1', 'Trimap2', 'Trimap3']:
                print('\n' + file + ': ' + trimap_sub_dir)
                time_start = time.time()
                # matting
                alpha_matte = information_flow_matting(input_dir + file,
                                                       trimap_dir + trimap_sub_dir + '/' + file,
                                                       (file != 'net.png' and file != 'plasticbag.png'))  # 人工区分高透明度
                time_end = time.time()
                print('cost: {:.2f}s'.format(time_end - time_start))
                cost_time[file + ',' + trimap_sub_dir] = time_end - time_start
                # save
                save_image(alpha_matte, out_dir + trimap_sub_dir + '/', file, True)

    print(cost_time)
    # show
    # show_image(alpha_matte)
