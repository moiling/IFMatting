import os

from IFM.ifm import information_flow_matting
from utils.utils import save_image, show_image

if __name__ == '__main__':

    # file_name = 'pineapple.png'

    for root, dirs, files in os.walk(r"./data/input_lowres/"):
        for file in files:
            # 获取文件路径
            print(file)

            for trimap_dir in ['Trimap1', 'Trimap2', 'Trimap3']:
                print(trimap_dir)
                # matting
                alpha_matte = information_flow_matting('./data/input_lowres/' + file,
                                                       './data/trimap_lowres/' + trimap_dir + '/' + file,
                                                       (file != 'net.png' and file != 'plasticbag.png'))  # 人工区分高透明度
                # save
                save_image(alpha_matte, './out/demo/' + trimap_dir + '/', file)

    # show
    # show_image(alpha_matte)

