from IFM.ifm import information_flow_matting
from utils.utils import save_image, show_image

if __name__ == '__main__':

    file_name = 'troll.png'
    trimap_dir = 'Trimap1'

    # matting
    alpha_matte = information_flow_matting('./data/input_lowres/' + file_name,
                                           './data/trimap_lowres/' + trimap_dir + '/' + file_name,
                                           (file_name != 'net.png' and file_name != 'plasticbag.png'))  # 人工区分高透明度
    # save
    save_image(alpha_matte, './out/test/' + trimap_dir + '/', file_name)

    # show
    show_image(alpha_matte)
