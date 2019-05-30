from IFM.ifm import information_flow_matting
from utils import save_image, show_image

if __name__ == '__main__':

    file_name = 'pineapple.png'

    # matting
    alpha_matte = information_flow_matting('./data/input_lowres/' + file_name,
                                           './data/trimap_lowres/Trimap1/' + file_name,
                                           True)
    # save
    save_image(alpha_matte, './out/test/', file_name)

    # show
    show_image(alpha_matte)

