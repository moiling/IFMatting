from utils.foreground_background import estimate_foreground_background
from utils.utils import save_image, show_image, stack_alpha
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_name = 'troll.png'
    alpha_sub_dir = 'demo/Trimap1/'

    alpha_dir = './out/'
    image_dir = './data/input_lowres/'
    out_dir = './out/cut_out/'

    image = plt.imread(image_dir + file_name)
    alpha = plt.imread(alpha_dir + alpha_sub_dir + file_name)

    foreground, background = estimate_foreground_background(image, alpha, print_info=True)

    # Make new image from foreground and alpha
    cutout = stack_alpha(foreground, alpha)

    # save
    save_image(cutout, out_dir + alpha_sub_dir, file_name)

    # show
    # show_image(cutout)
