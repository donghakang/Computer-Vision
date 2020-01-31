import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_differential_filter():
    # horizontal differential filter
    filter_x = np.array([-1,0,1,-1,0,1,-1,0,1]).reshape(3,3)

    ## shift right
    # filter_x = np.array([0,0,0,1,0,0,0,0,0]).reshape(3,3)

    ## sharpen
    # filter_x = np.array([1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9]).reshape(3,3)
    # filter_y = np.array([0,0,0,0,2,0,0,0,0]).reshape(3,3)
    # filter_x = np.subtract(filter_y,filter_x)


    # vertical differential filter
    filter_y = np.array([-1,-1,-1,0,0,0,1,1,1]).reshape(3,3)

    return filter_x, filter_y


########################################################################
###########################    HELPERS     #############################
########################################################################
# change the values and images to make it visual
def image_processing (im):
    im = np.where(im < 0, 0, im)
    im = np.where(im > 255, 255, im)
    im = im.astype(np.uint8)
    return im

def normalize_angle(theta):
    if theta > np.pi:
        return theta
    else:
        return theta

# multiply 3x3 kernel and 3x3 filter to get a specific number for a pixel
def calculate_filter(filter, kernel):
    pixel_value = np.sum(np.multiply(filter, kernel))
    return pixel_value

def histo_normalization(block, block_size):
    e = 0.001       # prevent divison by 0.
    x,y,z = np.shape(block)
    division = np.sum(block ** 2 + e ** 2, axis = 2) # block_size x block_size
    for i in range (0, block_size):
        for j in range (0, block_size):
            block[i,j,:] = block[i,j,:]/division[i,j]

    # print(normal_block)
    return block.reshape(x*y*z)








########################################################################
#####################    GIVEN CODE : MODIFIED     #####################
########################################################################
def filter_image(im, filter):
    im_h, im_w = im.shape           # size of the image
    print(im.shape)
    cell_size = 8
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    print("NUMCELLSS: " + str(num_cell_h) + ", " + str(num_cell_w))
    block_size = 2
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    print("NUMBLOCKS: " + str(num_blocks_h) + ", " + str(num_blocks_w))

    new_im = im                     # create temporary image.
    return_im = np.zeros(im.shape)
    new_im = np.insert(new_im, im_h, 0, axis=0)
    new_im = np.insert(new_im, im_w, 0, axis=1)
    new_im = np.insert(new_im, 0, 0, axis=0)
    new_im = np.insert(new_im, 0, 0, axis=1)

    for i in range(1, im_h):
        for j in range(1, im_w):
            kernel = new_im[i-1:i+2, j-1: j+2]
            return_im[i-1,j-1] = calculate_filter(filter, kernel)

    return return_im

def get_gradient(im_dx, im_dy):
    # im_dx: x differential image
    # im_dy: y differential image

    # magnitude
    dx = np.power(im_dx, 2)
    dy = np.power(im_dy, 2)
    grad_mag = np.sqrt(np.add(dx, dy))

    # angle
    grad_angle = np.arctan2(im_dy, im_dx)
    grad_angle = np.where(grad_angle < 0, grad_angle + np.pi, grad_angle)

    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    # To do
    print("GRADMAG")
    grad_mag_M, grad_mag_N = np.shape(grad_mag)
    print(grad_mag_M, grad_mag_N)
    grad_angle_M, grad_angle_N = np.shape(grad_angle)
    if grad_mag_M != grad_angle_M or grad_mag_N != grad_angle_N:
        print("Error in Build histogram")
        exit(0)

    M = int(grad_mag_M / cell_size)
    N = int(grad_mag_N / cell_size)

    print("==================================")
    print("==================================")

    print((M,N))

    ori_histo = np.zeros((M, N, 6))    # M, N, degree

    # typical cell size is 8

    for m in range (0, M):
        for n in range (0, N):
            grad_mag1 = 0.0
            grad_mag2 = 0.0
            grad_mag3 = 0.0
            grad_mag4 = 0.0
            grad_mag5 = 0.0
            grad_mag6 = 0.0
            for cell_m in range (0, cell_size):
                for cell_n in range (0, cell_size):
                    angle = grad_angle[cell_m + (m * cell_size), cell_n + (n * cell_size)]
                    if angle >= 0 and np.pi/6 > angle:
                        grad_mag1 += grad_mag[cell_m + (m * cell_size), cell_n + (n * cell_size)]
                    elif angle >= np.pi/6 and np.pi/3 > angle:
                        grad_mag2 += grad_mag[cell_m + (m * cell_size), cell_n + (n * cell_size)]
                    elif angle >= np.pi/3 and np.pi/2 > angle:
                        grad_mag3 += grad_mag[cell_m + (m * cell_size), cell_n + (n * cell_size)]
                    elif angle >= np.pi/2 and np.pi * 4 / 6 > angle:
                        grad_mag4 += grad_mag[cell_m + (m * cell_size), cell_n + (n * cell_size)]
                    elif angle >= np.pi * 4 / 6 and np.pi * 5 / 6 > angle:
                        grad_mag5 += grad_mag[cell_m + (m * cell_size), cell_n + (n * cell_size)]
                    elif angle >= np.pi * 5 / 6 and np.pi > angle:
                        grad_mag6 += grad_mag[cell_m + (m * cell_size), cell_n + (n * cell_size)]
                    elif angle == np.pi:
                        grad_mag1 += grad_mag[cell_m + (m * cell_size), cell_n + (n * cell_size)]
                    else:
                        print(angle)
                        print("Error in Build Histogram")
                        print("      occurred in grad_angle")
                        exit(0)
            ori_histo[m, n, :] = [grad_mag1, grad_mag2, grad_mag3, grad_mag4, grad_mag5, grad_mag6]
    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    M, N, cell_size = np.shape(ori_histo)
    n_M = M - (block_size - 1)
    n_N = N - (block_size - 1)
    n_block = 6 * (block_size ** 2)
    ori_histo_normalized = np.zeros((n_M, n_N, n_block))

    print((n_M, n_N))
    # go to helper for the function

    for i in range (0, n_M):
        for j in range (0, n_N):
            block_ = ori_histo[i:i+block_size, j:j+block_size, :]
            ori_histo_normalized[i, j, :] = histo_normalization(block_, block_size)
    return ori_histo_normalized


def extract_hog(im):
# 1: Convert the gray-scale image to float format and normalize to range [0, 1].
# 2: Get differential images using get_differential_filter and filter_image
# 3: Compute the gradients using get_gradient
# 4: Build the histogram of oriented gradients for all cells using build_histogram
# 5: Build the descriptor of all blocks with normalization using get_block_descriptor
# 6: Return a long vector (hog) by concatenating all block descriptors.

    im = im.astype('float') / 255.0
    # To do
    filter_x, filter_y = get_differential_filter()
    im_dx = filter_image(im, filter_x)
    im_dy = filter_image(im, filter_y)
    grad_mag, grad_angle = get_gradient(im_dx, im_dy)
    ori = build_histogram(grad_mag, grad_angle, 8)
    hog = get_block_descriptor(ori, 2)
    # visualize to verify
    visualize_hog(im, hog, 8, 2)

    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='red', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()


if __name__=='__main__':
    im = cv2.imread('balloons.tif', 0)
    hog = extract_hog(im)

    # filter_x, filter_y = get_differential_filter()
    # im_dx = filter_image(im, filter_x)
    # im_dy = filter_image(im, filter_y)
    # grad_mag, grad_angle = get_gradient(im_dx, im_dy)
    # ori = build_histogram(grad_mag, grad_angle, 6)
    # get_block_descriptor(ori, 2)
    # # DEBUG
    # #
    #
    # im_dx      = image_processing(im_dx)
    # im_dy      = image_processing(im_dy)
    # grad_mag   = image_processing(grad_mag)
    # grad_angle = image_processing(grad_angle)
    #
    # print(grad_mag[2,2])
    # image_processing(im_dy)
    # image_processing(grad_mag)
    # image_processing(grad_angle)

    # cv2.imshow('im', im)
    # cv2.imshow('im_dx', im_dx)
    # cv2.imshow('im_dy', im_dy)
    # cv2.imshow('grad_mag', grad_mag)
    # cv2.imshow('grad_angle', grad_angle)



    # plt.imshow(im_dx, cmap='jet')
    # plt.show()
    # plt.imshow(im_dy, cmap='jet')
    # plt.show()
    # plt.imshow(grad_mag, cmap='jet')
    # plt.show()
    # plt.imshow(get_gradient(im_dx, im_dy)[1], cmap='jet')
    # plt.imshow(grad_mag, cmap = 'jet')
    # plt.show()
    # plt.imshow(grad_angle, cmap = 'jet')
    # plt.show()

    # cv2.imshow('image', filter_image(im, im_dy))
    # cv2.imshow('image', im)
    key = cv2.waitKey(0)


    # exit when ESC
    if key == 27:
        cv2.destroyAllWindows()
        del(im)
        del(im_dx)
        del(im_dy)
