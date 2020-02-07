import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_differential_filter():
    filter_x = np.array([-1,0,1,-1,0,1,-1,0,1]).reshape(3,3)
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
    # division = np.sum(block ** 2, axis = 2) + e ** 2  # block_size x block_size
    return_block = np.zeros(x * y * z).reshape(np.shape(block))
    for i in range (0, block_size):
        for j in range (0, block_size):
            division = np.sqrt(np.sum(block[i,j,:] ** 2) + e ** 2)
            return_block[i,j,:] = block[i,j,:]/division

    return return_block.reshape(x*y*z)


def block_descriptor(im):
    im = im.astype('float') / 255.0

    filter_x, filter_y = get_differential_filter()
    im_dx = filter_image(im, filter_x)
    im_dy = filter_image(im, filter_y)
    grad_mag, grad_angle = get_gradient(im_dx, im_dy)
    ori = build_histogram(grad_mag, grad_angle, 8)
    hog = get_block_descriptor(ori, 2)

    return hog


# input template and target vector.
def ncc(tar, tem):
    tar_h, tar_w, tar_d = np.shape(tar)
    tem_h, tem_w, tem_d = np.shape(tem)
    if np.shape(tem) != np.shape(tar):
        print("ERROR: NCC")
        print("       template and target image has to be same to do NCC")
        exit()

    mean_a = np.mean(tar)
    mean_b = np.mean(tem)
    dot = 0.0
    x   = 0.0
    y   = 0.0
    for i in range (tar_h):
        for j in range (tar_w):
            a = tar[i,j,:] - mean_a
            b = tem[i,j,:] - mean_b
            dot += np.dot(a, b)
            x += np.sum(a ** 2)
            y += np.sum(b ** 2)

    score = dot / np.sqrt(x * y)
    return score


# IoU : intersection of Union
def box_calculation(box1, box2, box_size):
    x1 = box1[0]
    x2 = box1[0] + box_size
    y1 = box1[1]
    y2 = box1[1] + box_size

    x3 = box2[0]
    x4 = box2[0] + box_size
    y3 = box2[1]
    y4 = box2[1] + box_size

    if x1 < x3 and x2 < x3:
        return 0
    elif x3 < x1 and x4 < x1:
        return 0

    if y1 < y3 and y2 < y3:
        return 0
    elif y3 < y1 and y4 < y1:
        return 0

    if x1 < x3:
        inter_x = x2 - x3
        union_x = x4 - x1
        union_sub1_x = x4 - x2
        union_sub2_x = x3 - x1
        if y1 < y3:
            inter_y = y2 - y3
            union_y = y4 - y1
            union_sub1_y = y3 - y1
            union_sub2_y = y4 - y2
        else:
            inter_y = y4 - y1
            union_y = y2 - y3
            union_sub1_y = y2 - y4
            union_sub2_y = y1 - y3
    else:
        inter_x = x4 - x1
        union_x = x2 - x3
        union_sub1_x = x2 - x4
        union_sub2_x = x1 - x3
        if y1 < y3:
            inter_y = y2 - y3
            union_y = y4 - y1
            union_sub1_y = y4 - y2
            union_sub2_y = y3 - y1
        else:
            inter_y = y4 - y1
            union_y = y2 - y3
            union_sub1_y = y1 - y3
            union_sub2_y = y2 - y4

    inter = inter_x * inter_y
    union = (union_x * union_y) - (union_sub1_x * union_sub1_y) - (union_sub2_x * union_sub2_y)
    iou = inter / union

    return iou


def IoU (boxes, box_size):
    count = np.shape(boxes)[0]
    i = 0   # dominant
    j = 1

    while True:
        iou = box_calculation(boxes[i], boxes[j], box_size)
        if iou > 0.5:
            if boxes[i, 2] > boxes[j, 2]:
                np.delete(boxes, j, 0)
                count -= 1
            else:
                boxes[[i,j],:] = boxes[[j,i],:] # swap the position
                np.delete(boxes, j, 0)
                count -= 1
        else:
            j += 1
            
        if j == count:
            i += 1
            j = i+1
        if i == count - 1:
            break

    return boxes




########################################################################
#####################    GIVEN CODE : MODIFIED     #####################
########################################################################

def filter_image(im, filter):
    im_h, im_w = im.shape           # size of the image

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
    grad_mag_M, grad_mag_N = np.shape(grad_mag)
    grad_angle_M, grad_angle_N = np.shape(grad_angle)

    if grad_mag_M != grad_angle_M or grad_mag_N != grad_angle_N:
        print("Error in Build histogram")
        exit(0)

    M = int(grad_mag_M / cell_size)
    N = int(grad_mag_N / cell_size)

    ori_histo = np.zeros((M, N, 6))    # M, N, degree
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
    # hog = np.zeros(111864)
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





def face_recognition(I_target, I_template):
    # bounding boxes n x 3 .. [x, y, s]
    # s = a . b
    target_h, target_w = np.shape(I_target)
    template_h, template_w = np.shape(I_template)
    if target_h < template_h:
        print("ERROR: template image is bigger than target image")
        exit(0)
    if target_w < template_w:
        print("ERROR: template image is bigger than target image")
        exit(0)


    template_hog = block_descriptor(I_template)


    epsilon = 0.3
    face_box = np.array([])
    count = 0

    # for i in range (int(template_h / 2), target_h - int(template_h / 2)):
    #     for j in range (int(template_w / 2), target_w - int(template_w / 2)):
    for i in range (0, target_h - template_h):
        for j in range(0, target_w - template_h):
            target = I_target[i:i+template_h, j:j+template_w]
            target_hog = block_descriptor(target)

            s = ncc(target_hog, template_hog)

            if (s > 0.4):
                face_box = np.append(face_box, [j,i,s])
                count += 1

    face_box = face_box.reshape(count, 3)
    return face_box




def box_visualization(I_target,bounding_boxes,box_size):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()

    ## IOU occurs here
    bouding_boxes = IoU(bounding_boxes, box_size)


    for ii in range(bounding_boxes.shape[0]):

        # bounding boxes are left top corner.
        x1 = bounding_boxes[ii, 0]
        y1 = bounding_boxes[ii, 1]
        x2 = bounding_boxes[ii, 0] + box_size
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)


    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()


if __name__=='__main__':
    # im = cv2.imread('balloons.tif', 0)
    # hog = extract_hog(im)

    I_target= cv2.imread('img/target.png', 0)
    #MxN image

    I_template = cv2.imread('img/template.png', 0)
    #mxn  face template

    bounding_boxes = face_recognition(I_target, I_template)

    I_target_c= cv2.imread('img/target.png')
    # MxN image (just for visualization)
    box_visualization(I_target_c, bounding_boxes, I_template.shape[0])
    #this is visualization code.
