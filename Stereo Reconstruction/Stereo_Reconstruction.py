import cv2
import numpy as np
from numpy.linalg import inv
from numpy.linalg import svd
from numpy.linalg import norm

import scipy.io as sio
import matplotlib.pyplot as plt
import sys
import random

from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D




def find_match(img1, img2):
    '''
    img1, img2 are arrays that already read using cv2.imread.
    no extraction needed.
    '''
    print('PHASE: find match...')
    ratio_test = 0.7
    sift = cv2.xfeatures2d.SIFT_create()
    kp1,des1 = sift.detectAndCompute(img1,None)
    kp2,des2 = sift.detectAndCompute(img2,None)

    # forward
    neigh1 = NearestNeighbors(n_neighbors=2)
    neigh1.fit(des2)
    match1 = neigh1.kneighbors(des1) #[0]-distance [1]-index
    x1=[]
    x2=[]
    for i in range(np.size(match1[0],0)):
        if match1[0][i][0] < ratio_test*match1[0][i][1]:
            x2.append(kp2[match1[1][i][0]].pt)
            x1.append(kp1[i].pt)
    x1 = np.floor(np.array(x1))
    x2 = np.floor(np.array(x2))

    # backward
    neigh2 =  NearestNeighbors(n_neighbors=2)
    neigh2.fit(des1)
    match2 = neigh2.kneighbors(des2)
    xx1=[]
    xx2=[]
    for i in range(np.size(match2[0],0)):
        if match2[0][i][0] < ratio_test*match2[0][i][1]:
            xx1.append(kp1[match2[1][i][0]].pt)
            xx2.append(kp2[i].pt)

    xx1 = np.floor(np.array(xx1))
    xx2 = np.floor(np.array(xx2))

    # bidirectional
    X1 = []
    X2 = []
    for ii in range(np.size(x1,0)):
        for jj in range(np.size(xx1,0)):
            if (x1[ii,:] == xx1[jj,:]).all():
                X1.append(xx1[jj,:])
                X2.append(xx2[jj,:])
                x1[ii,:] = [0, 0]
    pts1 = np.reshape(np.array(X1),[np.size(X1,0),2])
    pts2 = np.reshape(np.array(X2),[np.size(X2,0),2])

    return pts1, pts2










def set_diag(list):
    size = len(list)
    A    = np.zeros((size, size))
    for pos, i in enumerate(list):
        A[pos,pos] = i

    return A

def compute_F(pts1, pts2):
    print('PHASE: compute F ...')
    count_pts1, _ = pts1.shape
    count_pts2, _ = pts2.shape

    if count_pts1 != count_pts2:
        raise ValueError('SIFT feature does not have same matching points')

    # RANSAC parameters
    ransac_iter  = 10000
    ransac_thr   = 0.05

    inliners = np.array([])               # ransac inliner
    pad      = np.ones((count_pts1, 1))
    pts1_pad = np.hstack((pts1, pad))     # n x 3
    pts2_pad = np.hstack((pts2, pad))     # n x 3

    Fs = np.zeros((3,3,ransac_iter))

    z = np.zeros((8,1))

    for ransac in range(ransac_iter):
        place_list = random.sample(range(count_pts1), 8)         # pick 8 random points
        A = np.zeros((8,9))

        for pos, i in enumerate(place_list):
            A[pos, 0] = pts1[i,0] * pts2[i,0]
            A[pos, 1] = pts1[i,1] * pts2[i,0]
            A[pos, 2] = pts2[i,0]
            A[pos, 3] = pts1[i,0] * pts2[i,1]
            A[pos, 4] = pts1[i,1] * pts2[i,1]
            A[pos, 5] = pts2[i,1]
            A[pos, 6] = pts1[i,0]
            A[pos, 7] = pts1[i,1]
            A[pos, 8] = 1

        ### compute fundamental matrix
        _, _, v = svd(A)
        f = v.T[:,-1].reshape(3,3)      # Ax = 0

        U, S, V = svd(f)                # f == U @ set_diag(S) @ V
        S[-1] = 0                       # rank 2
        S = set_diag(S)
        F = U @ S @ V

        ### RANSAC implementation

        inliner = 0
        for count in range(count_pts1):
            if count not in place_list:
                distance = abs(pts2_pad[count,:] @ F @ pts1_pad[count,:].T)
                if distance < ransac_thr:
                    inliner += 1

        inliners = np.append(inliners, inliner)
        Fs[:,:,ransac] = F

    max_iter = np.argmax(inliners)
    F = Fs[:,:,max_iter]

    return F



def triangulation(P1, P2, pts1, pts2):
    print('PHASE: triangulation ...')
    n, _ = pts1.shape
    pts3D = np.empty((n, 3))

    for i in range(n):
        pts_p1 = np.array([0, -1, pts1[i,1], 1, 0, -pts1[i,0], -pts1[i,1], pts1[i,0], 0]).reshape(3,3)       # skew-symmetric matrix
        u_x    = pts_p1 @ P1
        pts_p2 = np.array([0, -1, pts2[i,1], 1, 0, -pts2[i,0], -pts2[i,1], pts2[i,0], 0]).reshape(3,3)       # skew-symmetric matrix
        v_x    = pts_p2 @ P2
        pts    = np.vstack((u_x, v_x))          # 6 x 4 matrix

        _, _, v = svd(pts)
        x = v.T[:,-1]      # Ax = 0
        pts3 = x / x[-1]
        pts3D[i,:] = pts3[:-1]

    return pts3D


def disambiguate_pose(Rs, Cs, pts3Ds):
    print('PHASE: disambiguate pose ...')
    Rs_size = len(Rs)
    Cs_size = len(Cs)
    pts3Ds_size = len(pts3Ds)

    if Rs_size != Cs_size or Cs_size != pts3Ds_size or Rs_size != pts3Ds_size:
         raise ValueError('disambiguate_pose: Different number of arguments')

    step       = Rs_size
    cheirality = np.zeros(step)

    for i in range (step):
        nValid = 0
        r = Rs[i]
        c = Cs[i]
        p = pts3Ds[i]

        r_z = r[2, :]
        for x in p:
            if r_z @ (x.reshape(3,1) - c) > 0 and x[2] > 0:
                nValid += 1
        cheirality[i] = nValid

    max_iter = np.argmax(cheirality)
    R = Rs[max_iter]
    C = Cs[max_iter]
    pts3D = pts3Ds[max_iter]

    return R, C, pts3D


def compute_rectification(K, R, C):
    print('PHASE: compute rectification ...')

    R_rect  = np.empty((3,3))
    r_tilde = np.array([0,0,1]).reshape(3,1)

    R_x_t = C / norm(C)
    R_x   = R_x_t.T       # 1 x 3

    r_z_temp = r_tilde - (np.dot(R_x, r_tilde) * R_x_t)  # 3 x 1
    R_z_t = r_z_temp / norm(r_z_temp)
    R_z   = R_z_t.T

    R_y = np.cross(R_z, R_x)

    R_rect[0,:] = R_x.reshape(-1)
    R_rect[1,:] = R_y.reshape(-1)
    R_rect[2,:] = R_z.reshape(-1)

    H1 = K @ R_rect @ inv(K)
    H2 = K @ R_rect @ R.T @ inv(K)

    return H1, H2



def dense_match(img1, img2):
    print('PHASE: dense match ...')

    if img1.shape != img2.shape:
        raise ValueError('dense_match: Image size does not match')

    h, w = img1.shape
    disparity = np.ones((h, w))

    sift = cv2.xfeatures2d.SIFT_create()

    step_size = 3
    kp = [cv2.KeyPoint(x, y, step_size) for y in range(h)
                                        for x in range(w)]

    kp1, des1 = sift.compute(img1, kp)
    kp2, des2 = sift.compute(img2, kp)

    des1 = des1.reshape(h, w, 128)
    des2 = des2.reshape(h, w, 128)


    for i in range (h):
        for j in range(w):
            if img1[i, j] == 0:
                continue
            d1 = des1[i, j, :]
            du2 = des2[i, 0:j+1, :]

            du1 = np.tile(d1, (j+1, 1))

            d_norm = norm(du1 - du2, axis = 1)
            d_argmin = np.argmin(d_norm) - j
            disparity[i, j] = np.abs(d_argmin)

    # normalization
    disparity[np.where(disparity > 150)] = 150          # set the limit to avoid maximum
    disparity = cv2.normalize(disparity, disparity, alpha=0, beta=150, norm_type=cv2.NORM_MINMAX)

    return disparity




# PROVIDED functions
def compute_camera_pose(F, K):
    E = K.T @ F @ K
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []
    for i in range(4):
        Cs.append(-Rs[i].T @ ts[i])
    return Rs, Cs


def visualize_img_pair(img1, img2):
    img = np.hstack((img1, img2))
    if img1.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def visualize_find_match(img1, img2, pts1, pts2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    img_h = img1.shape[0]
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    pts1 = pts1 * scale_factor1
    pts2 = pts2 * scale_factor2
    pts2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in range(pts1.shape[0]):
        plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'b.-', linewidth=0.5, markersize=5)
    plt.axis('off')
    plt.show()


def visualize_epipolar_lines(F, pts1, pts2, img1, img2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i][0] + 0.5), int(pts1[i][1] + 0.5)
        ax1.scatter(x1, y1, s=5)
        p1, p2 = find_epipolar_line_end_points(img2, F, (x1, y1))
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    for i in range(pts2.shape[0]):
        x2, y2 = int(pts2[i][0] + 0.5), int(pts2[i][1] + 0.5)
        ax2.scatter(x2, y2, s=5)
        p1, p2 = find_epipolar_line_end_points(img1, F.T, (x2, y2))
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    ax1.axis('off')
    ax2.axis('off')
    plt.show()


def find_epipolar_line_end_points(img, F, p):
    img_width = img.shape[1]
    el = np.dot(F, np.array([p[0], p[1], 1]).reshape(3, 1))
    p1, p2 = (0, -el[2] / el[1]), (img.shape[1], (-img_width * el[0] - el[2]) / el[1])
    _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), p1, p2)
    return p1, p2


def visualize_camera_poses(Rs, Cs):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2 = Rs[i], Cs[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1)
        draw_camera(ax, R2, C2)
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def visualize_camera_poses_with_pts(Rs, Cs, pts3Ds):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2, pts3D = Rs[i], Cs[i], pts3Ds[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1, 5)
        draw_camera(ax, R2, C2, 5)
        ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def draw_camera(ax, R, C, scale=0.2):
    axis_end_points = C + scale * R.T  # (3, 3)
    vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
    vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)

    # draw coordinate system of camera
    ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
    ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
    ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

    # draw square window and lines connecting it to camera center
    ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], 'k-')
    ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], 'k-')
    ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], 'k-')
    ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], 'k-')
    ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], 'k-')


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_disparity_map(disparity):
    plt.imshow(disparity, cmap='jet')
    plt.show()


if __name__ == '__main__':
    # read in left and right images as RGB images
    img_left = cv2.imread('./left.bmp', 1)
    img_right = cv2.imread('./right.bmp', 1)
    # visualize_img_pair(img_left, img_right)

    # Step 1: find correspondences between image pair
    pts1, pts2 = find_match(img_left, img_right)
    visualize_find_match(img_left, img_right, pts1, pts2)

    # Step 2: compute fundamental matrix
    F = compute_F(pts1, pts2)
    visualize_epipolar_lines(F, pts1, pts2, img_left, img_right)

    # Step 3: computes four sets of camera poses
    K = np.array([[350, 0, 960/2], [0, 350, 540/2], [0, 0, 1]])
    Rs, Cs = compute_camera_pose(F, K)
    visualize_camera_poses(Rs, Cs)

    # Step 4: triangulation
    pts3Ds = []
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    for i in range(len(Rs)):
        P2 = K @ np.hstack((Rs[i], -Rs[i] @ Cs[i]))
        pts3D = triangulation(P1, P2, pts1, pts2)
        pts3Ds.append(pts3D)
    visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

    # Step 5: disambiguate camera poses
    R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds)

    # Step 6: rectification
    H1, H2 = compute_rectification(K, R, C)
    img_left_w = cv2.warpPerspective(img_left, H1, (img_left.shape[1], img_left.shape[0]))
    img_right_w = cv2.warpPerspective(img_right, H2, (img_right.shape[1], img_right.shape[0]))
    visualize_img_pair(img_left_w, img_right_w)


    # Step 7: generate disparity map
    img_left_w = cv2.resize(img_left_w, (int(img_left_w.shape[1] / 2), int(img_left_w.shape[0] / 2)))  # resize image for speed
    img_right_w = cv2.resize(img_right_w, (int(img_right_w.shape[1] / 2), int(img_right_w.shape[0] / 2)))
    img_left_w = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    img_right_w = cv2.cvtColor(img_right_w, cv2.COLOR_BGR2GRAY)
    disparity = dense_match(img_left_w, img_right_w)
    visualize_disparity_map(disparity)

    # save to mat
    sio.savemat('stereo.mat', mdict={'pts1': pts1, 'pts2': pts2, 'F': F, 'pts3D': pts3D, 'H1': H1, 'H2': H2,
                                     'img_left_w': img_left_w, 'img_right_w': img_right_w, 'disparity': disparity})
