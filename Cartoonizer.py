import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

def display_image(image):
    fig = plt.figure()
    fig.set_size_inches((5,5))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap('hot')
    ax.imshow(image, aspect='equal')

def grayscale(image):
    new_image = np.zeros_like(image)
    (N, M, _) = image.shape

    for i in range(N):
        for j in range(M):
            val = image[i][j][0] * 0.2 + image[i][j][1] * 0.71 + image[i][j][2] * 0.07
            new_image[i][j][0] = val
            new_image[i][j][1] = val
            new_image[i][j][2] = val

    return new_image

def blur(image):
    kernel =   [[0.05, 0.11, 0.05],
                [0.11, 0.25, 0.11],
                [0.05, 0.11, 0.05]]

    (N, M, _) = image.shape
    new_image = np.zeros_like(image)

    for i in range(N):
        for j in range(M):
            val_R = 0
            val_G = 0
            val_B = 0
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if i + x >= 0 and i + x < N and j + y >= 0 and j + y < M:
                        val_R += image[i + x][j + y][0] * kernel[x + 1][y + 1]
                        val_G += image[i + x][j + y][1] * kernel[x + 1][y + 1]
                        val_B += image[i + x][j + y][2] * kernel[x + 1][y + 1]
            new_image[i][j][0] = val_R
            new_image[i][j][1] = val_G
            new_image[i][j][2] = val_B
    return new_image

def sobel(image):
    kernelX =  [[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]]

    kernelY =  [[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]]

    (N, M, _) = image.shape
    new_image = np.zeros_like(image)
    D = np.zeros((N, M))
    direction = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            Dx = 0
            Dy = 0
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if i + x >= 0 and i + x < N and j + y >= 0 and j + y < M:
                        Dx += image[i + x][j + y][0] * kernelX[x + 1][y + 1]
                        Dy += image[i + x][j + y][1] * kernelY[x + 1][y + 1]

            D[i][j] = abs(Dx) + abs(Dy)
            theta = np.arctan(Dy / Dx) if Dx != 0 else 0

            if -22.5 <= theta < 22.5 or -157.5 <= theta < 157.5:
                direction[i][j] = 0
            if 22.5 <= theta < 67.5 or  202.5 <= theta < 247.5:
                direction[i][j] = 1
            if 67.5 <= theta < 112.5 or 247.5 <= theta < 292.5:
                direction[i][j] = 2
            if 112.5 <= theta < 157.5 or 292.5 <= theta < 337.5:
                direction[i][j] = 3
            if D[i][j] > 0.5:
                new_image[i][j][0] = 1
                new_image[i][j][1] = 1
                new_image[i][j][2] = 1

    return new_image, D, direction




def non_maxima(image, D, direction):
    (N, M, _) = image.shape
    new_image = np.copy(image)
    for i in range(N):
        for j in range(M):
            if direction[i][j] == 0:
                A = D[i - 1][j] if i - 1 >= 0 else -1
                B = D[i + 1][j] if i + 1 < N else -1
            elif direction[i][j] == 1:
                A = D[i - 1][j - 1] if i - 1 >= 0 and j - 1 >= 0 else -1
                B = D[i + 1][j + 1] if i + 1 < N and j + 1 < M else -1
            elif direction[i][j] == 2:
                A = D[i][j - 1] if j - 1 >= 0 else -1
                B = D[i][j + 1] if j + 1 < M else -1
            elif direction[i][j] == 3:
                A = D[i - 1][j + 1] if i - 1 >= 0 and j + 1 < M else -1
                B = D[i + 1][j - 1] if i + 1 < N and j - 1 >= 0 else -1

            if A > D[i][j] or B > D[i][j]:
                new_image[i][j] = np.zeros(3)
            else:
                if direction[i][j] == 0:
                    if A != -1:
                        new_image[i - 1][j] = np.zeros(3)
                    if B != -1:
                        new_image[i + 1][j] = np.zeros(3)
                elif direction[i][j] == 1:
                    if A != -1:
                        new_image[i - 1][j - 1] = np.zeros(3)
                    if B != -1:
                        new_image[i + 1][j + 1] = np.zeros(3)
                elif direction[i][j] == 2:
                    if A != -1:
                        new_image[i][j - 1] = np.zeros(3)
                    if B != -1:
                        new_image[i][j + 1] = np.zeros(3)
                elif direction[i][j] == 3:
                    if A != -1:
                        new_image[i - 1][j + 1] = np.zeros(3)
                    if B != -1:
                        new_image[i + 1][j - 1] = np.zeros(3)
    return new_image

def hysteresis_thresholding(image, D, direction):
    (N, M, _) = image.shape
    new_image = np.zeros_like(image)

    low_threshold = 0.5
    high_threshold = 0.8

    neighbors = np.array([[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]])

    T1 = np.array([(i, j) for i in range(N) for j in range(M) if low_threshold < D[i][j] <= high_threshold])
    T2 = np.array([(i, j) for i in range(N) for j in range(M) if D[i][j] >= high_threshold])

    for position in T2:
        for neighbor in neighbors:
            x = position[0] + neighbor[0]
            y = position[1] + neighbor[1]
            if x >= 0 and x < N and y >= 0 and y < M and (x, y) in T2:
                new_image[position[0]][position[1]] = np.array([1, 1, 1])
                break
            else:
                for neighbor in neighbors:
                    x = position[0] + neighbor[0]
                    y = position[1] + neighbor[1]
                    if x >= 0 and x < N and y >= 0 and y < M and (x, y) in T1:
                        new_image[position[0]][position[1]] = np.array([1, 1, 1])
                        break

    return new_image

def median_cut(image, rounds = 3):
    print(image.shape)
    (N, M, _) = image.shape
    new_image = np.zeros_like(image)

    start_colors = np.array([(image[i][j], (i, j)) for i in range(N) for j in range(M)], dtype=object)
    color_group = [start_colors]
    new_color_group = []
    for round in range(rounds + 1):
        for member in color_group:
            R_range = max([member[i][0][0] for i in range(member.shape[0])]) - \
                min([member[i][0][0] for i in range(member.shape[0])])
            B_range = max([member[i][0][1] for i in range(member.shape[0])]) - \
                min([member[i][0][1] for i in range(member.shape[0])])
            G_range = max([member[i][0][2] for i in range(member.shape[0])]) - \
                min([member[i][0][2] for i in range(member.shape[0])])

            if R_range >= max(B_range, G_range):
                R_mid = np.average([member[i][0][0] for i in range(member.shape[0])])
                low = np.array([member[i] for i in range(member.shape[0]) if member[i][0][0] < R_mid], dtype=object)
                high = np.array([member[i] for i in range(member.shape[0]) if member[i][0][0] >= R_mid], dtype=object)

                new_color_group.append(low)
                new_color_group.append(high)
            elif B_range >= max(R_range, G_range):
                B_mid = np.average([member[i][0][1] for i in range(member.shape[0])])
                low = np.array([member[i] for i in range(member.shape[0]) if member[i][0][1] < B_mid], dtype=object)
                high = np.array([member[i] for i in range(member.shape[0]) if member[i][0][1] >= B_mid], dtype=object)

                new_color_group.append(low)
                new_color_group.append(high)
            elif G_range >= max(R_range, B_range):
                G_mid = np.average([member[i][0][2] for i in range(member.shape[0])])
                low = np.array([member[i] for i in range(member.shape[0]) if member[i][0][2] < G_mid], dtype=object)
                high = np.array([member[i] for i in range(member.shape[0]) if member[i][0][2] >= G_mid], dtype=object)

                new_color_group.append(low)
                new_color_group.append(high)

        color_group = new_color_group
        new_color_group = []

    for member in color_group:
        R_avg = np.average([member[i][0][0] for i in range(member.shape[0])])
        G_avg = np.average([member[i][0][1] for i in range(member.shape[0])])
        B_avg = np.average([member[i][0][2] for i in range(member.shape[0])])

        for i in range (member.shape[0]):
            new_image[member[i][1][0]][member[i][1][1]] = np.array([R_avg, G_avg, B_avg])

    return new_image

def add_outline_to_image(image, borders):
    (N, M, _) = image.shape
    new_image = np.copy(image)

    fill_color_R = np.average([image[i][j][0] for i in range(N) for j in range(M)]) / 3
    fill_color_G = np.average([image[i][j][1] for i in range(N) for j in range(M)]) / 3
    fill_color_B = np.average([image[i][j][2] for i in range(N) for j in range(M)]) / 3

    R_range = max([image[i][j][0] for i in range(N) for j in range(M)]) - \
                min([image[i][j][0] for i in range(N) for j in range(M)])
    B_range = max([image[i][j][1] for i in range(N) for j in range(M)]) - \
                min([image[i][j][1] for i in range(N) for j in range(M)])
    G_range = max([image[i][j][2] for i in range(N) for j in range(M)]) - \
                min([image[i][j][2] for i in range(N) for j in range(M)])

    if R_range >= max(B_range, G_range):
        fill_color_R /= 1.5
        fill_color_G /= 2 if B_range > G_range else 2.5
        fill_color_B /= 2.5 if B_range > G_range else 2
    elif G_range >= max(R_range, B_range):
        fill_color_G /= 1.5
        fill_color_R /= 2 if B_range > R_range else 2.5
        fill_color_B /= 2.5 if B_range > R_range else 2
    elif B_range >= max(R_range, G_range):
        fill_color_B /= 1.5
        fill_color_R /= 2 if G_range > R_range else 2.5
        fill_color_G /= 2.5 if G_range > R_range else 2


    for i in range(N):
        for j in range(M):
            if borders[i][j][0] == 1:
                new_image[i][j] = np.array([fill_color_R, fill_color_G, fill_color_B])
    return new_image

os.chdir('photos')
fnames = os.listdir()
os.chdir(os.path.dirname(__file__))
# os.chdir('/mnt/c/Users/dsabi/Desktop/spam/SPG/Tema 2')

for fname in fnames:
    image = plt.imread('photos/' + fname, format=None)
    image = image / 255
    dir = os.path.dirname(__file__) + '/' + fname[:-4]
    if fname[:-4] in os.listdir():
        shutil.rmtree(dir)
    os.mkdir(dir)
    plt.imsave(dir + "/1_original_" + fname, image)

    print("Doing median cut...")
    image_color = median_cut(image, 2)
    plt.imsave(dir + "/2_median_cut_" + fname, image_color)

    print("Getting rid of noise...")
    image_color = blur(image_color)
    plt.imsave(dir + "/3_blured_median_cut_" + fname, image_color)

    print("Blurring original image...")
    borders = blur(image)
    plt.imsave(dir + "/4_blur_" + fname, borders)

    print("Converting to grayscale...")
    borders = grayscale(borders)
    plt.imsave(dir + "/5_grayscale_blur_" + fname, borders)

    print("Applying Sobel...")
    (borders, D, direction) = sobel(borders)
    plt.imsave(dir + "/6_sobel_" + fname, borders)

    print("Applying non maxima thresholding...")
    borders = non_maxima(borders, D, direction)
    plt.imsave(dir + "/7_non_maxima_" + fname, borders)

    print("Applying isterie...")
    borders = hysteresis_thresholding(borders, D, direction)
    plt.imsave(dir + "/8_hysteresis_" + fname, borders)

    print("Combinig image and borders...")
    image = add_outline_to_image(image_color, borders)
    plt.imsave(dir + "/9_final_" + fname, image)

    display_image(image)

plt.show()