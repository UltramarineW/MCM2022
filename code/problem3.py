import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import math
import random
import open3d as o3d
import copy


dataset_num = 2000
max_pos_err = 6
generate_rates = [0, 0.5, 1, 2, 4, 8, 12, 16, 20]
generate_num = 50
near_choose = 4

Point = namedtuple('Point', ['x', 'y'])


def addPoint(a: Point, b: Point):
    return Point(a.x + b.x, a.y + b.y)


Point.__add__ = addPoint


def subPoint(a: Point, b: Point):
    return Point(a.x - b.x, a.y - b.y)


Point.__sub__ = subPoint


def divf(x):
    if x == 0:
        return 1e-7
    return x


def roatate(p: Point, theta):
    d = dis(p, Point(0, 0))
    theta0 = get_ang(p)
    return Point(d * math.cos(theta + theta0), d * math.sin(theta + theta0))


def get_ang(p: Point):
    ang = math.atan(p.y / divf(p.x))
    if p.x < 0:
        ang += math.pi
    if ang < 0:
        ang += 2 * math.pi
    return ang


def dis(a: Point, b: Point):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def calc_jiao(a: Point, b: Point, ra, rb):
    l = dis(a, b)
    if l > ra + rb:
        return None
    k1 = (b.y - divf(a.y)) / (b.x - divf(a.x))
    k2 = - (b.x - divf(a.x)) / (b.y - divf(a.y))
    x0 = a.x + (b.x - a.x) * (ra ** 2 - rb ** 2 + l ** 2) / 2 / l ** 2
    y0 = a.y + k1 * (x0 - a.x)
    r2 = ra ** 2 - (x0 - a.x) ** 2 - (y0 - a.y) ** 2
    return Point(x0 - math.sqrt(r2 / (1 + k2 ** 2)), y0 - k2 * math.sqrt(r2 / (1 + k2 ** 2))), \
           Point(x0 + math.sqrt(r2 / (1 + k2 ** 2)), y0 + k2 * math.sqrt(r2 / (1 + k2 ** 2)))


def get_ans(a0, ap1, ap2, cross):  # corss: p2-p1
    a2, a1 = abs(ap1 - a0), abs(ap2 - a0)
    theta = cross * 40 / 180 * math.pi

    if ap1 >= a0:
        a = Point(50, 50 / divf(math.tan(a2)))
    else:
        a = Point(50, -50 / divf(math.tan(a2)))

    if a0 >= ap2:
        b = Point(50 * math.sin(theta + a1) / divf(math.sin(a1)), -50 * math.cos(theta + a1) / divf(math.sin(a1)))
    else:
        b = Point(-50 * math.sin(theta - a1) / divf(math.sin(a1)), 50 * math.cos(theta - a1) / divf(math.sin(a1)))

    ra, rb = math.fabs(50 / divf(math.sin(a2))), math.fabs(50 / divf(math.sin(a1)))
    tmp = calc_jiao(a, b, ra, rb)
    ans = None
    if tmp is not None:
        c, d = tmp
        if dis(c, Point(0, 0)) >= dis(d, Point(0, 0)):
            ans = c
        else:
            ans = d
    return ans, (a, ra), (b, rb)


def getans3p1x(p1: Point, p2: Point, p3: Point, a1_, a2_):
    theta0 = get_ang(p1 - p2)
    a1, a2 = a2_, a1_
    theta = get_ang(p3 - p2) - theta0
    l12, l32 = dis(p1, p2), dis(p3, p2)
    a = Point(l12 / 2, l12 / 2 / divf(math.tan(a2)))
    b = Point(l32 / 2 * math.sin(theta + a1) / divf(math.sin(a1)), -l32 / 2 * math.cos(theta + a1) / divf(math.sin(a1)))
    ra, rb = math.fabs(l12 / 2 / divf(math.sin(a2))), math.fabs(l32 / 2 / divf(math.sin(a1)))
    tmp = calc_jiao(a, b, ra, rb)
    ans = None
    if tmp is not None:
        c, d = tmp
        if dis(c, Point(0, 0)) >= dis(d, Point(0, 0)):
            ans = c
        else:
            ans = d
        ans = roatate(ans, theta0) + p2
    return ans, (roatate(a, theta0) + p2, ra), (roatate(b, theta0) + p2, rb)


def showdata(data: list[Point]):
    for fly in std_flys:
        plt.scatter(*fly, s=20, c='blue')
    for fly in data:
        plt.scatter(*fly, s=20, c='green')


def get_basic_change():
    global generate_rates
    min_dis, min_fa = 1e9, None
    max_dis, max_fa = 0, None
    for i in np.linspace(0, math.pi, 200)[:-1]:
        pos = std_flys[2] + Point(generate_rates[-1] * math.cos(i), generate_rates[-1] * math.sin(i))
        a0, a1, a2 = geta0a1a2(p0, p1, p2, p3, pos)
        pos = std_flys[2] + Point(generate_rates[-1] * math.cos(i + math.pi),
                                  generate_rates[-1] * math.sin(i + math.pi))
        op_a0, op_a1, op_a2 = geta0a1a2(p0, p1, p2, p3, pos)
        tmp_dis = (a0 - op_a0) ** 2 + (a1 - op_a1) ** 2 + (a2 - op_a2) ** 2
        if tmp_dis < min_dis:
            min_dis = tmp_dis
            min_fa = np.array([(a0 - op_a0), (a1 - op_a1), (a2 - op_a2)])
        if tmp_dis > max_dis:
            max_dis = tmp_dis
            max_fa = np.array([(a0 - op_a0), (a1 - op_a1), (a2 - op_a2)])
    generate_angs = []
    for j in generate_rates:
        for i in np.linspace(0, 2 * math.pi, generate_num + 1)[:-1]:
            pos = std_flys[2] + Point(j * math.cos(i), j * math.sin(i))
            a0, a1, a2 = get_ang(p2 - pos) - get_ang(p0 - pos), get_ang(p1 - pos) - get_ang(p2 - pos), \
                         get_ang(p2 - pos) - get_ang(p3 - pos)
            generate_angs.append(np.array([a0, a1, a2]))
    generate_angs = generate_angs[generate_num - 1:]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(generate_angs))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(1, 5))
    top_fas = np.asarray(pcd.normals)
    top_fa = np.array([np.median(top_fas[:, 0]), np.median(top_fas[:, 1]), np.median(top_fas[:, 2])])
    change_basic = np.concatenate([[min_fa], [max_fa], [top_fa]], axis=0)
    pos = std_flys[2]
    a0, a1, a2 = get_ang(p2 - pos) - get_ang(p0 - pos), get_ang(p1 - pos) - get_ang(p2 - pos), \
                 get_ang(p2 - pos) - get_ang(p3 - pos)
    centure = np.array([a0, a1, a2])
    # o3d.visualization.draw_geometries([pcd])
    print(np.linalg.inv(change_basic.T))
    print(centure)
    return np.linalg.inv(change_basic.T), centure


def get_move_vec(policy_vecs, angles, p0_angle, self_num):
    if self_num % 2 == 0:
        a0, a1, a2 = angles
    else:
        a0, a2, a1 = angles
        a0 = -a0
    change_pos = np.matmul(change_basic, (np.array([a0, a1, a2]) - centure).T)
    [k, idx, pdis] = pcd_tree.search_knn_vector_3d(change_pos, near_choose)
    pdis = np.asfarray(pdis)
    select_vecs = policy_vecs[idx]
    vec = np.average(select_vecs, axis=0, weights=pdis)
    vec_d = dis(Point(*vec), Point(0, 0))
    vec_ang = get_ang(Point(*vec))
    if self_num % 2 == 0:
        vec = Point(vec_d * math.cos(vec_ang + p0_angle), vec_d * math.sin(vec_ang + p0_angle))
    else:
        vec = Point(vec_d * math.cos(p0_angle - vec_ang), vec_d * math.sin(p0_angle - vec_ang))
    return vec


def norm_ang(ang):
    pi = math.pi
    if ang > 2 * pi:
        ang -= 2 * pi
    if ang < 0:
        ang += 2 * pi
    return ang

def geta0a1a2(p0, p1, p2, p3, pos):
    a0, a1, a2 = get_ang(p2 - pos) - get_ang(p0 - pos), get_ang(p1 - pos) - get_ang(p2 - pos), \
        get_ang(p2 - pos) - get_ang(p3 - pos)
    return a0, norm_ang(a1), norm_ang(a2)

r = 100
std_flys = [Point(0, 0)]
for i in range(9):
    ang = i * 40 / 180 * math.pi
    std_flys.append(Point(r * math.cos(ang), r * math.sin(ang)))

dataset = []
for i in range(dataset_num):
    data = []
    for fly in std_flys:
        data.append(fly + Point((random.randint(0, 2) * 2 - 1) * random.randint(0, max_pos_err * 10) / 10,
                                (random.randint(0, 2) * 2 - 1) * random.randint(0, max_pos_err * 10) / 10))
    dataset.append(data)

generate_points = []
generate_angs = []
policy_vecs_beg = []
p0 = std_flys[0]
p1 = std_flys[1]
p2 = std_flys[7]
p3 = std_flys[4]
change_basic, centure = get_basic_change()
for j in generate_rates:
    for i in np.linspace(0, 2 * math.pi, generate_num + 1)[:-1]:
        pos = std_flys[2] + Point(j * math.cos(i), j * math.sin(i))
        a0, a1, a2 = geta0a1a2(p0, p1, p2, p3, pos)
        generate_angs.append(np.matmul(change_basic, (np.array([a0, a1, a2]) - centure).T))
        generate_points.append(pos)
        if j != 0:
            vec = generate_points[-generate_num - 1] - pos
            vec_d = dis(vec, Point(0, 0)) * 0.8
            if j < 1.1:
                vec_d *= 0.8
            if j < 0.51:
                vec_d *= 0.5
            vec_ang = get_ang(vec) - get_ang(p0 - pos)
            vec = Point(vec_d * math.cos(vec_ang), vec_d * math.sin(vec_ang))
            policy_vecs_beg.append(np.array(vec))
        else:
            policy_vecs_beg.append(np.zeros(2))
generate_points = generate_points[generate_num - 1:]
generate_angs = generate_angs[generate_num - 1:]
policy_vecs_beg = np.array(policy_vecs_beg[generate_num - 1:])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(generate_angs))
pcd_tree = o3d.geometry.KDTreeFlann(pcd)


data = [(0,0),
(100,0),
(98,40.10),
(112,80.21),
(105,119.75),
(98,159.86),
(112,199.96),
(105,240.07),
(98,280.17),
(112,320.28),]
data = [Point(i[0] * math.cos(i[1] / 180 * math.pi), i[0] * math.sin(i[1] / 180 * math.pi)) for i in data]

if True:
    data_flys = copy.deepcopy(data)
    p0 = data_flys[0]
    # data_flys[0] = std_flys[0]
    # data_flys[1] = std_flys[1]
    # data_flys[4] = std_flys[4]
    # data_flys[7] = std_flys[7]
    flys = data_flys[1:]
    p_idx = [0, 3, 6]
    for fly in data_flys:
        plt.scatter(*fly, s=20, c='blue')
    for i in range(50):
        j = (p_idx[0] + 1) % len(flys)
        while j != p_idx[0]:
            # print(j)
            if j in p_idx:
                j += 1
                if j == len(flys):
                    j = 0
                continue
            pos = flys[j]
            if j - 1 in p_idx:
                p1 = flys[j - 1]
                p2 = flys[j - 4]
                p3 = flys[j - 7]
                a0, a1, a2 = geta0a1a2(p0, p1, p2, p3, pos)
                print([a0, a1, a2])
                vec = get_move_vec(policy_vecs_beg, (a0, a1, a2), get_ang(p0 - pos), 0)
            else:
                p1 = flys[j - 2]
                p2 = flys[j - 5]
                p3 = flys[j - 8]
                print([a0, a1, a2])
                a0, a1, a2 = geta0a1a2(p0, p1, p2, p3, pos)
                vec = get_move_vec(policy_vecs_beg, (a0, a1, a2), get_ang(p0 - pos), 1)
            plt.scatter(*pos, s=20, c='red')
            # plt.plot([pos.x, pos.x + vec[0] * 10], [pos.y, pos.y + vec[1] * 10], 'm')
            plt.arrow(pos.x, pos.y, vec[0] , vec[1] , width = 0.2, head_width = 5, ec = "m", fc = 'm')
            # plt.plot([pos.x, pos.x + vec[0]], [pos.y, pos.y + vec[1]], 'm')

            flys[j] = pos + Point(*vec)

            j += 1
            if j == len(flys):
                j = 0
        plt.axis('equal')
        plt.pause(1)
        plt.clf()
        for fly in flys:
            plt.scatter(*fly, s=20, c='blue')
        plt.scatter(*p0, s=20, c='blue')
        for t in range(len(p_idx)):
            p_idx[t] += 1
            if p_idx[t] == len(flys):
                p_idx[t] = 0
    r = 0
    for fly in flys:
        r += dis(fly, p0)
    r /= len(flys)
    draw_circle = plt.Circle(p0, r, fill=False)
    plt.gcf().gca().add_artist(draw_circle)
    print(r)
    err = 0
    for fly in flys:
        err += math.fabs(dis(p0, fly) - r)
    print(err / len(flys) / r)
    r = 0
    for i in range(len(flys)):
        r += dis(flys[i], flys[i - 1])
    r /= len(flys)
    err = 0
    for i in range(len(flys)):
        err += math.fabs(dis(flys[i], flys[i - 1]) - r)
    print(err / len(flys) / r)

plt.axis('equal')
plt.show()
