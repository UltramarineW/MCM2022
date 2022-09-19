import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import math
import random
import open3d as o3d
import copy
import multiprocessing as mp
import pickle

dataset_num = 5000
max_pos_err = 3
generate_rates = [0, 2, 3.5, 5.5, 6.5,]
generate_num = 30
near_choose = 4

change_num = 20
calc_loss_num = 10
people_num = 48


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
    for i in range(len(std_flys)):
        plt.scatter(*std_flys[i], s=20, c='blue')

    for fly in data:
        plt.scatter(*fly, s=20, c='green')


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


def get_policy(p1, p2, p3, p):
    min_dis, min_fa = 1e9, None
    max_dis, max_fa = 0, None
    for i in np.linspace(0, math.pi, 200)[:-1]:
        pos = p + Point(generate_rates[-1] * math.cos(i), generate_rates[-1] * math.sin(i))
        a1, a2 = get_ang(p1 - pos) - get_ang(p2 - pos), get_ang(p2 - pos) - get_ang(p3 - pos)
        a1, a2 = norm_ang(a1), norm_ang(a2)
        pos = p + Point(generate_rates[-1] * math.cos(i + math.pi), generate_rates[-1] * math.sin(i + math.pi))
        op_a1, op_a2 = get_ang(p1 - pos) - get_ang(p2 - pos), get_ang(p2 - pos) - get_ang(p3 - pos)
        op_a1, op_a2 = norm_ang(op_a1), norm_ang(op_a2)
        tmp_dis = (a1 - op_a1) ** 2 + (a2 - op_a2) ** 2
        if tmp_dis < min_dis:
            min_dis = tmp_dis
            min_fa = np.array([(a1 - op_a1), (a2 - op_a2), 0])
        if tmp_dis > max_dis:
            max_dis = tmp_dis
            max_fa = np.array([(a1 - op_a1), (a2 - op_a2), 0])
    change_basic = np.concatenate([[min_fa], [max_fa], [np.array([0, 0, 1])]], axis=0)
    change_basic = np.linalg.inv(change_basic.T)
    a1, a2 = get_ang(p1 - p) - get_ang(p2 - p), get_ang(p2 - p) - get_ang(p3 - p)
    a1, a2 = norm_ang(a1), norm_ang(a2)
    centure = np.array([a1, a2, 0])
    generate_points = []
    generate_angs = []
    policy_vecs_beg = []
    for j in generate_rates:
        for i in np.linspace(0, 2 * math.pi, generate_num + 1)[:-1]:
            pos = p + Point(j * math.cos(i), j * math.sin(i))
            a1, a2 = get_ang(p1 - pos) - get_ang(p2 - pos), get_ang(p2 - pos) - get_ang(p3 - pos)
            a1, a2 = norm_ang(a1), norm_ang(a2)
            change_ang = np.matmul(change_basic, (np.array([a1, a2, 0]) - centure).T)
            generate_angs.append(change_ang)
            generate_points.append(pos)
            if j != 0:
                vec = generate_points[-generate_num - 1] - pos
                vec_d = dis(vec, Point(0, 0)) * 0.8
                if j < 1.1:
                    vec_d *= 0.8
                if j < 0.51:
                    vec_d *= 0.5
                vec_ang = get_ang(vec) - get_ang(p2 - pos)
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
    return generate_points, generate_angs, policy_vecs_beg, pcd_tree, (change_basic, centure)


def get_move_vec(policy_vecs, pcd_tree, angles, p0_angle, query, self_num):
    if self_num % 2 == 0:
        a1, a2 = angles
    else:
        a2, a1 = angles
    change_ang = np.matmul(query[0], (np.array([a1, a2, 0]) - query[1]).T)
    [k, idx, pdis] = pcd_tree.search_knn_vector_3d(change_ang, near_choose)
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


def get_top_policy(policy, angle):
    a1, a2 = angle
    a0 = math.fabs(a1 - a2)
    pos = np.searchsorted(angel_set_v, a0)
    if pos == 0:
        d = policy[0]
    elif pos == len(angel_set_v):
        d = policy[-1]
    else:
        d = policy[pos - 1] * (a0 - angel_set_v[pos - 1]) + policy[pos] * (angel_set_v[pos] - a0)
    ang = (a1 + a2) / 2
    if ang > math.pi / 2:
        ang -= math.pi / 2
    return Point(d * math.cos(ang), d * math.sin(ang))



def calc_loss(policy, data_use):
    data = copy.deepcopy(data_use)
    top_flys = [0, 10, 14]
    policy_top, policy_in = policy
    for i in range(30):
        for j in range(len(data)):
            if j in top_flys:
                pos_j = top_flys.index(j)
                p1, p2 = data[top_flys[pos_j - 1]], data[top_flys[pos_j - 2]]
                a1, a2 = get_ang(p1 - data[j]), get_ang(p2 - data[j])
                vec = get_top_policy(policy_top, (a1, a2))
            else:
                p1, p2, p3 = (data[t - 1] for t in flys_types[j][1])
                a1, a2 = get_ang(p1 - data[j]) - get_ang(p2 - data[j]), get_ang(p2 - data[j]) - get_ang(p3 - data[j])
                a1, a2 = norm_ang(a1), norm_ang(a2)
                policy = policy_in[flys_types[j][0] - 1]
                vec = get_move_vec(policy[0], query_info[policy[1]][0], (a1, a2), get_ang(p2 - data[j]),
                                   query_info[policy[1]][1], flys_types[j][2])
            data[j] = data[j] + Point(*vec)
    r = 0
    for side in sides:
        r += dis(data[side[0] - 1], data[side[1] - 1])
    r /= len(sides)
    err = 0
    for side in sides:
        err += math.fabs(dis(data[side[0] - 1], data[side[1] - 1]) - r)
    err /= len(sides) * r
    return err


def alldata_loss(args):
    data_use, policy = args
    data = copy.deepcopy(data_use)
    err = 0
    for d in data:
        err += calc_loss(policy, d)
    return err, policy



std_flys = [
        (200, 0),
        (175, 25 * math.sqrt(3)),
        (150, 0),
        (150, 50 * math.sqrt(3)),
        (125, 25 * math.sqrt(3)),
        (100, 0),
        (125, 75 * math.sqrt(3)),
        (100, 50 * math.sqrt(3)),
        (75, 25 * math.sqrt(3)),
        (50, 0),
        (100, 100 * math.sqrt(3)),
        (75, 75 * math.sqrt(3)),
        (50, 50 * math.sqrt(3)),
        (25, 25 * math.sqrt(3)),
        (0, 0),
    ]
std_flys = [Point(*i) for i in std_flys]
flys_types = [
        [0, ],
        [1, (1, 15, 11), 0],
        [1, (15, 11, 1), 1],
        [2, (1, 15, 11), 0],
        [3, (1, 15, 11), 0],
        [2, (15, 11, 1), 0],
        [1, (1, 15, 11), 1],
        [3, (11, 1, 15), 0],
        [3, (15, 11, 1), 0],
        [1, (15, 11, 1), 0],
        [0, ],
        [1, (11, 1, 15), 0],
        [2, (11, 1, 15), 0],
        [1, (11, 1, 15), 1],
        [0, ]
    ]
sides = [
        (1, 2),
        (2, 3),
        (1, 3),
        (2, 4),
        (4, 5),
        (5, 2),
        (5, 3),
        (5, 6),
        (6, 3),
        (7, 4),
        (7, 8),
        (8, 4),
        (8, 5),
        (8, 9),
        (9, 5),
        (9, 6),
        (9, 10),
        (10, 6),
        (11, 7),
        (11, 12),
        (12, 7),
        (12, 8),
        (12, 13),
        (13, 8),
        (13, 8),
        (13, 14),
        (14, 9),
        (14, 10),
        (14, 15),
        (15, 10),
    ]

query_info = []
angel_set_v = []

p1 = Point(25, 0)
p2 = Point(-25, 0)
p = Point(0, 25 * math.sqrt(3))
point_set_v = []
policy_vec_beg_top = []
for j in range(len(generate_rates) - 1, -1, -1):
    # 小角
    pos = p + Point(0, generate_rates[j])
    point_set_v.append(pos)
    a = get_ang(p1 - pos) - get_ang(p2 - pos)
    angel_set_v.append(a)
    if j == 0:
        policy_vec_beg_top.append(0)
    else:
        policy_vec_beg_top.append(generate_rates[j] - generate_rates[j - 1])
for j in range(len(generate_rates)):
    # 大角
    if j != 0:
        pos = p - Point(0, generate_rates[j])
        point_set_v.append(pos)
        a = get_ang(p1 - pos) - get_ang(p2 - pos)
        angel_set_v.append(a)
        policy_vec_beg_top.append(generate_rates[j - 1] - generate_rates[j])
angel_set_v = np.array(angel_set_v)
policy_vec_beg_top = np.array(policy_vec_beg_top)

p1 = Point(200, 0)
p2 = Point(0, 0)
p3 = Point(100, 100*math.sqrt(3))
p_pos = [Point(175, 25*math.sqrt(3)), Point(150, 50*math.sqrt(3)), Point(125, 25*math.sqrt(3))]
p_policys = []
for p in range(len(p_pos)):
    _, _, policy_vec, pcd_tree, query = get_policy(p1, p2, p3, p_pos[p])
    query_info.append((pcd_tree, query))
    p_policys.append((policy_vec, p))

if __name__ == '__main__':
    dataset = []
    for i in range(dataset_num):
        data = []
        for fly in std_flys:
            data.append(fly + Point((random.randint(0, 1) * 2 - 1) * random.randint(0, max_pos_err * 10) / 10,
                                    (random.randint(0, 1) * 2 - 1) * random.randint(0, max_pos_err * 10) / 10))
        dataset.append(data)

 
    all_policy = [(policy_vec_beg_top, p_policys) ]
    while True:
        for _ in range(people_num):
            policy_change = copy.deepcopy(random.choice(all_policy)[1])
            for i in random.sample(range(len(policy_change[0][0])), random.randint(0, change_num)):
                change_point = random.randint(0, len(p_pos) - 1)
                if random.randint(0, 6) == 1:
                    # 变异
                    if random.randint(0, 1) == 1:
                        # 改角度
                        d = dis(Point(0, 0), Point(*policy_change[change_point][0][i]))
                        ang = random.randint(0, 1000000)
                        policy_change[change_point][0][i][0] = d * math.cos(ang)
                        policy_change[change_point][0][i][1] = d * math.sin(ang)
                    else:
                        # 改长度
                        d = dis(Point(0, 0), Point(*policy_change[change_point][0][i]))
                        ang = get_ang(Point(*policy_change[change_point][0][i]))
                        d *= random.randint(3000, 8000) / 5000
                        policy_change[change_point][0][i][0] = d * math.cos(ang)
                        policy_change[change_point][0][i][1] = d * math.sin(ang)
                else:
                    # 杂交
                    hooker = random.choice(all_policy)[1]
                    policy_change[change_point][0][i][0] = hooker[change_point][0][i][0]
                    policy_change[change_point][0][i][1] = hooker[change_point][0][i][1]
            all_policy.append((policy_vec_beg_top, policy_change))

        data_use = random.sample(dataset, calc_loss_num)
        all_loss = []
        # for ans in mp.Pool(processes=12).imap(alldata_loss, [(data_use, p) for p in all_policy]):
        #     all_loss.append(ans)
        for t in [(data_use, p) for p in all_policy]:
            ans = alldata_loss(t)
            all_loss.append(ans)
        all_loss.sort(key=lambda x: x[0])
        all_loss = all_loss[: people_num // 2]
        print(all_loss[0][0])
        pickle.dump(all_loss[0][1], open('policy.pkl', 'wb'))
        all_policy = [i[1] for i in all_loss]
        del all_loss