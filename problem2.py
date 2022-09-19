from sympy import *
from collections import namedtuple
import math
import matplotlib.pyplot as plt
import random
import time

Point = namedtuple('Point', ['x', 'y'])

def addPoint(a: Point, b: Point):
    return Point(a.x + b.x, a.y + b.y)

Point.__add__ = addPoint

def subPoint(a: Point, b: Point):
    return Point(a.x - b.x, a.y - b.y)

Point.__sub__ = subPoint

def averPoint(points: list[Point]):
    l = len(points)
    x, y = 0, 0
    for i in range(l):
        x += points[i].x
        y += points[i].y
    x /= l
    y /= l
    return Point(x, y)

def divf(x):
    if x == 0:
        return 1e-7
    return x

def dis(a: Point, b: Point):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

def get_ang(p: Point):
    ang = math.atan(p.y / divf(p.x))
    if p.x < 0:
        ang += math.pi
    if ang < 0:
        ang += 2 * math.pi
    return ang

def get_ang3(p1: Point, p2: Point, p3: Point):
    a1 = get_ang(Point(p1.x - p2.x, p1.y - p2.y))
    a2 = get_ang(Point(p3.x - p2.x, p3.y - p2.y))
    return math.fabs(a2 - a1)

def roatate(p: Point, theta):
    d = dis(p, Point(0, 0))
    theta0 = get_ang(p)
    return Point(d * math.cos(theta + theta0), d * math.sin(theta + theta0))

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

r = 100
flys = [Point(0, 0)]
for i in range(9):
    ang = i * 40 / 180 * math.pi
    flys.append(Point(r * math.cos(ang), r * math.sin(ang)))


def geta0ap1ap2(p1, p2):  # p2 > p1
    a0 = get_ang(flys[0] - target)
    ap1 = get_ang(flys[p1] - target)
    ap2 = get_ang(flys[p2] - target)
    return a0, ap1, ap2

while True:
    # target = Point(random.randint(-200, 200), random.randint(-200, 200))
    sel_p = random.choice(flys[1:])
    target = sel_p + Point((random.randint(0, 2) * 2 - 1) * random.randint(80, 120) / 10,
                           (random.randint(0, 2) * 2 - 1) * random.randint(80, 120) / 10)
    p1 = 1
    p2rand = random.sample(range(2, 9), 2)
    p2rand = sorted(p2rand)
    std_a0, std_ap1, std_ap2 = geta0ap1ap2(p1, p2rand[0])
    _, _, std_ap3 = geta0ap1ap2(p1, p2rand[1])
    print(target, p2rand)
    ans, ans_dis = None, 1e15
    ans_p1, ans_p2 = None, None
    ans_a1, ans_b1 = None, None
    ans_a2, ans_b2 = None, None
    for i in range(2, 9):
        for j in range(i + 1, 9):
            ans1, a1, b1 = get_ans(std_a0, std_ap1, std_ap2, i - p1)
            ans2, a2, b2 = get_ans(std_a0, std_ap1, std_ap3, j - p1)
            tmp_dis = dis(ans1, ans2)
            if tmp_dis < ans_dis:
                ans_dis = tmp_dis
                ans = averPoint([ans1, ans2])
                ans_p1, ans_p2 = i, j
                ans_a1, ans_b1 = a1, b1
                ans_a2, ans_b2 = a2, b2

    for fly in flys:
        plt.scatter(*fly, s=20, c='blue')
    if ans is not None:
        plt.scatter(*ans, s=40, c='red')
    plt.scatter(*target, s=20, c='green')
    plt.scatter(*flys[1], s=20, c='yellow')
    plt.scatter(*flys[p2rand[0]], s=20, c='yellow')
    plt.scatter(*flys[p2rand[1]], s=20, c='yellow')
    plt.plot([0, flys[p1].x], [0, flys[p1].y], 'b:')
    plt.plot([0, flys[ans_p1].x], [0, flys[ans_p1].y], 'b:')
    plt.plot([0, flys[ans_p2].x], [0, flys[ans_p2].y], 'b:')
    draw_circle = plt.Circle(ans_a1[0], ans_a1[1], fill=False)
    plt.gcf().gca().add_artist(draw_circle)
    draw_circle = plt.Circle(ans_b1[0], ans_b1[1], fill=False)
    plt.gcf().gca().add_artist(draw_circle)
    draw_circle = plt.Circle(ans_a2[0], ans_a2[1], fill=False)
    plt.gcf().gca().add_artist(draw_circle)
    draw_circle = plt.Circle(ans_b2[0], ans_b2[1], fill=False)
    plt.gcf().gca().add_artist(draw_circle)
    plt.axis('equal')
    # plt.show()
    plt.pause(2)
    plt.clf()