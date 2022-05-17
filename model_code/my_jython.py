# -*- coding: utf-8 -*-
import random


def call_module(query, num):
    tl = [[] for i in range(num)]
    for i in range(num):
        tl[i].append(str(query) + ' ' + str(i))
        tl[i].append(str(random.randint(0, 100)))
    return tl
