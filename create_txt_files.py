#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import os.path


def get_files_list(dir):
    """

    :param dir: 指定文件夹目录
    :return: 包含所有文件的列表
    """
    files_list = []
    for parent, dirnames, filenames in os.walk(dir):   # os.walk()遍历dir文件夹
        for filename in filenames:
            current_file = parent.split('/')[-1]
            if current_file == 'flower':
                label = 0
            elif current_file == 'guitar':
                label = 1
            elif current_file == 'animal':
                label = 2
            elif current_file == 'houses':
                label = 3
            elif current_file == 'plane':
                label = 4
            files_list.append([os.path.join(current_file, filename), label])
    return files_list


def write_txt_file(content, filename, mode='w'):
    """

    :param content: 需要保存的数据
    :param filename: 保存的文件名
    :param mode:
    :return:
    """
    with open(filename, mode) as f:
        for line in content:
            str_line = ""       # 这里不是空格符，只是表示str_line为string型
            for col, data in enumerate(line):   # enumerate列出数据下标和数据[(0, '/jpg'), (1, 0)]
                if not col == len(line) - 1:
                    # 以空格作为分隔符
                    str_line = str_line + str(data) + " "
                else:
                    # 每行最后一个数据用换行符“\n”
                    str_line = str_line + str(data) + "\n"
            f.write(str_line)


if __name__ == '__main__':
    train_dir = 'dataset/train'
    train_txt = 'dataset/train.txt'
    train_data = get_files_list(train_dir)
    write_txt_file(train_data, train_txt)

    val_dir = 'dataset/val'
    val_txt = 'dataset/val.txt'
    val_data = get_files_list(val_dir)
    write_txt_file(val_data, val_txt)


