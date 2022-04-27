import os
import numpy as np


def coverage():
    report_path = 'report_coverage.txt'
    file_list = os.listdir('EMNLP22-res')
    file_list.sort()
    context = np.load('golden_dataset/sys_dialog_texts.test.npy')
    context = [' '.join(x) for x in context]
    for filename in file_list:
        with open(filename, 'r') as f:
            lines = f.readlines()
        lines = [l.strip() for l in lines]
