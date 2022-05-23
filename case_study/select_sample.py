import textwrap
import numpy as np
import json
import random
import os

role = {1: "Bot", 0: "User"}

bot_role = ('MoEL'.ljust(8), 'MIME'.ljust(8), 'EmpDG'.ljust(8),
            'RecEC'.ljust(8), 'CEM'.ljust(8), 'KEMP'.ljust(8),
            'GEE'.ljust(8), 'GREmp'.ljust(8), 'ablation'.ljust(8))
index2model = {0: 'MoEL', 1: 'MIME', 2: 'EmpDG', 3: 'RecEC', 4: 'CEM', 5: 'KEMP', 6: 'GEE', 7: 'GREmp', 8: 'ablation'}
model2index = {'MoEL': 0, 'MIME': 1, 'EmpDG': 2, 'RecEC': 3, 'CEM': 4, 'KEMP': 5, 'GEE': 6, 'GREmp': 7, 'ablation': 8}
prefix = {'MoEL': '-42', 'MIME': '-1024', 'EmpDG': '-4096', 'RecEC': '-1024', 'CEM': '-1024', 'KEMP': '-0',
          'GEE': '_MIME-1024', 'GREmp': '-0', 'ablation': '-42'}
sample_path = 'EMNLP22-sample'
input_path = 'EMNLP22-res'


def format_file(contexts, emotions, responses, selected_index, file_name):
    with open(file_name, 'w') as f:
        for idx, (context, emotion, response, real_index) in enumerate(
                zip(contexts, emotions, responses, selected_index)):
            f.write('************* Sample %d (real index: %d)\n' % (idx + 1, real_index))
            f.write('************* Emotion: %s\n' % emotion)
            for role_idx, sentence in enumerate(context):
                f.write('{}: {}\n'.format(role[role_idx % 2], ' '.join(sentence)))
            f.write('\n')
            for reply_idx, reply in enumerate(response):
                f.write('{}: {}\n'.format(bot_role[reply_idx], reply))
            f.write('\n')


def main(sample=-1, ab=False):
    context = np.load('golden_dataset/sys_dialog_texts.test.npy', allow_pickle=True)
    emotion = np.load('golden_dataset/sys_emotion_texts.test.npy', allow_pickle=True)
    res_list = []
    for key, value in index2model.items():
        with open(os.path.join(input_path, '%s%s.txt' % (value, prefix[value])), 'r') as f:
            res_list.append([line.strip() for line in f.readlines()])
    all_response = [x for x in zip(*res_list)]
    if not os.path.exists(sample_path):
        os.mkdir(sample_path)
    if sample > 0:
        if os.path.exists(os.path.join(sample_path, '128-indexes.txt')):
            with open(os.path.join(sample_path, '128-indexes.txt'), 'r') as f:
                selected_index = f.readlines()
            selected_index = [int(x) for x in selected_index]
        else:
            selected_index = random.sample([i for i in range(len(context))], 128)
        selected_context, selected_emotion, selected_response = [], [], []

        for idx in selected_index:
            selected_context.append(context[idx])
            selected_emotion.append(emotion[idx])
            selected_response.append(all_response[idx])

        with open(os.path.join(sample_path, '128-indexes.txt'), 'w') as f:
            for i in selected_index:
                f.write('{}\n'.format(str(i)))
        format_file(selected_context, selected_emotion, selected_response, selected_index,
                    os.path.join(sample_path, '128-samples.txt'))
    else:
        format_file(context, emotion, all_response, [i for i in range(len(context))],
                    os.path.join(sample_path, 'all-samples.txt'))


def interactive():
    if os.path.exists(os.path.join(sample_path, 'score.json')):
        store_dict = json.load(open(os.path.join(sample_path, 'score.json')))
    else:
        store_dict = {}
        for key, value in model2index.items():
            store_dict[key] = {'Empathy': [], 'Relevance': [], 'Fluency': []}

    counter = len(store_dict['MoEL']['Empathy'])

    with open(os.path.join(sample_path, '128-samples.txt'), 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    pack, package = [], []
    for line in lines:
        if line.startswith('************* Sample'):
            package.append(tuple(pack))
            pack = [line]
        else:
            pack.append(line)
    package.append(pack)
    package = package[1:]

    try:
        for pack_id in range(counter, len(package)):
            output = ''
            empathy_list, relevance_list, fluency_list = [0] * 9, [0] * 9, [0] * 9
            one = package[pack_id]
            start = len(one) - 10
            end = len(one) - 1
            for i in range(start):
                output += textwrap.fill(one[i], 150) + '\n'
            random_index = [x for x in range(start, end)]
            random.shuffle(random_index)
            # print('{}{}'.format(i, one[idx][10:]))
            response_dict = {}
            for i, idx in enumerate(random_index):
                if one[idx][9:] in response_dict.keys():
                    empathy = empathy_list[response_dict[one[idx][9:]]]
                    relevance = relevance_list[response_dict[one[idx][9:]]]
                    fluency = fluency_list[response_dict[one[idx][9:]]]
                else:
                    response_dict[one[idx][9:]] = i
                    print(output)
                    for x, jdx in enumerate(random_index):
                        if x < i:
                            print('{:<2}{:<75} **Score: {} {} {}'.format(x, one[jdx][9:], empathy_list[x],
                                                                                  relevance_list[x], fluency_list[x]))
                        else:
                            print('{:<2}{}'.format(x, one[jdx][9:]))
                    print()
                    while True:
                        print('Response {}:{}'.format(i, one[idx][9:]))
                        empathy = input('Empathy 1~5: ')
                        while not empathy.isdigit() or int(empathy) < 1 or int(empathy) > 5:
                            print('Empathy should be an integer in the range 1~5.', sep=' ')
                            empathy = input('Empathy 1~5: ')

                        relevance = input('Relevance 1~5: ')
                        while not relevance.isdigit() or int(relevance) < 1 or int(relevance) > 5:
                            print('Relevance should be an integer in the range 1~5.', sep=' ')
                            relevance = input('Relevance 1~5: ')

                        fluency = input('Fluency 1~5: ')
                        while not fluency.isdigit() or int(fluency) < 1 or int(fluency) > 5:
                            print('Fluency should be an integer in the range 1~5.', sep=' ')
                            fluency = input('Fluency 1~5: ')

                        again = input('Do you want to regrade? (Y/N)')
                        if again == 'Y':
                            continue
                        else:
                            break

                empathy_list[i] = int(empathy)
                relevance_list[i] = int(relevance)
                fluency_list[i] = int(fluency)

            for i, idx in enumerate(random_index):
                store_dict[index2model[idx - start]]['Empathy'].append(empathy_list[i])
                store_dict[index2model[idx - start]]['Relevance'].append(relevance_list[i])
                store_dict[index2model[idx - start]]['Fluency'].append(fluency_list[i])

    except KeyboardInterrupt:
        with open(os.path.join(sample_path, 'score.json'), 'w') as f:
            json.dump(store_dict, f, indent=4)

    with open(os.path.join(sample_path, 'score.json'), 'w') as f:
        json.dump(store_dict, f, indent=4)


def compute_score(ab=False):
    empathy, relevance, fluency = [], [], []
    if ab:
        store_dict = json.load(open(os.path.join(sample_path, 'score.json')))
        for i in range(3):
            empathy.append(round(np.mean(store_dict[index2model[i + 11]]['Empathy']), 2))
            relevance.append(round(np.mean(store_dict[index2model[i + 11]]['Relevance']), 2))
            fluency.append(round(np.mean(store_dict[index2model[i + 11]]['Fluency']), 2))

        for i in range(3):
            print(
                '{}: {} & {} & {}'.format(index2model[i + 11], empathy[i], relevance[i], fluency[i]))
    else:
        store_dict = json.load(open(os.path.join(sample_path, 'score.json')))
        for i in range(11):
            empathy.append(round(np.mean(store_dict[index2model[i]]['Empathy']), 2))
            relevance.append(round(np.mean(store_dict[index2model[i]]['Relevance']), 2))
            fluency.append(round(np.mean(store_dict[index2model[i]]['Fluency']), 2))

        for i in range(11):
            print(
                '{}: {} & {} & {}'.format(index2model[i], empathy[i], relevance[i], fluency[i]))


def AB_test():
    scores_dict = json.load(open(os.path.join(sample_path, 'score.json')))
    with open(os.path.join(sample_path, '128-samples.txt'), 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    if os.path.exists(os.path.join(sample_path, 'ABtest.json')):
        store_dict = json.load(open(os.path.join(sample_path, 'ABtest.json')))
    else:
        store_dict = {
            'counter': 0,
            'MoEL': {'EmoRG win': 0, 'EmoRG lost': 0, 'Tie': 0},
            'MIME': {'EmoRG win': 0, 'EmoRG lost': 0, 'Tie': 0},
            'EmpDG': {'EmoRG win': 0, 'EmoRG lost': 0, 'Tie': 0},
            'GREC': {'EmoRG win': 0, 'EmoRG lost': 0, 'Tie': 0},
            'GEE': {'EmoRG win': 0, 'EmoRG lost': 0, 'Tie': 0}}
    pack, package = [], []
    for line in lines:
        if line.startswith('************* Sample'):
            package.append(tuple(pack))
            pack = [line]
        else:
            pack.append(line)
    package.append(pack)
    package = package[1:]
    counter = store_dict['counter']
    all_aspect = ['Empathy', 'Relevance', 'Fluency', 'ER', 'IP', 'EX']

    try:
        for pack_id in range(counter, 120):
            winlist = ['', '', '', '', '']
            one = package[pack_id]
            start = len(one) - 12
            output = ""
            for i in range(start):
                output += one[i] + '\n'
            print(output)
            # MoEL, MIMI, EmpDG, GREC, GEE
            # compare the scores:
            moel_score = [scores_dict['MoEL'][x][pack_id] for x in all_aspect]
            mime_score = [scores_dict['MIME'][x][pack_id] for x in all_aspect]
            empdg_score = [scores_dict['EmpDG'][x][pack_id] for x in all_aspect]
            grec_score = [scores_dict['GREC'][x][pack_id] for x in all_aspect]
            gee_score = [scores_dict['GEE'][x][pack_id] for x in all_aspect]
            ec_trs_score = [scores_dict['EC-TRS'][x][pack_id] for x in all_aspect]
            ec_blender_score = [scores_dict['EC-blender'][x][pack_id] for x in all_aspect]

            counts = []
            for s1, s2 in zip(moel_score, ec_trs_score):
                if s1 > s2:
                    counts.append(1)
                elif s1 < s2:
                    counts.append(-1)
            if len(counts) == 0:
                winlist[0] = 'Tie'
            elif not (-1 in counts):
                winlist[0] = 'GREmp lost'
            elif not (1 in counts):
                winlist[0] = 'GREmp win'

            counts = []
            for s1, s2 in zip(mime_score, ec_trs_score):
                if s1 > s2:
                    counts.append(1)
                elif s1 < s2:
                    counts.append(-1)
            if len(counts) == 0:
                winlist[1] = 'Tie'
            elif not (-1 in counts):
                winlist[1] = 'EmoRG lost'
            elif not (1 in counts):
                winlist[1] = 'EmoRG win'

            counts = []
            for s1, s2 in zip(empdg_score, ec_trs_score):
                if s1 > s2:
                    counts.append(1)
                elif s1 < s2:
                    counts.append(-1)
            if len(counts) == 0:
                winlist[2] = 'Tie'
            elif not (-1 in counts):
                winlist[2] = 'EmoRG lost'
            elif not (1 in counts):
                winlist[2] = 'EmoRG win'

            counts = []
            for s1, s2 in zip(grec_score, ec_trs_score):
                if s1 > s2:
                    counts.append(1)
                elif s1 < s2:
                    counts.append(-1)
            if len(counts) == 0:
                winlist[3] = 'Tie'
            elif not (-1 in counts):
                winlist[3] = 'EmoRG lost'
            elif not (1 in counts):
                winlist[3] = 'EmoRG win'

            counts = []
            for s1, s2 in zip(gee_score, ec_blender_score):
                if s1 > s2:
                    counts.append(1)
                elif s1 < s2:
                    counts.append(-1)
            if len(counts) == 0:
                winlist[4] = 'Tie'
            elif not (-1 in counts):
                winlist[4] = 'EmoRG lost'
            elif not (1 in counts):
                winlist[4] = 'EmoRG win'

            indexlist = ['0', '1']
            if '' in winlist[:-1]:
                ec_trs_response = one[start + model2index['EC-TRS']][12:]
                if winlist[0] == '':
                    moel_response = one[start + model2index['MoEL']][12:]
                    random.shuffle(indexlist)
                    if indexlist[0] == '1':
                        print(moel_response)
                        print(ec_trs_response)
                    else:
                        print(ec_trs_response)
                        print(moel_response)
                    better = input("which is better 0 or 1, 2 for Tie?")
                    while better not in ['0', '1', '2']:
                        better = input("which is better 0 or 1, 2 for Tie?")
                    if better == '2':
                        winlist[0] = 'Tie'
                    elif better == indexlist[0]:
                        winlist[0] = 'EmoRG win'
                    else:
                        winlist[0] = 'EmoRG lost'
                    print()
                if winlist[1] == '':
                    mime_response = one[start + model2index['MIME']][12:]
                    random.shuffle(indexlist)
                    if indexlist[0] == '1':
                        print(mime_response)
                        print(ec_trs_response)
                    else:
                        print(ec_trs_response)
                        print(mime_response)
                    better = input("which is better 0 or 1, 2 for Tie?")
                    while better not in ['0', '1', '2']:
                        better = input("which is better 0 or 1, 2 for Tie?")
                    if better == '2':
                        winlist[1] = 'Tie'
                    elif better == indexlist[0]:
                        winlist[1] = 'EmoRG win'
                    else:
                        winlist[1] = 'EmoRG lost'
                    print()
                if winlist[2] == '':
                    empdg_response = one[start + model2index['EmpDG']][12:]
                    random.shuffle(indexlist)
                    if indexlist[0] == '1':
                        print(empdg_response)
                        print(ec_trs_response)
                    else:
                        print(ec_trs_response)
                        print(empdg_response)
                    better = input("which is better 0 or 1, 2 for Tie?")
                    while better not in ['0', '1', '2']:
                        better = input("which is better 0 or 1, 2 for Tie?")
                    if better == '2':
                        winlist[2] = 'Tie'
                    elif better == indexlist[0]:
                        winlist[2] = 'EmoRG win'
                    else:
                        winlist[2] = 'EmoRG lost'
                    print()
                if winlist[3] == '':
                    grec_response = one[start + model2index['GREC']][12:]
                    random.shuffle(indexlist)
                    if indexlist[0] == '1':
                        print(grec_response)
                        print(ec_trs_response)
                    else:
                        print(ec_trs_response)
                        print(grec_response)
                    better = input("which is better 0 or 1, 2 for Tie?")
                    while better not in ['0', '1', '2']:
                        better = input("which is better 0 or 1, 2 for Tie?")
                    if better == '2':
                        winlist[3] = 'Tie'
                    elif better == indexlist[0]:
                        winlist[3] = 'EmoRG win'
                    else:
                        winlist[3] = 'EmoRG lost'
                    print()
            if winlist[-1] == '':
                ec_blender_response = one[start + model2index['EC-blender']][12:]
                gee_response = one[start + model2index['GEE']][12:]
                random.shuffle(indexlist)
                if indexlist[0] == '1':
                    print(gee_response)
                    print(ec_blender_response)
                else:
                    print(ec_blender_response)
                    print(gee_response)
                better = input("which is better 0 or 1, 2 for Tie?")
                while better not in ['0', '1', '2']:
                    better = input("which is better 0 or 1, 2 for Tie?")
                if better == '2':
                    winlist[4] = 'Tie'
                elif better == indexlist[0]:
                    winlist[4] = 'EmoRG win'
                else:
                    winlist[4] = 'EmoRG lost'
                print()

            store_dict['counter'] += 1
            store_dict['MoEL'][winlist[0]] += 1
            store_dict['MIME'][winlist[1]] += 1
            store_dict['EmpDG'][winlist[2]] += 1
            store_dict['GREC'][winlist[3]] += 1
            store_dict['GEE'][winlist[4]] += 1

    except KeyboardInterrupt:
        with open('./sample/ABtest.json', 'w') as f:
            json.dump(store_dict, f, indent=4)

    with open('./sample/ABtest.json', 'w') as f:
        json.dump(store_dict, f, indent=4)

def randomRes():
    with open(os.path.join(sample_path, '128-samples.txt'), 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    pack, package = [], []
    for line in lines:
        if line.startswith('************* Sample'):
            package.append(tuple(pack))
            pack = [line]
        else:
            pack.append(line)
    package.append(pack)
    package = package[1:]
    model_order = open(os.path.join(sample_path, 'model_order.txt'), 'w')
    random_res = open(os.path.join(sample_path, 'random_128.txt'), 'w')

    for pack_id in range(len(package)):
        output = ''
        one = package[pack_id]
        start = len(one) - 10
        end = len(one) - 1
        for i in range(start):
            output += textwrap.fill(one[i], 150) + '\n'
        random_index = [x for x in range(start, end)]
        random.shuffle(random_index)
        random_res.write(output)
        order = ''
        for i, idx in enumerate(random_index):
            order += (one[idx][:8]).strip()
            order += ' '
            random_res.write('Response %d: ' % i)
            random_res.write(one[idx][10:])
            random_res.write('\n***Score %d: \n' % i)
        model_order.write(order)
        model_order.write('\n')
        random_res.write('\n')

    model_order.close()
    random_res.close()


if __name__ == '__main__':
    # main(128)
    # interactive()
    randomRes()
