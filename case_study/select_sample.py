import numpy as np
import json
import random
import os

role = {1: "Bot", 0: "User"}

bot_role = ('TRS'.ljust(5), 'MoEL'.ljust(5), 'MIME'.ljust(5), 'EmpDG'.ljust(5),
            'RecEC'.ljust(5), 'CEM'.ljust(5), 'KEMP'.ljust(5), 'GEE'.ljust(5), 'GREmp'.ljust(5))
index2model = {0: 'TRS', 1: 'MoEL', 2: 'MIME', 3: 'EmpDG', 4: 'RecEC', 5: 'CEM', 6: 'KEMP', 7: 'GEE', 8: 'GREmp'}
model2index = {'TRS': 0, 'MoEL': 1, 'MIME': 2, 'EmpDG': 3, 'RecEC': 4, 'CEM': 5, 'KEMP': 6, 'GEE': 7, 'GREmp': 8}
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
        with open(os.path.join(input_path, '%s.txt' % value), 'r') as f:
            res_list.append([line.strip() for line in f.readlines()])
    all_response = [x for x in zip(*res_list)]
    if not os.path.exists(sample_path):
        os.mkdir(sample_path)
    if sample > 0:
        selected_index = random.sample([i for i in range(len(context))], 128)
        selected_context, selected_emotion, selected_response = [], []

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
    with open(os.path.join(sample_path, '128-emotion.txt'), 'r') as f:
        emotion = [x.strip() for x in f.readlines()]

    if os.path.exists(os.path.join(sample_path, 'score.json')):
        store_dict = json.load(open(os.path.join(sample_path, 'score.json')))
    else:
        store_dict = {'TRS': {'Empathy': [], 'Relevance': [], 'Fluency': [], 'ER': [], 'IP': [], 'EX': []},
                      'MoEL': {'Empathy': [], 'Relevance': [], 'Fluency': [], 'ER': [], 'IP': [], 'EX': []},
                      'MIME': {'Empathy': [], 'Relevance': [], 'Fluency': [], 'ER': [], 'IP': [], 'EX': []},
                      'EmpDG': {'Empathy': [], 'Relevance': [], 'Fluency': [], 'ER': [], 'IP': [], 'EX': []},
                      'GREC': {'Empathy': [], 'Relevance': [], 'Fluency': [], 'ER': [], 'IP': [], 'EX': []},
                      'EC-TRS': {'Empathy': [], 'Relevance': [], 'Fluency': [], 'ER': [], 'IP': [], 'EX': []},
                      'BlenderBOT': {'Empathy': [], 'Relevance': [], 'Fluency': [], 'ER': [], 'IP': [], 'EX': []},
                      'GEE': {'Empathy': [], 'Relevance': [], 'Fluency': [], 'ER': [], 'IP': [], 'EX': []},
                      'EC-blender': {'Empathy': [], 'Relevance': [], 'Fluency': [], 'ER': [], 'IP': [], 'EX': []},
                      'BART': {'Empathy': [], 'Relevance': [], 'Fluency': [], 'ER': [], 'IP': [], 'EX': []},
                      'EC-bart': {'Empathy': [], 'Relevance': [], 'Fluency': [], 'ER': [], 'IP': [], 'EX': []},
                      }
    counter = len(store_dict['TRS']['Empathy'])

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
    assert len(package) == len(emotion)

    try:
        for pack_id in range(counter, len(package)):
            # print(emotion[pack_id])
            output = emotion[pack_id] + '\n'
            empathy_list, relevance_list, fluency_list, er_list, ip_list, ex_list = [0] * 11, [0] * 11, [0] * 11, [
                0] * 11, [0] * 11, [0] * 11
            one = package[pack_id]
            start = len(one) - 12
            end = len(one) - 1
            for i in range(start):
                output += one[i] + '\n'
            # print(one[i])
            random_index = [x for x in range(start, end)]
            random.shuffle(random_index)
            # print('{}{}'.format(i, one[idx][10:]))
            response_dict = {}
            for i, idx in enumerate(random_index):
                if one[idx][10:] in response_dict.keys():
                    empathy = empathy_list[response_dict[one[idx][10:]]]
                    relevance = relevance_list[response_dict[one[idx][10:]]]
                    fluency = fluency_list[response_dict[one[idx][10:]]]
                    er = er_list[response_dict[one[idx][10:]]]
                    ip = ip_list[response_dict[one[idx][10:]]]
                    ex = ex_list[response_dict[one[idx][10:]]]
                else:
                    response_dict[one[idx][10:]] = i
                    print(output)
                    for x, jdx in enumerate(random_index):
                        if x < i:
                            print('{:<2}{:<50} **Score: {} {} {} {} {} {}'.format(x, one[jdx][10:], empathy_list[x],
                                                                                  relevance_list[x], fluency_list[x],
                                                                                  er_list[x], ip_list[x],
                                                                                  ex_list[x]))
                        else:
                            print('{:<2}{}'.format(x, one[jdx][10:]))
                    print()
                    while True:
                        print('Response {}{}'.format(i, one[idx][10:]))
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

                        er = input('ER 0~2: ')
                        while not er.isdigit() or int(er) < 0 or int(er) > 2:
                            print('Emotion reaction should be an integer in the range 0~2.', sep=' ')
                            er = input('ER 0~2: ')

                        ip = input('IP 0~2: ')
                        while not ip.isdigit() or int(ip) < 0 or int(ip) > 2:
                            print('Interpretation should be an integer in the range 0~2.', sep=' ')
                            ip = input('IP 0~2: ')

                        ex = input('EX 0~2: ')
                        while not ex.isdigit() or int(ex) < 0 or int(ex) > 2:
                            print('Explaination should be an integer in the range 0~2.', sep=' ')
                            ex = input('EX 0~2: ')

                        again = input('Do you want to regrade? (Y/N)')
                        if again == 'Y':
                            continue
                        else:
                            break

                empathy_list[i] = int(empathy)
                relevance_list[i] = int(relevance)
                fluency_list[i] = int(fluency)
                er_list[i] = int(er)
                ip_list[i] = int(ip)
                ex_list[i] = int(ex)

            for i, idx in enumerate(random_index):
                store_dict[index2model[idx - start]]['Empathy'].append(empathy_list[i])
                store_dict[index2model[idx - start]]['Relevance'].append(relevance_list[i])
                store_dict[index2model[idx - start]]['Fluency'].append(fluency_list[i])
                store_dict[index2model[idx - start]]['ER'].append(er_list[i])
                store_dict[index2model[idx - start]]['IP'].append(ip_list[i])
                store_dict[index2model[idx - start]]['EX'].append(ex_list[i])

    except KeyboardInterrupt:
        with open(os.path.join(sample_path, 'score.json'), 'w') as f:
            json.dump(store_dict, f, indent=4)

    with open(os.path.join(sample_path, 'score.json'), 'w') as f:
        json.dump(store_dict, f, indent=4)


def compute_score(ab=False):
    empathy, relevance, fluency, ER, IP, EX = [], [], [], [], [], []
    if ab:
        store_dict = json.load(open(os.path.join(sample_path, 'score.json')))
        for i in range(3):
            empathy.append(round(np.mean(store_dict[index2model[i + 11]]['Empathy']), 2))
            relevance.append(round(np.mean(store_dict[index2model[i + 11]]['Relevance']), 2))
            fluency.append(round(np.mean(store_dict[index2model[i + 11]]['Fluency']), 2))
            ER.append(round(np.mean(store_dict[index2model[i + 11]]['ER']), 2))
            IP.append(round(np.mean(store_dict[index2model[i + 11]]['IP']), 2))
            EX.append(round(np.mean(store_dict[index2model[i + 11]]['EX']), 2))

        for i in range(3):
            print(
                '{}: {} & {} & {} & {} & {} & {}'.format(index2model[i + 11], empathy[i], relevance[i], fluency[i],
                                                         ER[i], IP[i],
                                                         EX[i]))
    else:
        store_dict = json.load(open(os.path.join(sample_path, 'score.json')))
        for i in range(11):
            empathy.append(round(np.mean(store_dict[index2model[i]]['Empathy']), 2))
            relevance.append(round(np.mean(store_dict[index2model[i]]['Relevance']), 2))
            fluency.append(round(np.mean(store_dict[index2model[i]]['Fluency']), 2))
            ER.append(round(np.mean(store_dict[index2model[i]]['ER']), 2))
            IP.append(round(np.mean(store_dict[index2model[i]]['IP']), 2))
            EX.append(round(np.mean(store_dict[index2model[i]]['EX']), 2))

        for i in range(11):
            print(
                '{}: {} & {} & {} & {} & {} & {}'.format(index2model[i], empathy[i], relevance[i], fluency[i], ER[i],
                                                         IP[i],
                                                         EX[i]))


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
                winlist[0] = 'EmoRG lost'
            elif not (1 in counts):
                winlist[0] = 'EmoRG win'

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


def ab_interactive():
    with open('./sample/128-emotion.txt', 'r') as f:
        emotion = [x.strip() for x in f.readlines()]

    exist_store_dict = json.load(open(os.path.join(sample_path, 'score.json')))

    if os.path.exists('./sample/ab-score.json'):
        store_dict = json.load(open('./sample/ab-score.json'))
    else:
        store_dict = {'ab-filter': {'Empathy': [], 'Relevance': [], 'Fluency': [], 'ER': [], 'IP': [], 'EX': []},
                      'ab-copy': {'Empathy': [], 'Relevance': [], 'Fluency': [], 'ER': [], 'IP': [], 'EX': []},
                      'ab-msa': {'Empathy': [], 'Relevance': [], 'Fluency': [], 'ER': [], 'IP': [], 'EX': []},
                      }
    counter = len(store_dict['ab-msa']['Empathy'])

    with open('./sample/128-ab-samples.txt', 'r') as f:
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
    assert len(package) == len(emotion)

    score_title = ['Empathy', 'Relevance', 'Fluency', 'ER', 'IP', 'EX']

    try:
        for pack_id in range(counter, 120):
            # print(emotion[pack_id])
            output = emotion[pack_id] + '\n'
            empathy_list, relevance_list, fluency_list, er_list, ip_list, ex_list = [0] * 14, [0] * 14, [0] * 14, [
                0] * 14, [0] * 14, [0] * 14
            one = package[pack_id]
            start = len(one) - 4
            end = len(one) - 1

            exist_responses, exist_scores = [], []
            for idx, response in enumerate(one[len(one) - 15:start]):
                exist_responses.append(response[10:])
                exist_scores.append([exist_store_dict[index2model[idx]][x][pack_id] for x in score_title])

            for i in range(start - 11):
                output += one[i] + '\n'
            for x, i in enumerate(range(start - 11, start)):
                output += '{} {}\n'.format(one[i], exist_scores[x])
            random_index = [x for x in range(start, end)]
            random.shuffle(random_index)
            response_dict = {}

            for i, idx in enumerate(random_index):
                if one[idx][10:] in response_dict.keys():
                    empathy = empathy_list[response_dict[one[idx][10:]]]
                    relevance = relevance_list[response_dict[one[idx][10:]]]
                    fluency = fluency_list[response_dict[one[idx][10:]]]
                    er = er_list[response_dict[one[idx][10:]]]
                    ip = ip_list[response_dict[one[idx][10:]]]
                    ex = ex_list[response_dict[one[idx][10:]]]
                elif one[idx][10:] in exist_responses:
                    the_one_index = exist_responses.index(one[idx][10:])
                    empathy = exist_scores[the_one_index][0]
                    relevance = exist_scores[the_one_index][1]
                    fluency = exist_scores[the_one_index][2]
                    er = exist_scores[the_one_index][3]
                    ip = exist_scores[the_one_index][4]
                    ex = exist_scores[the_one_index][5]
                else:
                    response_dict[one[idx][10:]] = i
                    print(output)
                    for x, jdx in enumerate(random_index):
                        if x < i:
                            print('{:<2}{:<50} **Score: {} {} {} {} {} {}'.format(x, one[jdx][10:], empathy_list[x],
                                                                                  relevance_list[x], fluency_list[x],
                                                                                  er_list[x], ip_list[x],
                                                                                  ex_list[x]))
                        else:
                            print('{:<2}{}'.format(x, one[jdx][10:]))
                    print()
                    while True:
                        print('Response {}{}'.format(i, one[idx][10:]))
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

                        er = input('ER 0~2: ')
                        while not er.isdigit() or int(er) < 0 or int(er) > 2:
                            print('Emotion reaction should be an integer in the range 0~2.', sep=' ')
                            er = input('ER 0~2: ')

                        ip = input('IP 0~2: ')
                        while not ip.isdigit() or int(ip) < 0 or int(ip) > 2:
                            print('Interpretation should be an integer in the range 0~2.', sep=' ')
                            ip = input('IP 0~2: ')

                        ex = input('EX 0~2: ')
                        while not ex.isdigit() or int(ex) < 0 or int(ex) > 2:
                            print('Explaination should be an integer in the range 0~2.', sep=' ')
                            ex = input('EX 0~2: ')

                        again = input('Do you want to regrade? (Y/N)')
                        if again == 'Y':
                            continue
                        else:
                            break

                empathy_list[i] = int(empathy)
                relevance_list[i] = int(relevance)
                fluency_list[i] = int(fluency)
                er_list[i] = int(er)
                ip_list[i] = int(ip)
                ex_list[i] = int(ex)

            for i, idx in enumerate(random_index):
                store_dict[index2model[idx - start + 11]]['Empathy'].append(empathy_list[i])
                store_dict[index2model[idx - start + 11]]['Relevance'].append(relevance_list[i])
                store_dict[index2model[idx - start + 11]]['Fluency'].append(fluency_list[i])
                store_dict[index2model[idx - start + 11]]['ER'].append(er_list[i])
                store_dict[index2model[idx - start + 11]]['IP'].append(ip_list[i])
                store_dict[index2model[idx - start + 11]]['EX'].append(ex_list[i])

    except KeyboardInterrupt:
        with open('./sample/ab-score.json', 'w') as f:
            json.dump(store_dict, f, indent=4)

    with open('./sample/ab-score.json', 'w') as f:
        json.dump(store_dict, f, indent=4)


if __name__ == '__main__':
    main()
    # interactive()
