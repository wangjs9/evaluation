import os
import numpy as np
import json

useless_word = {'hear', 'happen', 'happened', 'happening', 'hope', 'glad', 'true', 'okey',
                'feel', 'sound', 'time', 'felt', 'yeah', 'thing'}


def coverage():
    report_path = 'report_coverage.txt'
    # read the context
    context = np.load('golden_dataset/sys_dialog_texts.test.npy', allow_pickle=True)
    emotion = np.load('golden_dataset/sys_emotion_texts.test.npy', allow_pickle=True)
    target = np.load('golden_dataset/sys_target_texts.test.npy', allow_pickle=True)
    next_graph = np.load('golden_dataset/sys_next_graphs.test.npy', allow_pickle=True)
    # causal information
    CauseDict = json.load(open('../data/Cause_Effect/cause_dic.json', 'r'))
    EffectDict = json.load(open('../data/Cause_Effect/effect_dic.json', 'r'))
    CauseDict['proud'] = CauseDict['pride']
    EffectDict['proud'] = EffectDict['pride']
    with open('golden_dataset/ConceptVocabulary.txt', 'r') as f:
        lines = f.readlines()
    Vocabulary = {word.strip() for word in lines}
    with open('golden_dataset/vocabulary.txt', 'r') as f:
        lines = f.readlines()
        lines = {word.strip() for word in lines}
    Vocabulary = Vocabulary & lines
    # read the generation
    file_list = os.listdir('EMNLP22-res')
    file_list.sort()
    file_dict = {}
    model = [x.split('-')[0] for x in file_list]
    model = set(model)
    for filename in file_list:
        with open(os.path.join('EMNLP22-res', filename), 'r') as f:
            lines = f.readlines()
        lines = [l.strip() for l in lines]
        file_dict[filename] = lines
    # begin computation
    result_dict = {name: [] for name in file_list}
    for idx, (con, emo, tgt) in enumerate(zip(context, emotion, target)):
        con_list = [x for lst in con[::2] for x in lst if x in Vocabulary]
        con_list.append(emo)
        con_list = set(con_list)
        target_len = len(tgt)
        all_cand = set()
        for c in con_list:
            all_cand.update(CauseDict.get(c, []))
            all_cand.update(EffectDict.get(c, []))
        for name in file_list:
            token_list = file_dict[name][idx].split()
            sent_len = len(token_list)
            token_list = [x for x in token_list if x in Vocabulary]
            valid = []
            for t in token_list:
                if t in next_graph[idx]['nodes']:
                    valid.append(t)
            result_dict[name].append(len(valid) / len(next_graph[idx]['nodes']))
            # result_dict[name].append(len(valid) / sent_len * np.exp(min(0, 1- sent_len/target_len)))

    # if not os.path.exists('coverage_files/'):
    #     os.mkdir('coverage_files')
    # for name in file_list:
    #     with open('coverage_files/{}'.format(name), 'w') as f:
    #         for lst in result_dict[name]:
    #             f.write(json.dumps(lst))
    #             f.write('\n')

    print_res = {x: 0 for x in model}
    for name in file_list:
        # num = np.mean([len(x) for x in result_dict[name]])
        num = np.mean(result_dict[name])
        print_res[name.split('-')[0]] += num
    for name, num in print_res.items():
        print('%s : %f' % (name, num / 5))


coverage()
