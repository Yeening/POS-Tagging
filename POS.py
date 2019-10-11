import numpy as np
import math
from copy import deepcopy

K = 2
alpha = 1
beta = 1


def load_data_set(filename):
    # your code
    tokens = []
    tags = []
    f = open(filename, 'r')
    for line in f:
        pairs = line.split()
        current_tokens = []
        current_tags = []
        for pair in pairs:
            data = pair.split('/')
            current_tokens.append(data[0].lower())
            current_tags.append(data[1])
        tokens.append(current_tokens)
        tags.append(current_tags)
    return tokens, tags


def preprocessing(tokens, tags):
    k = K
    dic = {}
    prepocessed_tags = deepcopy(tags)
    for sentence in tokens:
        for token in sentence:
            if token in dic:
                dic[token] += 1
            else:
                dic[token] = 1
    for i in range(len(tokens)):
        for j in range(len(tokens[i])):
            if dic[tokens[i][j]] < k:
                prepocessed_tags[i][j] = "UNK"
    return prepocessed_tags, dic


def get_probilities(tokens, tags, x_dic):
    y_dic = {}
    y_y_dic = {}
    for tag_set in tags:
        new_tag_set = list(tag_set)
        new_tag_set.insert(0, 'start')
        new_tag_set.append('end')

        for i in range(len(new_tag_set)):
            if new_tag_set[i] in y_dic:
                y_dic[new_tag_set[i]] += 1
            else:
                y_dic[new_tag_set[i]] = 1

            if i == len(new_tag_set) - 1:
                break

            if new_tag_set[i] + '|' + new_tag_set[i + 1] in y_y_dic:
                y_y_dic[new_tag_set[i] + '|' + new_tag_set[i + 1]] += 1
            else:
                y_y_dic[new_tag_set[i] + '|' + new_tag_set[i + 1]] = 1
    #     print(y_dic,y_y_dic)
    p_y_y = y_y_dic.copy()
    for y_y in y_y_dic.keys():
        p_y_y[y_y] = 0.0
        p_y_y[y_y] = (y_y_dic[y_y] + alpha) / (y_dic[y_y.split('|')[0]] + alpha * (len(y_dic) + 1))

    x_y_dic = {}
    for i in range(len(tokens)):
        for j in range(len(tokens[i])):
            if tokens[i][j] + '|' + tags[i][j] in x_y_dic:
                x_y_dic[tokens[i][j] + '|' + tags[i][j]] += 1
            else:
                x_y_dic[tokens[i][j] + '|' + tags[i][j]] = 1

    p_x_y = {}
    for x_y in x_y_dic.keys():
        p_x_y[x_y] = 0.0
        p_x_y[x_y] = (x_y_dic[x_y] + beta) / (y_dic[x_y.split('|')[1]] + beta * len(x_dic))

    #     count = 0.0
    #     for x_y in p_x_y.keys():
    #         if x_y.split('|')[1] == 'C':
    #             count += p_x_y[x_y]
    #     print(count)

    #     count = 0.0
    #     for y_y in p_y_y.keys():
    #         if y_y[0] == 'P':
    #             count += p_y_y[y_y]
    #     print(count)
    return p_y_y, p_x_y, y_dic


def to_logistic(p_y_y, p_x_y):
    log_p_y_y = p_y_y.copy()
    log_p_x_y = p_x_y.copy()
    for key in log_p_y_y:
        log_p_y_y[key] = math.log2(log_p_y_y[key])
    for key in log_p_x_y:
        log_p_x_y[key] = math.log2(log_p_x_y[key])
    return log_p_y_y, log_p_x_y


def Viterbi(log_p_y_y, log_p_x_y, dev_tokens, y_dic):
    vs = []
    for scentence in dev_tokens:
        v = []
        # First loop, v1(k)
        v_s = {}
        token = scentence[0]
        for y_y in log_p_y_y.keys():
            if y_y.split('|')[0] == 'start':
                v_s[y_y] = log_p_y_y[y_y]
                if token + '|' + y_y.split('|')[1] in log_p_x_y:
                    v_s[y_y] += log_p_x_y[token + '|' + y_y.split('|')[1]]
                else:
                    v_s.pop(y_y)
        v.append(v_s)
        # for 2 to M
        pre_v_m = v_s
        for token in scentence[1:]:
            v_m = {}
            b_m = {}
            # for k
            for tag in y_dic:
                # skip the impossible tags
                if token + '|' + tag not in log_p_x_y:
                    continue
                # for each possibile tags, select the most scored previous tag
                max_b = list(pre_v_m.keys())[0].split('|')[1]
                max_v = list(pre_v_m.values())[0]
                for prev in pre_v_m:
                    last_y = prev.split('|')[1]
                    if pre_v_m[prev] + log_p_y_y[last_y + '|' + tag] >= max_v + log_p_y_y[max_b + '|' + tag]:
                        max_b = last_y
                        max_v = pre_v_m[prev] + log_p_y_y[last_y + '|' + tag]
                v_m[max_b + '|' + tag] = max_v + log_p_x_y[token + '|' + tag]
            v.append(v_m)
            pre_v_m = v_m.copy()
            print(token,pre_v_m)
        vs.append(v)
        # for end
        for pairs in vs[-1]:
            pass
    break
    #                 # for k'
    #                 max_v = log_p_y_y[log_p_y_y.keys()[0]]
    #                 max_k = ''
    #                 for pre_tag in y_dic:
    #                     s = log_p_x_y[token+'|'+pre_tag] +

    # print(len(v), len(v[0]), vs[0][2])


trn_tokens, trn_tags = load_data_set("./data/dev.pos")
prepocessed_tags, x_dic = preprocessing(trn_tokens, trn_tags)

# print(len(trn_texts[0]),len(trn_tags[0]))
p_y_y, p_x_y, y_dic = get_probilities(trn_tokens, prepocessed_tags, x_dic)
# print(p_x_y)
log_p_y_y, log_p_x_y = to_logistic(p_y_y, p_x_y)
# print(len(trn_tokens))

dev_tokens, dev_tags = load_data_set("./data/dev.pos")
Viterbi(log_p_y_y, log_p_x_y, dev_tokens, y_dic)