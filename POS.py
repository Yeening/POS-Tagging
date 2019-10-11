import numpy as np
import math
from copy import deepcopy

K = 2
# alpha = 1
# beta = 1
alpha = 1.5
beta = 0.001


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


def preprocessing(tokens):
    k = K
    dic = {}
    dic["UNK"] = 0
    #     prepocessed_tags = deepcopy(tags)
    for sentence in tokens:
        for token in sentence:
            if token in dic:
                dic[token] += 1
            else:
                dic[token] = 1
    #     new_dic = {}
    for i in range(len(tokens)):
        for j in range(len(tokens[i])):
            if dic[tokens[i][j]] < k:
                ori = tokens[i][j]
                dic["UNK"] += dic[ori]
                tokens[i][j] = "UNK"
                dic.pop(ori)

    return tokens, dic


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

    # fill blanks
    for y in list(y_dic.keys())[1:]:
        if y == 'end':
            continue
        for y1 in list(y_dic.keys())[1:]:
            y_y = y + '|' + y1
            if y_y not in p_y_y:
                p_y_y[y_y] = (0 + alpha) / (y_dic[y] + alpha * (len(y_dic) + 1))

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

        # fill blanks
    for x in x_dic:
        for y in list(y_dic.keys())[1:]:
            if y == 'end':
                continue
            x_y = x + '|' + y
            if x_y not in p_x_y:
                p_x_y[x_y] = (0 + beta) / (y_dic[y] + beta * len(x_dic))

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
    predicts = []
    for scentence in dev_tokens:
        v = []
        b = []
        predict = []
        # First loop, v1(k)
        v_s = {}
        token0 = scentence[0]
        for y_y in log_p_y_y.keys():
            if y_y.split('|')[0] == 'start':
                v_s[y_y] = log_p_y_y[y_y]
                if token0 + '|' + y_y.split('|')[1] in log_p_x_y:
                    v_s[y_y] += log_p_x_y[token0 + '|' + y_y.split('|')[1]]
                # else:
                    # v_s.pop(y_y)
        v.append(v_s)
        # for 2 to M
        pre_v_m = v_s
        for token in scentence[1:]:
            v_m = {}
            b_m = {}
            # for k
            for tag in list(y_dic.keys())[1:]:
                if tag == 'end':
                    continue
                # skip the impossible tags
                if token + '|' + tag not in log_p_x_y:
                    #                     print(token + '|' + tag)
                    token = "UNK"
                #                     continue
                # for each possibile tags, select the most scored previous tag
                max_b = list(pre_v_m.keys())[0].split('|')[1]
                max_v = list(pre_v_m.values())[0]
                for prev in pre_v_m:
                    last_y = prev.split('|')[1]
                    if last_y + '|' + tag not in log_p_y_y:
                        continue
                    if pre_v_m[prev] + log_p_y_y[last_y + '|' + tag] > max_v + log_p_y_y[max_b + '|' + tag]:
                        max_b = last_y
                        max_v = pre_v_m[prev] + log_p_y_y[last_y + '|' + tag]
                v_m[max_b + '|' + tag] = max_v + log_p_x_y[token + '|' + tag]
                b_m[tag] = max_b
            v.append(v_m)
            b.append(b_m)
            pre_v_m = v_m
        vs.append(v)

        # for last
        pairs = list(v[-1].keys())
        last_tag = pairs[0].split('|')[1]
        for pair in pairs[1:]:
            cur_tag = pair.split('|')[1]
            if log_p_y_y[cur_tag + '|' + 'end'] > log_p_y_y[last_tag + '|' + 'end']:
                last_tag = cur_tag
        #         print(last_tag)
        #         print(b)

        # read the result
        prev = last_tag
        predict.append(prev)
        for b_m in reversed(b):
            predict.append(b_m[prev])
            prev = b_m[prev]
        predict = [ele for ele in reversed(predict)]
        predicts.append(predict)
    #         print(predict)

    #         break

    return predicts


def get_acc(tags, predicts):
    total = 0.0
    correct = 0.0
    for i in range(len(tags)):
        total += len(tags[i])
        for j in range(len(tags[i])):
            if tags[i][j] == predicts[i][j]:
                correct += 1
    return correct / total


trn_tokens, trn_tags = load_data_set("./data/dev.pos")
prepocessed_tokens, x_dic = preprocessing(trn_tokens)
# print(len(x_dic))

# print(len(trn_texts[0]),len(trn_tags[0]))
p_y_y, p_x_y, y_dic = get_probilities(prepocessed_tokens, trn_tags, x_dic)
# print(y_dic)
# print(p_y_y.keys())
# print(len(p_x_y))
log_p_y_y, log_p_x_y = to_logistic(p_y_y, p_x_y)
# print(len(log_p_y_y))

# dev_tokens, dev_tags = load_data_set("./data/dev.pos")
# prepocessed_dev_tokens,x_dev_dic = preprocessing(dev_tokens)
# a,b,y_dic = get_probilities(prepocessed_dev_tokens,dev_tags,x_dev_dic)
# dev_predicts = Viterbi(log_p_y_y,log_p_x_y,prepocessed_dev_tokens,y_dic)
# acc = get_acc(dev_tags,dev_predicts)

test_tokens, test_tags = load_data_set("./data/tst.pos")
prepocessed_test_tokens, x_test_dic = preprocessing(test_tokens)
# a, b, y_dic = get_probilities(prepocessed_dev_tokens, dev_tags, x_test_dic)
test_predicts = Viterbi(log_p_y_y, log_p_x_y, prepocessed_test_tokens, y_dic)
acc = get_acc(test_tags, test_predicts)
print("The accuracy is: " + str(acc))