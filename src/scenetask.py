import io
import os
import numpy as np
from numpy import linalg as LA
import json
import operator
import spacy
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import models, transforms

nlp = spacy.load('en_core_web_lg')

global task, labs
task = None
labs = None

def cosine_similarity(vector1, vector2):
    return np.dot(vector1, vector2) / (LA.norm(vector1) * LA.norm(vector2))

def firstordtrickle(tree):
    score = {}
    for lev in tree["children"]:
        try:
            maxi = cosine_similarity(doc, lev["avg_embed"])
        except:
            maxi = 0
        score[lev["name"]] = maxi
    return sorted(score.items(), key=operator.itemgetter(1), reverse=True)

# def secondordtrickle(tree):
#     global labs
#     maxsc = 0
#     maxlab = None
#     for kk in labs:
#         doc = nlp("".join(kk.split(','))).vector
#         print(kk)
#         score = {}
#         try:
#             for lev in tree["children"]:
#                 if "avg_embed" in lev:
#                     maxi = cosine_similarity(doc, lev["avg_embed"])
#                 else:
#                     maxi = cosine_similarity(doc, lev["personal_embed"])
#                 # for key, value in lev.items():
#                 #     print(key)
#                 # print(maxi, lev["name"])
#                 score[lev["name"]] = maxi

#             score = sorted(score.items(), key=operator.itemgetter(1), reverse=True)

#             if score[0][1] > maxsc:
#                 maxsc = score[0][1]
#                 maxlab = score[0][0]
#                 print('--------------------')
#                 print(score[1][0])
#                 print(maxlab)
#         except:
#             pass
#     return (maxlab, maxsc)


def tree_construct(tree, doc, prev_score=[], relation=[], score_list=[]):
    global task
    score = firstordtrickle(tree)
    try:
        if score[0][1] > cosine_similarity(doc, tree["personal_embed"]):
            relation.append(score[0][0])
            score_list.append(score[0][1])
            new_tree = (item for item in tree["children"] if item["name"] == score[0][0]).__next__()
            tree_construct(new_tree, doc, prev_score=score, relation=relation, score_list=score_list)
        else:
            # secondordtrickle(tree)
            task_score = {}
            for kk in tree["content"]:
                maxi = cosine_similarity(doc, nlp(kk["name"]).vector)
                task_score[kk["name"]] = maxi
            task_score = sorted(task_score.items(), key=operator.itemgetter(1), reverse=True)
            task = task_score[0][0]
            return
    except Exception as e:
        print(e)
        pass

# net = models.vgg16(pretrained=True)

# normalize = transforms.Normalize(
#     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# preprocess = transforms.Compose([
#     transforms.Scale(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(), normalize
# ])

# np.random.seed(0)

# img = Image.open('../resources/surf.png')
# img_tensor = preprocess(img)
# img_tensor.unsqueeze_(0)
# img_tensor = img_tensor[:, :3, :, :]

# with open('../resources/labels.json', 'r') as f:
#     labeljson = json.load(f)

# labels = {int(key):value for (key, value)
#           in labeljson.items()}

# img_variable = Variable(img_tensor)
# fc_out = net(img_variable)

# scorelist = fc_out.data.numpy()
# args = np.argsort(scorelist).flatten().tolist()
# scores = scorelist.flatten().tolist()
# j2 = sorted([i for i in scores], reverse=True)

# finlist = []
# for i in j2:
#     if i < 0.9*j2[0]:
#         break
#     finlist.append(i)

# limfac = len(finlist)
# print(finlist)


# top10 = [(labels[ind], scores[ind]) for ind in args[-1*limfac:]]
# labs = [i[0] for i in top10]
# print(top10)

# doc = np.zeros(300)
# for i in top10:
#     doc += i[1] * (nlp(i[0]).vector)

doc = nlp('surfing surfboard').vector

tree_instant = json.load(open('../resources/avg_embed.json'))

relation = []
score_list = []
tree_construct(tree_instant, doc, relation=relation, score_list=score_list)
print(relation, task)