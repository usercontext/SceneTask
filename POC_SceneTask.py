import io
import os
import numpy as np
from numpy import linalg as LA
import json
import operator
import spacy
nlp = spacy.load('en_core_web_lg')


# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

global task
task = None

def cosine_similarity(vector1, vector2):
    return np.dot(vector1, vector2) / (LA.norm(vector1) * LA.norm(vector2))

def tree_construct(tree, doc, prev_score=[], relation=[], score_list=[]):
    global task
    score = {}
    for lev in tree["children"]:
        try:
            maxi = cosine_similarity(doc, lev["avg_embed"])
        except:
            maxi = 0
        score[lev["name"]] = maxi
    score = sorted(score.items(), key=operator.itemgetter(1), reverse=True)
    try:
        if score[0][1] > cosine_similarity(doc, tree["personal_embed"]):
            relation.append(score[0][0])
            score_list.append(score[0][1])
            new_tree = (item for item in tree["children"] if item["name"] == score[0][0]).__next__()
            tree_construct(new_tree, doc, prev_score=score, relation=relation, score_list=score_list)
        else:
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


# Instantiates a client
client = vision.ImageAnnotatorClient()

# The name of the image file to annotate
file_name = os.path.join(os.path.dirname(__file__), 'resources/polo.jpg')

# Loads the image into memory
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)

# Performs label detection on the image file
response = client.label_detection(image=image)
labels = response.label_annotations


doc = np.zeros(300)
for label in labels:
    doc += label.score * (nlp(label.description).vector)
    print(label.description, label.score)

# doc = nlp('baseball pitching').vector

# print(cosine_similarity(nlp('baseball pitch').vector, doc))
tree_instant = json.load(open('resources/avg_embed.json'))

relation = []
score_list = []
tree_construct(tree_instant, doc, relation=relation, score_list=score_list)
print(relation, task)
# related = ' --> '.join(relation)
