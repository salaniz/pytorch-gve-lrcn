#Code by Lisa Anne Hendricks for "Generating Visual Explanations"
#
#@article{hendricks2016generating,
#  title={Generating Visual Explanations},
#  author={Hendricks, Lisa Anne and Akata, Zeynep and Rohrbach, Marcus and Donahue, Jeff and Schiele, Bernt and Darrell, Trevor},
#  journal={arXiv preprint arXiv:1603.08507},
#  year={2016}
#}

#Preprocess *.tsv raw captions into MSCOCO description format

#import sys
#sys.path.append('utils/')
import re
from cub_utils import *
import argparse

data_prefix = 'data/cub/'

sentence_split_regex = re.compile(r'(\W+)')
def split_sentence(sentence):
  sentence = [s.lower() for s in sentence_split_regex.split(sentence.strip()) if len(s.strip()) >0]
  if sentence[-1] != '.':
    return sentence
  return sentence[:-1]

def bird_preprocess(raw_descriptions):
  #takes raw descriptions (*.tsv) and generates json file in the same format as MSCOCO
  json_descriptions = {}
  json_descriptions['images'] = []
  json_descriptions['annotations'] = []
  json_descriptions['type'] = 'captions'

  des = open_txt(raw_descriptions)
  des = [d.split('\t') for d in des]
  important_info = [(d[-1], d[-2]) for d in des if d[-7] == '']

  descriptions = list(zip(*important_info))[0]
  im_ids = ['/'.join(i.split('/')[-2:]) for i in list(zip(*important_info))[1]]

  unique_ims = list(set(im_ids))

  for unique_im in unique_ims:
    im = {}
    im['file_name'] = unique_im
    im['id'] = unique_im
    json_descriptions['images'].append(im)

  count = 0
  for description, im_id in zip(descriptions, im_ids):
    a = {}
    a['caption'] = description
    a['id'] = count
    a['image_id'] = im_id
    count += 1
    json_descriptions['annotations'].append(a)

  return json_descriptions

dataset_hash = {'bird': {'raw_descriptions': data_prefix + 'cub_0917_5cap.tsv',
                         'preprocessor': bird_preprocess}}

def create_json(im_to_annotations, im_to_images, set_ims):
  json_descriptions = {}
  json_descriptions['images'] = []
  json_descriptions['annotations'] = []
  json_descriptions['type'] = 'captions'

  for im in set_ims:
    json_descriptions['images'].append(im_to_images[im])
    json_descriptions['annotations'].extend(im_to_annotations[im])

  return json_descriptions

def create_finegrained(im_to_annotations, im_to_images, ims):

  ims = open_txt(ims)

  descriptions = create_json(im_to_annotations, im_to_images, ims)

  return descriptions

def create_json_finegrained(im_to_annotations, im_to_images, image_base_path, train_ims, test_ims):

  train_classes = open_txt(train_ims)
  test_classes = open_txt(test_ims)

  train_descriptions = create_json(train_ims)
  if test_ims:
    test_descriptions = create_json(test_imso)
  else:
    test_descriptions = None

  return train_descriptions, test_descriptions

def create_im_dicts(descriptions):
  im_to_annotations = {}
  for anno in descriptions['annotations']:
    if anno['image_id'] in im_to_annotations.keys():
      im_to_annotations[anno['image_id']].append(anno)
    else:
      if not 'cub_missing' in anno['image_id']:
        im_to_annotations[anno['image_id']] = [anno]

  im_to_images = {}
  for im in descriptions['images']:
    im_to_images[im['id']] = im

  return im_to_annotations, im_to_images

def save_descriptions(save_dict, save_tag):
  save_path = data_prefix + 'descriptions_' + save_tag + '.json'
  save_json(save_dict, save_path)
  print('Saved captions to %s.\n' % save_path)

if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument("--description_type", type=str, default='bird')
  parser.add_argument("--splits", type=str, default=None)

  args = parser.parse_args()

  raw_descriptions = dataset_hash[args.description_type]['raw_descriptions']
  json_descriptions = dataset_hash[args.description_type]['preprocessor'](raw_descriptions)

  im_to_annotations, im_to_images = create_im_dicts(json_descriptions)

  save_tag = args.description_type

  for ims in args.splits.split(','):
    descriptions = create_finegrained(im_to_annotations, im_to_images, ims)
    split = ims.split('/')[-1].split('.txt')[0]
    tag = '%s.%s.fg' %(save_tag, split)
    save_descriptions(descriptions, tag)
    if 'train' in ims: #make vocab from train set
      words = []
      for caption in descriptions['annotations']:
        words.extend([s.strip() for s in split_sentence(caption['caption'])])
      vocab = sorted(list(set(words)))
      vocab_file = open(data_prefix + 'vocab.txt', 'w')
      for v in vocab:
        vocab_file.writelines('%s\n' %v)
      vocab_file.close()


