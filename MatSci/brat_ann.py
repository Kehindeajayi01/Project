import nltk
from nltk.tokenize import sent_tokenize
import glob
from brat_parser import get_entities_relations_attributes_groups
import json
import stanza

stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize')

def get_files(file_path):
    files = glob.glob(file_path)
    return files
    
files = "C:/Users/ajayi/OneDrive/Desktop/Brat_annotation_cp/anns/*.ann"
txt_files = "C:/Users/ajayi/OneDrive/Desktop/Brat_annotation_cp/txts/*.txt"

ann_files = get_files(files)
text_files = get_files(txt_files)


def ents_rel_annotations(file, i):
    results = {}
    rel_result = []
    ents_result = []
    # get the entities and the annotations from the ann file
    entities, relations, _, _ = get_entities_relations_attributes_groups(file[i])
    # read and tokenize the text file corresponding to the ann file
    fn = open(text_files[i], 'r', encoding='utf-8')
    text = fn.read()
    doc = nlp(text)
    tokens = [token.text for sentence in doc.sentences for token in sentence.tokens]
    # splitting into sentences
    #sent_tokens = sent_tokenize(text)
    # tokenizing each sentence
    #tokens = [line.split() for line in sent_tokens]
    for (key, val), j in zip(entities.items(), relations):
        ents_result.append([val.start, val.end, val.type])
        id1 = relations[j].subj
        id2 = relations[j].obj
        span1 = entities[id1].span[0]
        span2 = entities[id2].span[0]
        min_start, min_end = min(span1, span2)
        max_start, max_end = max(span1, span2)
        rel_result.append([min_start, min_end, max_start, max_end, relations[j].type])
       
        results['sentences'] = tokens
        results['ner'] = ents_result
        results['relations'] = rel_result
        
    
    return results
    

with open('new_result.json', 'w', encoding='utf-8') as fp:
    for i in range(len(ann_files)):
        sample = ents_rel_annotations(ann_files, i)
        json.dump(sample, fp, ensure_ascii=False)
    fp.close()
    
