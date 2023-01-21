import itertools
import os 
import re
import xml.etree.ElementTree as ET


# pattern to filiter only amharic words 
pattern  = re.compile(r'[ሀ-ፚ\s]+', re.DEBUG) 
input_dir = "dataset/text/AA"
output_path = "dataset/amwiki.txt"
texts = []
for file in os.listdir(input_dir):
    with open(os.path.join(input_dir, file)) as f:
        it = itertools.chain('<root>', f, '</root>')
        root = ET.fromstringlist(it)
    for doc in root.findall("doc"):
        doc_txt = ""
        for match in pattern.findall(doc.text.strip()):
            doc_txt+= match.strip().replace("\n"," ")
        texts.append(doc_txt)

all_texts = "\n".join(texts)

with open (output_path,"w+") as f:
    f.write(all_texts + "\n")

