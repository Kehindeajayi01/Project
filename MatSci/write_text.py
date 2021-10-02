import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help = "path to the text files")
    parser.add_argument("--outputDirectory", help = "path to store the text files")
    return parser    
    


def clean_text(index):
    parser = get_args()
    args = parser.parse_args()
    input_path = args.file_path
    output_path = args.outputDirectory
    
    rel_paths = os.listdir(input_path)
    relative_path = rel_paths[index]
    
    f = open(os.path.join(input_path, relative_path), "r", encoding="utf-8")
    text = f.read()
    
    f.close()
    text = text.replace("^", "")
    text = text.replace("_", "")
    text =  text.replace("{", "")
    text = text.replace("}", "")
    text = text.replace("→", "=")
    
    out = open(os.path.join(output_path, relative_path), "w", encoding="utf-8")
    out.write(text)
    out.close()
    
#clean_text()