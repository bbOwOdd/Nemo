import json

def read_jsonl(fname):
    obj = []
    with open(fname, 'rt') as f:
        st = f.readline()
        while st:
            obj.append(json.loads(st))
            st = f.readline()
    return obj

def write_jsonl(fname, json_objs):
    with open(fname, 'wt') as f:
        for o in json_objs:
            f.write(json.dumps(o)+"\n")
            
def form_question(obj):
    st = ""    
    for i, label in enumerate(obj['LABELS']):
        st += f"{label}: {obj['CONTEXTS'][i]}\n"
    st += f"QUESTION: {obj['QUESTION']}\n"
    st += f" ### ANSWER (yes|no|maybe): "
    return st

def convert_to_jsonl(data_path, output_path):
    data = json.load(open(data_path, 'rt'))
    json_objs = []
    for k in data.keys():
        obj = data[k]
        prompt = form_question(obj)
        completion = obj['final_decision']
        json_objs.append({"input": prompt, "output": f"<<< {completion} >>>"})
    write_jsonl(output_path, json_objs)
    return json_objs


test_json_objs = convert_to_jsonl("/home/z890/pubmedqa/data/test_set.json", "/home/z890/pubmedqa/data/pubmedqa_test.jsonl")
train_json_objs = convert_to_jsonl("/home/z890/pubmedqa/data/pqal_fold0/train_set.json", "/home/z890/pubmedqa/data/pubmedqa_train.jsonl")
dev_json_objs = convert_to_jsonl("/home/z890/pubmedqa/data/pqal_fold0/dev_set.json", "/home/z890/pubmedqa/data/pubmedqa_val.jsonl")
