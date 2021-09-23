import yaml
with open("/home/ddl/git/kaggle_project1/conf/optimizer/adamw.yaml") as f:
     list_doc = yaml.load(f)
list_doc['params']['lr'] = 5e-6

with open("/home/ddl/git/kaggle_project1/conf/optimizer/adamw.yaml", "w") as f:
    yaml.dump(list_doc, f)

exec(open('/home/ddl/git/kaggle_project1/train.py').read())