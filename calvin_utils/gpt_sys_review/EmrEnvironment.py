import os
import json

class EmrEnvironment:

    def __init__(self, path):
        self.path=self.find_filenum(path)
        self.parameters={}
    
    def find_filenum(self,filename):

        counter=0
        previous_name=None
        if os.path.isfile(filename):

            counter=1
            previous_name=filename

            while os.path.isfile(filename.replace('.json',f"_{counter}.json")):
                previous_name=filename.replace('.json',f"_{counter}.json")
                counter+=1

        self.previous_run=json.load(open(previous_name)) if previous_name else {}
        self.counter=counter

        return filename.replace('.json',f"_{counter}.json") if counter>0 else filename
    
    def append_filenum(self, filename):
        if self.counter != 0:
            filename=filename.replace('.json',f"_{self.counter}.json")
            filename=filename.replace('.csv',f"_{self.counter}.csv")
        return filename 
    
    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.parameters, f, indent=4)

    def update(self, dict):
        self.parameters.update(dict)
        self.save()
