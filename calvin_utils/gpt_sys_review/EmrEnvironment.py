import os
import json

class EmrEnvironment:

    def __init__(self, path):
        self.run_number=self.find_filenum(path)
        self.path=self.append_filenum(path)
        self.parameters={}
    
    def find_filenum(self,filename):

        counter=0
        previous_name=None
        if os.path.isfile(filename):

            counter=1
            previous_name=filename

            while os.path.isfile(filename.replace('.json',f"_{counter:03}.json")):
                previous_name=filename.replace('.json',f"_{counter:03}.json")
                counter+=1

        self.previous_run=json.load(open(previous_name)) if previous_name else {}
        return counter
    
    def append_filenum(self, filename):
        if self.run_number != 0:
            filename=filename.replace('.json',f"_{self.run_number:03}.json")
            filename=filename.replace('.csv',f"_{self.run_number:03}.csv")
        return filename 
    
    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.parameters, f, indent=4)

    def update(self, dict):
        self.parameters.update(dict)
        self.save()
