import os 

experiences     = {}

PATH            = "C:/data/chess/mcts_copy/"

for filename in os.listdir(PATH):
    id      = filename[:5]
    
    print(f"id is {id}")

    with open(os.path.join(PATH,filename),'r') as file1obj:
        if id in experiences:
            experiences[id].append(file1obj.read())
        else:
            experiences[id] = [file1obj.read()]

for id in experiences:

    for i,contents in enumerate(experiences[id]):

        #Create new file
        fname           = f"C:/data/chess/game_experiences/{id}_{i}"
        with open(fname,'w') as fileobj:
            fileobj.write(contents)
        fileobj.close()