"""
sumarize.py
"""
import pickle
import networkx as nx
import os

def main():
    if not os.path.isfile("users.pkl"):
        print ("data not loaded properly")
    else:
        if (os.path.getsize("users.pkl")==0):
            print ("data size not proper")
        else:
            users = pickle.load(open("users.pkl","rb"))
            friendlist = pickle.load(open("friendlist.pkl","rb"))
            followerdetails = pickle.load(open("followerdetails.pkl","rb"))
            graph = pickle.load(open("graph.pkl","rb"))
    no_users = len(users)+ len(followerdetails.keys())
    components  = [c for c in nx.connected_component_subgraphs(graph)]
    no_of_nodes=0
    for component in components:
        no_of_nodes+= nx.number_of_nodes(component)
    avg_community = no_of_nodes/len(components)
    followerlist = [follow_details for follow_details in followerdetails.keys()]
    male_instance = ''
    female_instance = ''
    malecount = 0
    femalecount=0
    for follow in followerdetails.keys():
        followername = followerdetails[follow]
        if ('gender' in followername.keys()):
            if (followername['gender']=='male'):
                malecount+=1
                if(len(male_instance)==0):
                    male_instance = followername
            if (followername['gender']=='female'):
                femalecount+=1
                if(len(female_instance)==0):
                    female_instance = followername 
    result = "Number of users collected: " + str(no_users) + "\n" + "Number of messages collected: " + str(len(followerlist)) + "\n" + "Number of communities discovered: "+ str(len(components)) + "\n" + "Average number of users per community: " + str(avg_community) + "\n" + "Number of instances per class found: \n \t" + "Male :"+ str(malecount) + "\n \t" + "Female :"+ str(femalecount) + "\n" + "Example from Male class: \n \t" + "Name: " + male_instance['name'] +"\n \t" + "Screen Name: " + male_instance['screen_name'] +"\n\tDescription: " + male_instance['description'] + "\n\tGender: " + male_instance['gender'] + "\nExample from Female class: \n \t" + "Name: " + female_instance['name'] +"\n \t" + "Screen Name: " + female_instance['screen_name'] +"\n\tDescription: " + female_instance['description'] + "\n\tGender: " + female_instance['gender'] 
    with open("summary.txt", "w") as text_file:
        text_file.write(result)
    text_file.close()
    
if __name__ == '__main__':
    main()