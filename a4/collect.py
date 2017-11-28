"""
collect.py
"""
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import sys
import pickle
import time
import warnings
import requests
from TwitterAPI import TwitterAPI

consumer_key = '8VEfsBxDFtmN0xHYM0saAf7Dq'
consumer_secret = 'Imtc63LZhSs0UvkMBqxUfLnHJEInDjhevY8ojFPo0y9klgnGFX'
access_token = '148849891-90859yOIGG33fmEdhhlHtzMK2EPOz4EoacXEj53s'
access_token_secret = 'SnrSM5FQnpLpgtJh0q7CfZxvZ9RprC0afXGq74fEvekXU'

def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)

def get_users(twitter, screen_names):
    """Retrieve the Twitter user objects for each screen_name.
    Params:
        twitter........The TwitterAPI object.
        screen_names...A list of strings, one per screen_name
    Returns:
        A list of dicts, one per user, containing all the user information
        (e.g., screen_name, id, location, etc)
    See the API documentation here: https://dev.twitter.com/rest/reference/get/users/lookup
    In this example, I test retrieving two users: twitterapi and twitter.
    >>> twitter = get_twitter()
    >>> users = get_users(twitter, ['twitterapi', 'twitter'])
    >>> [u['id'] for u in users]
    [6253282, 783214]
    """
    request = robust_request(twitter,'users/lookup', {'screen_name':screen_names}) 
    user_object = [r for r in request]
    return user_object

def get_followers(twitter,users):
    followerdetails = {}
    for user in users:
        followers=[]
        request = robust_request(twitter,'followers/list', {'screen_name':user['screen_name'],'count':200})
        for r in request:
            if (len(r['description'])>1):
                followers.append(r['screen_name'])
                details = {}
                details['description'] = r['description']
                details['name'] = r['name']
                details['screen_name'] = r['screen_name']
                followerdetails[r['screen_name']]=details
        user['followers']=followers
    return users,followerdetails

def count_friends(users):
    """ Count how often each friend is followed.
    Args:
        users: a list of user dicts
    Returns:
        a Counter object mapping each friend to the number of candidates who follow them.
        Counter documentation: https://docs.python.org/dev/library/collections.html#collections.Counter
    In this example, friend '2' is followed by three different users.
    >>> c = count_friends([{'friends': [1,2]}, {'friends': [2,3]}, {'friends': [2,3]}])
    >>> c.most_common()
    [(2, 3), (3, 2), (1, 1)]
    """
    count = Counter()
    for i in range(len(users)):
        count.update(users[i]['followers'])
    return count

def followers(twitter, users):
    """ Get the list of accounts each user follows.
    I.e., call the get_friends method for all 4 candidates.
    Store the result in each user's dict using a new key called 'friends'.
    Args:
        twitter...The TwitterAPI object.
        users.....The list of user dicts.
    Returns:
        Nothing
    >>> twitter = get_twitter()
    >>> users = [{'screen_name': 'aronwc'}]
    >>> add_all_friends(twitter, users)
    >>> users[0]['friends'][:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    users,followerdetails = get_followers(twitter,users)
    Fordset = set(users[0]['followers'])
    Hondaset = set(users[1]['followers'])
    Hyundaiset = set(users[2]['followers'])
    Toyotaset = set(users[3]['followers'])
    friendsList = []
    common_followers = Fordset & Hondaset
    friendsList.extend(list(Fordset & Hyundaiset))
    friendsList.extend(list(Fordset & Toyotaset))
    friendsList.extend(list(Hondaset & Hyundaiset))
    friendsList.extend(list(Hondaset & Toyotaset))
    friendsList.extend(list(Hyundaiset & Toyotaset))
    return users,friendsList,followerdetails

def get_cosensusdata():
    males = requests.get('http://www2.census.gov/topics/genealogy/1990surnames/dist.male.first').text.split('\n')
    females = requests.get('http://www2.census.gov/topics/genealogy/1990surnames/dist.female.first').text.split('\n')
    male_dict = dict([(male.split()[0].lower(),float(male.split()[1])) for male in males if male])
    female_dict = dict([(female.split()[0].lower(),float(female.split()[1])) for female in females if female])
    male_names = set([m for m in male_dict if m not in female_dict or male_dict[m]>female_dict[m] ])
    female_names = set([m for m in female_dict if m not in male_dict or male_dict[m]<female_dict[m] ])
    return male_names, female_names

def main():
    twitter = get_twitter()
    screen_names = ['Honda','Toyota','Hyundai','Ford']
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    users,friendlist,followerdetails =followers(twitter,users) 
    friend_counts = count_friends(users)
    pickle.dump(users, open('users.pkl', 'wb'))
    pickle.dump(friendlist, open('friendlist.pkl', 'wb'))
    pickle.dump(followerdetails, open('followerdetails.pkl', 'wb'))
    male_names, female_names = get_cosensusdata()
    pickle.dump(male_names, open('maleNames.pkl', 'wb'))
    pickle.dump(female_names, open('femaleNames.pkl', 'wb'))
if __name__ == '__main__':
    main()