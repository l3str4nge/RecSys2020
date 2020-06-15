#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd


# In[2]:

for i in range(0, 10):

    in_file = "/home/layer6/recsys/clean/xlmr/chunk{}.p".format(i)
    out_file = "/home/layer6/recsys/clean/xlmr/clean/chunk{}.p".format(i)
    # in_file = "/media/kevin/datahdd/data/recsys/tweetstring/train/test.p"
    # out_file = "/media/kevin/datahdd/data/recsys/tweetstring/train/clean_test.p"
    df = pd.read_pickle(in_file)


    # In[3]:


    df.sample(50)


    # # Remove Spaces in the URL, Mentions

    # In[ ]:


    df["url_string"] = df["url_string"].str.replace(" ", "")
    df["main_mention"] = df["main_mention"].str.replace(" ", "")
    df["other_mentions"] = df["other_mentions"].map(lambda l: list(map(lambda x: x.replace(' ', ''), l)) if type(l) == list else l)


    # In[ ]:


    df.sample(50)


    # # Remove [CLS] and [SEP] tokens

    # In[ ]:


    df["tweet_clean"] = df["tweet_clean"].str.replace(r'\[CLS\]', "").str.replace(r'\[SEP\]', "").str.replace(r'\[UNK\]', "<unk>")


    # In[ ]:


    df["tweet_clean"] = df["tweet_clean"].str.strip()


    # In[ ]:


    pd.set_option('display.max_colwidth', -1)
    df["tweet_clean"].sample(200)


    # # Get word count

    # In[ ]:


    df["word_count"] = df["tweet_clean"].str.split().apply(len)
    df["char_count"] = df["tweet_clean"].str.len()


    # In[ ]:


    df.sample(50)


    # # Save

    # In[ ]:


    df.to_pickle(out_file, protocol=pickle.HIGHEST_PROTOCOL)
