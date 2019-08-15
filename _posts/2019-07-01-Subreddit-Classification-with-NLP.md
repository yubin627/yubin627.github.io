---
title: "Subreddit Classification with API and NLP"
layout: post
date: 2019-07-01 22:10
tag: NLP
headerImage: false
projects: true
hidden: true # don't count this post in blog pagination
description: ""
category: project
author: yubin
externalLink: false
---

Please see the [codes](https://github.com/yubin627/ga_projects/tree/master/Project_3)

---

### Problem Statement

Use NLP to train a classifier to tell which of the two selected subreddits a given post should come from.


### Executive Summary

As a diehard fan of Harry Potter series, the characters and the events in the books/movies are stored in my brain in chronological order. I could easily tell which era / which title of the book the stories belong to. However it might not be so easy for a casual fan of the Harry Potter film series or its spinoff, Fantastic Beasts and Where to Find Them. Also, it is natural to find a lot in common between the two series as there are a couple of characters that appear in both series, and of course topics could be a mix of the two since the author/film crew are (almost) the same. Therefore it would be quite normal to see posts wrongly created in the other subreddit.

This project aims to propose a classification model that would be able to help aggregate the posts more accurately based on the title and the body text of the post. To achieve this, I selected and evaluated a few classification models using Natural Language Processing (NLP) tools.

The subreddits chosen to compare are:

- [**r/harrypotter**](https://www.reddit.com/r/harrypotter/)
- [**r/FantasticBeasts**](https://www.reddit.com/r/FantasticBeasts/)
