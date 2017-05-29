---
layout: post
published: true
project: true
title: DailyDevTalk
blurb: Connecting developers on Twitter talking about the same thing
tags:
    - python
    - twitter bot
---

## What It Does

A Twitter bot to help foster discussion among the development community. By tweeting with the hashtag #DevTalk, you can be matched up with another member of the #DevTalk community for discussion.

## How It Works

Every time the system receives a tweet, it looks at the last 100 tweets using the hashtag #DevTalk and finds the cosine simularity between the new tweet and the tweet found. It finds the tweet with the highest similarity score, and then connects the two users.

## What's next

Natural language processing!

## Built with

Python, Tweepy, Heroku

## Links
 - [Github](https://github.com/ianmobbs/DailyDevTalk)
 - [Twitter](https://twitter.com/dailydevtalk/with_replies)