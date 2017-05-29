---
layout: post
published: true
project: true
title: PayWithText
blurb: Facilitating trade in emerging economies for those without smartphones
tags:
    - python
    - twilio
    - hackathon
---

## Inspiration
PayWithText is a simple way to make payments to others using only SMS. PayWithText doesn't require any internet - only a cell signal. It was designed with those who can't afford internet-enabled phones in mind by Ian Mobbs and Jacob Vanderlinden for [HackTX 2016](https://devpost.com/software/paywithtext).

## What it does
This application is a proof of concept. Every user is given a mock account using CapitalOne's Reimagine Banking API, "Nessie", and allocated a random amount of money between $1 and $10,000. It then allows you to pay, request, and receive money from anyone with a cell phone number.

## How we built it
PayWithText was built using Django, a set of Python web frameworks. We store our information using PostgresSQL and host our application on Heroku. Communication is facilitated by Twilio, and payment processing is done by the Capital One API.

## Built With
Python, Django, Heroku, Twilio

## What's Next
At the time, I had plenty of experience with Django and almost none with Flask - but Django is a VERY heavy tool for the job. In the future, I hope to re-write the entire application to use Flask. Additionally, we use Twilio headers for state management, which is bad practice at best and insecure at worst. This definitely needs to change.

## Links
 - [DevPost](https://devpost.com/software/paywithtext)
 - [Github](https://github.com/ianmobbs/PayWithText)