---
layout: post
published: true
project: true
title: KVRX Livestream
blurb: Chrome Extension allowing you to listen to KVRX while browsing
tags:
    - chrome extension
    - javascript
---

## What It Does

Built in an effort to increase KVRX availability to the general public, this extension allows you to listen to KVRX while browsing the web.

## How It Works

There are two main components to the extension, `popup.js` and `background.js` (named following typical Chrome Extension naming standards).

`popup.js` is displayed when the user selects the extension from their extension tray, and:
 - Scrapes the KVRX website for current track information
 - Displays system control buttons ("Start" and "Stop")
 - Communicates with `background.js`

`background.js` is a persistent script run in the background, and:
 - Loads an "invisible" audio element pointed at the KVRX livestream URL
 - Listens for button press signals from `popup.js` and controls the player accordingly

## Links
 - [Github](https://github.com/ianmobbs/KVRX-Livestream)
 - [Chrome Extension Store](https://chrome.google.com/webstore/detail/kvrx-livestream/oebohfmoakpighofinngopmedfmdbpdh)