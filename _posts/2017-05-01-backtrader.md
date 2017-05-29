---
layout: post
published: true
project: true
title: Backtrader
blurb: Evaluates and recommends trading strategies over historical stock performances
tags:
    - python
    - finance
    - stock trading
---

## What It Does

Backtrader allows you to write custom trading strategies and evaluate them over given timeframes. A sample of 30/180 SMA crossover strategy is provided as a reference.

## How It Works

By creating a custom `Backtrader` and custom `Strategy` class, you're able to retrieve historical stock data for any public company and evaluate the strategy you've defined.

The `Backtrader` class:
 - Retrieves historical stock data for a given ticker
 - Evaluates strategies defined in a `Strategy` class
 - Calculates returns over time
 - Determines whether or not strategy was successful over a given timeframe
 - Plots stock ownership and returns

The `Strategy` class:
 - Requires Plot Labels for plotting, strategy criteria, and ownership criteria
 - Strategy criteria creates new data in a dataframe for ownership calculations
 - Ownership looks at data created by the strategy and determines when to hold a stock

## What's next

Support for multiple stocks at once and potential Robinhood integration

## Built with

Python, Pandas

## Links
 - [Github](https://github.com/ianmobbs/Backtrader)