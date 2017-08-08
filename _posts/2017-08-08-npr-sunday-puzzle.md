---
layout: post
published: true
type: article
title: Brute-forcing the NPR Sunday Puzzle
blurb: Does the NPR Sunday puzzle really require inherent cleverness, or just the ability to code? Are they one and the same?
tags:
    - python
    - openlibrary
    - npr
---

## Inspiration

A couple weeks ago, I was looking at the [NPR Sunday Puzzle](http://www.npr.org/2017/07/16/537225382/sunday-puzzle-wehn-wrods-get-rearearngd) and brainstorming the answer with my girlfriend. The challenge is:

> Name a U.S. city and its state — 12 letters altogether. Change two letters in the state's name. The result will be the two-word title of a classic novel. What is it?

After some time just trying to list all the books we knew, we Googled something along the lines of "list of classic books" and running through lists checking to see if any of them met the criteria. After a few minutes I realized something - we were just brute-forcing our way through the puzzle. So why couldn't I just automate it?

## What It Does

I wrote a simple script that iterates over a list of books and checks to see if it meets the criteria given by the puzzle. I sourced the data on book titles from [OpenLibrary](http://openlibrary.org), which provides [data dumps](https://openlibrary.org/developers/dumps) on all the information in their catalog at any given time. The script itself is pretty simple. I wrote a function that accepts a word and returns any state that's exactly two characters away from it

```python
# Define a function that takes a word and checks to see if it's two characters away from a state
# If it is, it returns the state
# If it's not, it returns False
def word_to_state(word):
	# Make a list of every state
	states = ['Alaska', 'Alabama', 'Arkansas', 'American Samoa', 'Arizona', 'California', 'Colorado', 'Connecticut', 'District of Columbia', 'Delaware', 'Florida', 'Georgia', 'Guam', 'Hawaii', 'Iowa', 'Idaho', 'Illinois', 'Indiana', 'Kansas', 'Kentucky', 'Louisiana', 'Massachusetts', 'Maryland', 'Maine', 'Michigan', 'Minnesota', 'Missouri', 'Northern Mariana Islands', 'Mississippi', 'Montana', 'National', 'North Carolina', 'North Dakota', 'Nebraska', 'New Hampshire', 'New Jersey', 'New Mexico', 'Nevada', 'New York', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Puerto Rico', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Virginia', 'Virgin Islands', 'Vermont', 'Washington', 'Wisconsin', 'West Virginia', 'Wyoming']
	# Iterate over every state to check it against the word
	for state in states:
		# If the length of the state is the same as the length of the word...
		if len(state) == len(word):
			# Convert the word and the state to all lowercase to make it easier to compare them
			word, state = word.lower(), state.lower()

			# Create a counter to check the amount of changes that you make to the word
			changes = 0

			# Iterate over the index and character in each state (Texas -> (0, T) (1, E) (2, X) (3, A), (4, S))
			for index, character in enumerate(state):
				
				# Check to see if the letter in the state is the same as the letter in that position of the word
				if character == word[index]:
					# If it is, you don't need to do anything! Keep going
					pass
				else:
					# If it's not, increment the amount of changes we need to turn the state into the word
					changes += 1

			# After we're done checking to see how many changes it takes to turn a word 
			# into a state, we check to see if it's greater than 2
			if changes > 2:
				# If it is, we continue on to the next state
				continue
			else:
				# If it's not, this is the state we're looking for - we return the state
				# The function will end here if a state can be found
				return state
	
	# If you make it here, no state was found - return False
	return False
```

After writing this function, the rest was simple. We simply iterate over our list of books from OpenLibrary, check to see if each book meets the initial (non-state) criteria, then pass the state through our function and return the results.

```python
# Iterate over every book in the file
for book in books:
	# Figure out how many words are in the books title by splitting the title 
	words_in_book_title = book.split(' ')
	if len(words_in_book_title) == 2:
		# Find the amount of characters in the book title
		characters_in_book_title = ''.join([letter for letter in book if letter.isalpha()]) # 'Infinite Jest' -> 'InfiniteJest'
		if len(characters_in_book_title) == 12:
            # Get the second word in the book title, pass it to word_to_state
			second_word = book.split(' ')[1]
			# word_to_state('TexZZ') -> 'Texas'
			# word_to_state('gibberish1029831') -> False
			if word_to_state(second_word):
				# Print the title of the book and the state if there's a match
				print(book, word_to_state(second_word))
```

After cleaning up the OpenLibrary works dump a bit, I used it as input for the script. A few seconds later, the results were in! Possible titles were:

```
Miami Indians - Miami, IN
Lake Michigan - Lake, MI
Raymond Hains - Raymond, ME
Moon Virginia - Moon, VA
Joseph Crugon - Joseph, OR
Eugene Onegin - Eugene, OR
Richmond Whig - Richmond, OH
Columbus Ohio - Columbus, OH
Garrison town - Garrison, IA
Joseph Gregor - Joseph, OR
```

A little Google-fu (just to confirm which title was a classic book) later and we had the answer - [Eugene Onegin](https://en.wikipedia.org/wiki/Eugene_Onegin), a piece of classic Russian literate published in serial form between 1825 and 1832. Onegin is only two letters away from the state of Oregon. We did it!

After submitting our answer, we waited patiently until the next week for the results. [We were right](http://www.npr.org/2017/07/23/538343376/sunday-puzzle-same-sound-different-meaning)! Although our email wasn't selected as the winning answer, congratulations to Rob Hardy of Dayton, Ohio for getting it - you probably deserved it more ;)

## What's next

Let's be real, `word_to_state` isn't exactly an efficient function. I made it as verbose as possible to try and help my girlfriend who's learning to code understand it. Here's a quick improvement:

```python
def word_to_state(word):
	states = ['Alaska', 'Alabama', 'Arkansas', 'American Samoa', 'Arizona', 'California', 'Colorado', 'Connecticut', 'District of Columbia', 'Delaware', 'Florida', 'Georgia', 'Guam', 'Hawaii', 'Iowa', 'Idaho', 'Illinois', 'Indiana', 'Kansas', 'Kentucky', 'Louisiana', 'Massachusetts', 'Maryland', 'Maine', 'Michigan', 'Minnesota', 'Missouri', 'Northern Mariana Islands', 'Mississippi', 'Montana', 'National', 'North Carolina', 'North Dakota', 'Nebraska', 'New Hampshire', 'New Jersey', 'New Mexico', 'Nevada', 'New York', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Puerto Rico', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Virginia', 'Virgin Islands', 'Vermont', 'Washington', 'Wisconsin', 'West Virginia', 'Wyoming']
	for state in states:
		if len(word) == len(state):
			word, state = word.lower(), state.lower()
			changed_characters = [letter for index, letter in enumerate(word) if state[index] != letter]
			if len(changed_characters) == 2:
				return state.title()
	return False
```

I initially thought set comparisons would be the way to go, but for this challenge, **the order of the letters matters**. The challenge answer of Onegin would've failed using set comparisons, because we're changing the letter 'r' to a letter that's already in the word 'Oregon'. While there are definitely improvments to be made, those are for another day.