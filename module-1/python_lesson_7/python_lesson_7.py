from bs4 import BeautifulSoup
import requests


html_payload = requests.get('https://en.wikipedia.org/wiki/Google')
parsed_html = BeautifulSoup(html_payload.content, 'html.parser')

# removed junk text from the parse tree, ex: javascript, css, etc.
[s.extract() for s in parsed_html(['style', 'script', '[document]', 'head', 'title'])]

# Extract text only from paragraph tags
paragraphs = parsed_html.find_all('p')

# removing anywhere reference numbers occur. ie: [1][34][5]
import re
final_text = ' '.join([text.getText() for text in paragraphs if text.getText().strip()])
final_text = re.sub('\\[.+\\]', '', final_text)

with open('input.txt', 'w') as file:
    file.write(final_text)

# Tokenization
import nltk
sentence_tokens = nltk.sent_tokenize(final_text)
word_tokens = nltk.word_tokenize(final_text)

print('Printing sentence tokens')
for sentence in sentence_tokens[:20]:
    print(sentence)

print('\n\n\nPrinting word tokens')
for word in word_tokens[:20]:
    print(word)

# Part of Speech
print('Printing all of the parts of speech')
print(nltk.pos_tag(word_tokens))

from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer

# Stemming

# stem each word and place into a set to only show unique words
PStemmer = PorterStemmer()
p_stem_words = [PStemmer.stem(word) for word in word_tokens]
LStemmer = LancasterStemmer()
l_stem_words = [LStemmer.stem(word) for word in word_tokens]
SStemmer = SnowballStemmer('english')
s_stem_words = [SStemmer.stem(word) for word in word_tokens]

print('Porter Stemming:', p_stem_words[:20])
print('Lancaster Stemming: ', l_stem_words[:20])
print('Snowball Stemming: ', s_stem_words[:20])

# Lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lem_words = [lemmatizer.lemmatize(word) for word in word_tokens]
print('Lemmantization: ', lem_words[:20])

# Trigram
# we are only going to select the first 20
from nltk.util import trigrams
grams = [t_gram for x, t_gram in enumerate(trigrams(lem_words)) if x < 20]
print('Trigrams: ', grams[:20])

# Named Entity Recognition
from nltk import wordpunct_tokenize, pos_tag, ne_chunk
print('Named Entity Recognition: ', ne_chunk(pos_tag(wordpunct_tokenize(final_text)))[:20])

# see the text_classification file for classifier changes