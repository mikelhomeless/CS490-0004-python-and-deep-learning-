file_name = input('Enter file path you want to open: ')

# the test file was converted to utf-8
text = ''
with open(file_name) as file:
    text = file.read()

# tokenize and apply lemmatization
import nltk
word_tokens = nltk.word_tokenize(text)

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lem_words = [lemmatizer.lemmatize(word) for word in word_tokens]

# find all trigrams
from nltk.util import trigrams
trigrams = trigrams(lem_words)

from collections import defaultdict
trigram_dictionaries = defaultdict(int)
for x in trigrams:
    trigram_dictionaries[x] += 1

top_ten_trigrams = sorted(trigram_dictionaries.items(), key=lambda x: x[1], reverse=True)[:10]
print('Top ten trigrams')
print(top_ten_trigrams)

# extracting all the sentences in the file
sentence_tokens = nltk.sent_tokenize(text)


# gather all sentences that contain the top ten trigrams. Some sentences are repeated twice because they contain multiple of the common trigrams
sentences = [sentence for trigram in top_ten_trigrams for sentence in sentence_tokens if ' '.join(trigram[0]) in sentence]
print('sentences that contain the trigrams')
print(' '.join(sentences))