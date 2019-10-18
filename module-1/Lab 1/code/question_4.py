# 4)
from bs4 import BeautifulSoup
import urllib.request
import os

url = "https://scikit-learn.org/stable/modules/clustering.html#clustering"
#makes a request to open the url and set the contents of the url to a variable
source_code = urllib.request.urlopen(url)
plain_text = source_code

#Using BeautifulSoup to parse the html page
soup = BeautifulSoup(plain_text, "html.parser")

print(soup.get_text().strip())






