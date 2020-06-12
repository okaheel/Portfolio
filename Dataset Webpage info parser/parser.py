from bs4 import BeautifulSoup
from urllib.request import urlopen
import networkx as nx

def get_only_text(url):
    """ 
    return text from dataset
    at the specified url
    """
    page = urlopen(url)
    soup = BeautifulSoup(page, "lxml")
    text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
  
    print ("=====================")
    print (text)
    print ("=====================")
 
    return soup.title.text, text    
 
     
url="https://data.eol.ucar.edu/project/TORUS_2019"
text = get_only_text(url)