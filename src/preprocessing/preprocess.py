import re
from nltk.corpus import stopwords

class DfCleaner:
    def __init__(self):
        pass
    
    def __clean_text(self, text, remove_stopwords = False):
        text = text.lower()
        
        #Remove unwanted punctuations
        text = re.sub(r'  ', ' ', text)
        text = re.sub(r' — ', ' ', text)
        text = re.sub(r'[_"”\-;%()’“|‘+&=*%.!,?:#$@\[\]/]', ' ', text)
        text = re.sub(r'\'', ' ', text)

        # Remove stop words
        if remove_stopwords:
            text = text.split()
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]
            text = " ".join(text)

        return text
    
    def clean(self, lines,remove_stopwords = False):
        cleaned_lines=[]
        for line in lines:
            cleaned_lines.append(self.__clean_text(line,remove_stopwords))
        return cleaned_lines
