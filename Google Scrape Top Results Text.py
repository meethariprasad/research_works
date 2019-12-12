def scope_to_keyphrase(text):
    from rake_nltk import Rake
    r = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
    r.extract_keywords_from_text(text)
    return (' '.join(r.get_ranked_phrases()[0:10]))

def tech_info(supply_list,query_list):
    import urllib
    from bs4 import BeautifulSoup
    import requests
    import webbrowser
    import re

    supply_info_base=[]
    for supply in supply_list:
        
        supply=re.sub('[^A-Za-z0-9]+', ' ', supply)
        supply=re.sub(r'\d+', '', supply)
        print (supply)
        supply_info=[]
        for base_query in query_list:
            ####Query Block##########
            base_query=base_query+' '+supply
            query = urllib.parse.quote_plus(base_query)

            url = 'https://google.com/search?q=' + query

            response = requests.get(url)
            
            ####Result Clean Block##########

            soup = BeautifulSoup(response.text, 'lxml')
            # kill all script and style elements
            for script in soup(["script", "style"]):
                script.extract()    # rip it out

            # get text
            query_results = soup.get_text()

            import re

            query_results=query_results[275:800]
            query_results=re.sub('Verbatim',' ',query_results)
            query_results=re.sub('wikipedia',' ',query_results)
            query_results=re.sub('[^A-Za-z0-9]+', ' ', query_results)
            query_results=''.join(' ' + char if char.isupper() else char for char in query_results).strip()
            ####Result Collection Block##########
            supply_info.append([supply,base_query,query_results])
        supply_info_base.append(supply_info)
        flat_list=[item for sublist in supply_info_base for item in sublist]
        return_frame=pd.DataFrame(flat_list)
        return_frame.columns=['Supply','Query','Results']
    return (return_frame)
            
tech_information=tech_info(supply_list=['Windows','C'],query_list=['What is Software'
                                                                   ,'What are the related Software'])
pd.set_option('display.max_colwidth', -1)
tech_information
