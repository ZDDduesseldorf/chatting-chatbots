import wikipediaapi

def run_wikipedia_retriever(search_query: str):
    wiki_wiki = wikipediaapi.Wikipedia('en')

    page_py = wiki_wiki.page(search_query)

    summary=page_py.summary[0:200]
    res=page_py.title + " - " + summary.rsplit('.')[0]
    return res

print(run_wikipedia_retriever('Bikes'))
