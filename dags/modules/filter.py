import re
from dataclasses import dataclass
from typing import List

@dataclass
class Article:
    title: str
    link: str
    published: str
    content: str

def rule_based_filter(articles: List[Article]) -> List[Article]:
    """
    Filters out articles about 'ivory' that are obviously irrelevant 
    (e.g., 'Ivory Coast' sports, 'ivory color dress') and keeps those 
    that mention illicit wildlife trade, elephants, tusks, etc.
    """
    # Define sets of keywords
    positive_keywords = {
        "elephant", "elephants", "tusk", "tusks", "poach", "poaching",
        "wildlife", "illegal trade", "traffick", "trafficking", 
        "smuggl", "seizure", "confiscat", "ivory poaching", "ivory seizure"
    }
    
    # Negative contexts that often appear with "ivory" but are 
    # not about wildlife or illegal trade
    negative_contexts = {
        "ivory coast", "côte d’ivoire",
        "ivory tower", "ivory color", "ivory dress", 
        "ivory wedding", "ivory underwear", "ivory barn", "ivory paint",
        "ivory saree", "ivory trouser", "ivory boxer"
    }
    
    relevant_articles = []
    
    for article in articles:
        combined_text = (article.title + " " + article.content).lower()
        # Strip HTML
        combined_text = re.sub(r"<[^>]+>", "", combined_text)

        # 1) Check if any positive keyword is in the text
        has_positive = any(kw in combined_text for kw in positive_keywords)

        # 2) Check negative contexts
        has_negative = any(neg in combined_text for neg in negative_contexts)

        if has_positive:
            # It might mention "ivory coast" but also "elephants" or "seizure" 
            # => we keep it because it might be relevant.
            relevant_articles.append(article)
        else:
            # If we have no positive keywords but do have negative contexts => discard
            # Or if no positive keywords at all => discard
            # Because we want articles that must mention something about elephants or illegal trade 
            if has_negative:
                # definitely discard
                print("removed x" + article.title)
                pass
            else:
                # no negative, no positive => likely not relevant
                pass
                print("removed x" + article.title)

    return relevant_articles
