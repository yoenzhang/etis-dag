import json

# System and user messages for the OpenAI API
system_message = {
    "role": "system",
    "content": (
        "You are a data extraction assistant tasked with analyzing news articles about wildlife trafficking. "
        "You ONLY reply in valid JSON format and nothing else. Do not include any explanation or extra text. "
        "Extract information about elephant ivory seizures. If the article is not about an elephant seizures/trafficing/poaching, output the JSON with all fields null or an indication that it's not relevant. "
        "Fields to extract:\n"
        "- report_date: the article's publication date (or the pipeline run date if publication date not given).\n"
        "- seizure_date: the date when the seizure occurred (day, month, year).\n"
        "- location: the city/region and country where the seizure took place.\n"
        "- items_seized: a list of items seized (e.g. ivory tusks, elephant hides, trunks, et) with quantities or weights if mentioned.\n"
        "- arrest_count: number of people arrested (if mentioned).\n"
        "- amount_approximate: a boolean, true if the quantities/weights are described approximately (e.g. 'about 100 kg'), false if exact or not applicable.\n"
        "- comment: any other relevant detail (e.g. concealment method, smuggling method, context).\n"
        "- url: the article URL.\n\n"
        "Use null for any field not present. Ensure the JSON keys exactly match the specified fields."
    )
}

# Few-shot example (one-shot) as context
example_article = (
    "Title: Large ivory seizure in Mozambique comes amid worrying signs of increasing elephant poaching\n"
    "Content: Officials intercepted a container at Maputo port on 22 March 2024 and found 651 pieces of elephant ivory concealed in bags of corn, en route to Dubai. "
    "This large seizure was reported on 27 March 2024. No suspects were reported arrested.\n"
    "URL: https://eia-international.org/news/large-ivory-seizure-mozambique-2024"
)
example_output = {
    "report_date": "2024-03-27",
    "seizure_date": "2024-03-22",
    "location": "Maputo, Mozambique",
    "items_seized": [
        {"item": "elephant ivory tusks", "quantity": 651, "unit": "pieces"}
    ],
    "arrest_count": None,
    "amount_approximate": False,
    "comment": "Ivory was concealed in bags of corn, shipment was en route to Dubai.",
    "url": "https://eia-international.org/news/large-ivory-seizure-mozambique-2024"
}

def build_prompt(title, summary, content, url):
    prompt = (
        "You are a data extraction assistant tasked with analyzing news articles about wildlife trafficking. "
        "You ONLY reply in valid JSON format and nothing else. Do not include any explanation or extra text. "
        "Extract information about elephant ivory seizures. If the article is not about an elephant seizures/trafficing/poaching, output the JSON with all fields null or an indication that it's not relevant. "
        "Fields to extract:\n"
        "- report_date: the article's publication date (or the pipeline run date if publication date not given).\n"
        "- seizure_date: the date when the seizure occurred (day, month, year).\n"
        "- location: the city/region and country where the seizure took place.\n"
        "- items_seized: a list of items seized (e.g. ivory tusks, elephant hides, trunks, et) with quantities or weights if mentioned.\n"
        "- arrest_count: number of people arrested (if mentioned).\n"
        "- amount_approximate: a boolean, true if the quantities/weights are described approximately (e.g. 'about 100 kg'), false if exact or not applicable.\n"
        "- comment: any other relevant detail (e.g. concealment method, smuggling method, context).\n"
        "- url: the article URL.\n\n"
        "Use null for any field not present. Ensure the JSON keys exactly match the specified fields.\n\n"
        "Here is an example:\n"
        "Article:\n"
        "Title: Large ivory seizure in Mozambique comes amid worrying signs of increasing elephant poaching\n"
        "Content: Officials intercepted a container at Maputo port on 22 March 2024 and found 651 pieces of elephant ivory concealed in bags of corn, en route to Dubai. "
        "This large seizure was reported on 27 March 2024. No suspects were reported arrested.\n"
        "URL: https://eia-international.org/news/large-ivory-seizure-mozambique-2024\n"
        "Output:\n"
        '{"report_date": "2024-03-27", "seizure_date": "2024-03-22", "location": "Maputo, Mozambique", "items_seized": [{"item": "elephant ivory tusks", "quantity": 651, "unit": "pieces"}], "arrest_count": null, "amount_approximate": false, "comment": "Ivory was concealed in bags of corn, shipment was en route to Dubai.", "url": "https://eia-international.org/news/large-ivory-seizure-mozambique-2024"}'
        "\n\nNow analyze the following article and extract the fields as JSON:\n"
    )
    prompt += f"Title: {title}\n"
    if summary:
        prompt += f"Summary: {summary}\n"
    else:
        prompt += "Summary: None\n"
    prompt += f"Content: {content}\nURL: {url}\n"
    return prompt
