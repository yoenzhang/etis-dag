import requests
from bs4 import BeautifulSoup

url = "https://www.hindustantimes.com/cities/others/government-railway-police-grp-seizes-2-3kg-elephant-tusk-from-lachit-express-train-in-assam-arrests-smuggler-101692717988667.html"

url2 = "https://www.citizen.co.za/lowvelder/lnn/article/3-arrested-for-possession-of-elephant-tusks-in-joburg/"

url3 = "https://houseofheat.co/nike/nike-dunk-low-next-nature-light-smoke-grey-pink-rise-pale-ivory-dd1873-002"


# Fake browser header to bypass bot protection
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
}

resp = requests.get(url3, headers=headers, timeout=10)

if resp.status_code == 200:
    soup = BeautifulSoup(resp.text, "html.parser")
    paragraphs = soup.find_all("p")
    full_text = "\n\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())

    print("\n" + "=" * 80)
    print(" ARTICLE CONTENT ".center(80, "="))
    print("=" * 80 + "\n")
    print(full_text)
    print("\n" + "=" * 80)
else:
    print(f"Failed to fetch the page. Status code: {resp.status_code}")
