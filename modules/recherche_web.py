# ─────────────────────────────────────────────
# 🌐 Module de recherche web universelle - recherche_web.py
# ─────────────────────────────────────────────
import requests
from bs4 import BeautifulSoup

# 🌐 Recherche Bing
def recherche_web_bing(question: str) -> str:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        url = f"https://www.bing.com/search?q={question.replace(' ', '+')}"
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        resultats = soup.find_all("li", class_="b_algo")

        if resultats:
            message = "🔍 J'ai trouvé ça pour vous (Bing) :\n\n"
            for i, resultat in enumerate(resultats[:3]):
                titre = resultat.find("h2").get_text(strip=True) if resultat.find("h2") else "Titre indisponible"
                lien = resultat.find("a")["href"] if resultat.find("a") else "Lien indisponible"
                message += f"{i+1}. 📌 {titre}\n🔗 {lien}\n\n"
            return message.strip()
        return "🤷 Je n'ai pas trouvé d'information claire sur Bing."

    except Exception as e:
        return f"❌ Erreur pendant la recherche Bing : {e}"

# 🌐 Recherche Wikipédia
def recherche_web_wikipedia(question: str) -> str:
    try:
        url = f"https://fr.wikipedia.org/wiki/{question.replace(' ', '_')}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            return f"🌐 J'ai trouvé un article Wikipédia pour vous :\n🔗 {url}"
        else:
            return "🤷 Je n'ai pas trouvé de page Wikipédia correspondante."

    except Exception as e:
        return f"❌ Erreur pendant la recherche sur Wikipédia : {e}"

# 🔍 Recherche sur Google News (Actualités)
def recherche_web_google_news(question: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://news.google.com/search?q={question.replace(' ', '+')}&hl=fr"
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        resultats = soup.find_all("article", limit=3)

        if resultats:
            message = "📰 J'ai trouvé des actualités pour vous (Google News) :\n\n"
            for i, article in enumerate(resultats):
                titre = article.find("a").get_text(strip=True) if article.find("a") else "Titre indisponible"
                lien = "https://news.google.com" + article.find("a")["href"][1:] if article.find("a") else "Lien indisponible"
                message += f"{i+1}. 📌 {titre}\n🔗 {lien}\n\n"

            return message.strip()

        return "🤷 Je n'ai pas trouvé d'actualités claires sur Google News."

    except Exception as e:
        return f"❌ Erreur pendant la recherche Google News : {e}"



# 🔍 Recherche universelle (Bing > Google > Wikipédia)
def recherche_web_universelle(question: str) -> str:
    print("✅ Recherche universelle lancée :", question)
    
    # ✅ Priorité 1 : Recherche de personnalités (nom connu ou "qui est", "définition")
    if any(mot in question.lower() for mot in ["qui est", "qu'est-ce que", "c'est quoi", "définition"]):
        print("✅ Recherche de personnalité détectée.")

        # ✅ Étape 1 : Recherche avec Bing
        print("✅ [1] Recherche Bing en cours pour :", question)
        result_bing = recherche_web_bing(question)
        print("✅ [1] Résultat Bing brut :", result_bing)
        
        if result_bing and "🤷" not in result_bing and "❌" not in result_bing:
            print("✅ [1] Résultat Bing réussi.")
            return result_bing
        else:
            print("❌ [1] Bing a échoué ou n'a pas trouvé de résultat.")

        # ✅ Étape 2 : Recherche avec Google
        print("✅ [2] Recherche Google en cours pour :", question)
        result_google = recherche_web_google(question)
        print("✅ [2] Résultat Google brut :", result_google)
        
        if result_google and "🤷" not in result_google and "❌" not in result_google:
            print("✅ [2] Résultat Google réussi.")
            return result_google
        else:
            print("❌ [2] Google a échoué ou n'a pas trouvé de résultat.")

        # ✅ Étape 3 : Recherche avec Wikipédia
        print("✅ [3] Recherche Wikipédia en cours pour :", question)
        result_wikipedia = recherche_web_wikipedia(question)
        print("✅ [3] Résultat Wikipédia brut :", result_wikipedia)
        
        if result_wikipedia and "🤷" not in result_wikipedia and "❌" not in result_wikipedia:
            print("✅ [3] Résultat Wikipédia réussi.")
            return result_wikipedia
        else:
            print("❌ [3] Wikipédia a échoué ou n'a pas trouvé de résultat.")

        # ❌ Si toutes les sources échouent
        print("❌ Aucun résultat trouvé pour cette personnalité.")
        return "🤷 Je n'ai pas trouvé d'informations précises sur cette personnalité."

    # ✅ Priorité 2 : Recherche d'actualités avec Google News
    if any(mot in question.lower() for mot in ["nouvelles", "actualités", "dernier", "dernière", "récent", "récentes"]):
        print("✅ [4] Recherche d'actualités détectée, utilisation de Google News.")
        result_news = recherche_web_google_news(question)
        print("✅ [4] Résultat Google News brut :", result_news)
        
        if result_news and "🤷" not in result_news and "❌" not in result_news:
            print("✅ [4] Résultat Google News réussi.")
            return result_news
        else:
            print("❌ [4] Google News a échoué ou n'a pas trouvé de résultat.")

    # ✅ Priorité 3 : Recherche générale avec Bing
    print("✅ [5] Recherche générale Bing en cours pour :", question)
    result_bing = recherche_web_bing(question)
    print("✅ [5] Résultat Bing général brut :", result_bing)
    
    if result_bing and "🤷" not in result_bing and "❌" not in result_bing:
        print("✅ [5] Résultat Bing général réussi.")
        return result_bing

    # ✅ Priorité 4 : Recherche générale avec Google
    print("✅ [6] Recherche générale Google en cours pour :", question)
    result_google = recherche_web_google(question)
    print("✅ [6] Résultat Google général brut :", result_google)
    
    if result_google and "🤷" not in result_google and "❌" not in result_google:
        print("✅ [6] Résultat Google général réussi.")
        return result_google

    # ✅ Priorité 5 : Recherche générale avec Wikipédia
    print("✅ [7] Recherche générale Wikipédia en cours pour :", question)
    result_wikipedia = recherche_web_wikipedia(question)
    print("✅ [7] Résultat Wikipédia général brut :", result_wikipedia)
    
    if result_wikipedia and "🤷" not in result_wikipedia and "❌" not in result_wikipedia:
        print("✅ [7] Résultat Wikipédia général réussi.")
        return result_wikipedia

    # ❌ Si aucune source ne fonctionne
    print("❌ Aucun résultat clair trouvé dans les sources.")
    return "🤷 Je n'ai pas trouvé d'information claire, mais vous pouvez reformuler ou être plus spécifique."




def recherche_score_football(equipe: str) -> str:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # 🌐 Priorité 1 : Flashscore
        url_flashscore = f"https://www.flashscore.fr/recherche/?q={equipe.replace(' ', '+')}"
        response_flash = requests.get(url_flashscore, headers=headers, timeout=5)
        if response_flash.status_code == 200:
            return f"⚽ Résultats sur Flashscore :\n🔗 {url_flashscore}"

        # 🌐 Priorité 2 : Sofascore
        url_sofascore = f"https://www.sofascore.com/fr/recherche/{equipe.replace(' ', '-')}"
        response_sofa = requests.get(url_sofascore, headers=headers, timeout=5)
        if response_sofa.status_code == 200:
            return f"⚽ Résultats sur Sofascore :\n🔗 {url_sofascore}"

        # 🌐 Priorité 3 : Recherche Google si les deux échouent
        url_google = f"https://www.google.com/search?q=score+{equipe.replace(' ', '+')}"
        response_google = requests.get(url_google, headers=headers, timeout=5)
        if response_google.status_code == 200:
            return f"⚽ Résultats sur Google :\n🔗 {url_google}"

        return "🤷 Je n'ai pas trouvé d'information sur les scores de cette équipe."

    except Exception as e:
        return f"❌ Erreur pendant la recherche des scores : {e}"
