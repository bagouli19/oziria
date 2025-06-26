# fonctions_chat.py

def obtenir_reponse_ava(question):
    if not question or not isinstance(question, str):
        return "Je n’ai pas compris votre question. Essayez de reformuler 😊"

    question = question.lower()

    if "blague" in question:
        return "Pourquoi les traders aiment-ils les blagues ? Parce que ça fait des bougies vertes de rire ! 😄"

    elif "motivation" in question:
        return "🔥 L'avenir appartient à ceux qui codent tôt. Continuez !"

    elif "futur" in question:
        return "Le futur est digital, et il s’écrit ligne par ligne."

    else:
        return "Je suis AVA, votre assistante IA. Posez-moi une question sur la bourse, la météo ou les actualités ! 😊"
