import pickle


def sauvegardeFichier(nomFichier, obj):
    with open(nomFichier, "wb") as fichier:
        enregistre = pickle.Pickler(fichier)
        enregistre.dump(obj)


def lectureFichier(nomFichier):
    with open(nomFichier, "rb") as fichier:
        recupere = pickle.Unpickler(fichier)
        return recupere.load()
