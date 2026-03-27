import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Liste des fichiers dans l'ordre
images = ["output/roc_models.png", "output/gain_risque.png"]

for img_path in images:
    img = mpimg.imread(img_path)

    # Créer une figure plus grande
    plt.figure(
        figsize=(12, 8)
    )  # tu peux augmenter les chiffres si tu veux encore plus grand
    plt.imshow(img)
    plt.axis("off")  # masquer les axes
    plt.show()
