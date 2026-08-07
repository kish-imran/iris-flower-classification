from predict import predict_flower

while True:

    print("\n" + "=" * 40)
    print("   IRIS FLOWER CLASSIFICATION")
    print("=" * 40)

    print("1. Predict Flower Species")
    print("2. About Project")
    print("3. Exit")

    choice = input("\nEnter Your Choice: ")

    if choice == "1":
        predict_flower()

    elif choice == "2":
        print("\nProject Name : Iris Flower Classification")
        print("Algorithm    : K-Nearest Neighbors (KNN)")
        print("Language     : Python")
        print("Library      : Scikit-learn")
        print("Dataset      : Iris Dataset")
        print("Purpose      : Predict the species of an Iris flower.")

    elif choice == "3":
        print("\nThank you for using the Iris Flower Classification System!")
        break

    else:
        print("\nInvalid Choice! Please try again.")