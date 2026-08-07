import joblib

# Load Trained Model
try:
    model = joblib.load("model/iris_model.pkl")
except FileNotFoundError:
    print("Error: Trained model not found!")
    print("Please run train_model.py first.")
    exit()


def predict_flower():
    print("\n==============================")
    print("IRIS FLOWER PREDICTION")
    print("==============================")

    try:
        sepal_length = float(input("Enter Sepal Length (cm): "))
        sepal_width = float(input("Enter Sepal Width (cm): "))
        petal_length = float(input("Enter Petal Length (cm): "))
        petal_width = float(input("Enter Petal Width (cm): "))
    except ValueError:
        print("\n❌ Invalid input! Please enter numeric values.")
        return

    data = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    prediction = model.predict(data)

    print("\n==============================")
    print("PREDICTION RESULT")
    print("==============================")
    print("Predicted Flower Species:", prediction[0])


# Run prediction if this file is executed directly
if __name__ == "__main__":
    predict_flower()