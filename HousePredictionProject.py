import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("C:\\Users\\thsem\\Downloads\\bina-2026-07-28.csv")

df = df.dropna(subset=['NumberOfRooms', 'square', 'Price', 'floor', 'Location', 'Agency'])

df['NumberOfRooms'] = df['NumberOfRooms'].astype(str).str.extract(r'(\d+)').astype(int)
df['square'] = df['square'].astype(str).str.extract(r'(\d+)').astype(int)
df['Price'] = df['Price'].astype(str).str.replace(r'\s+', '', regex=True).astype(float)

df[['current_floor', 'total_floors']] = df['floor'].astype(str).str.split(' ', n=1, expand=True)[0].str.split('/', expand=True)
df['current_floor'] = pd.to_numeric(df['current_floor'])
df['total_floors'] = pd.to_numeric(df['total_floors'])
df = df.drop('floor', axis=1)

df = pd.get_dummies(df, columns=['Location', 'Agency'], drop_first=True)

price_low, price_high = df['Price'].quantile(0.01), df['Price'].quantile(0.99)
square_low, square_high = df['square'].quantile(0.01), df['square'].quantile(0.99)
df = df[(df['Price'] >= price_low) & (df['Price'] <= price_high)]
df = df[(df['square'] >= square_low) & (df['square'] <= square_high)]

X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.2, random_state=42)
X_cv, X_test, y_cv, y_test = train_test_split(X_, y_, test_size=0.5, random_state=42)

linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)

def model_score():
    print("Training Score:", linear_regression_model.score(X_train, y_train))
    print("CV Score:", linear_regression_model.score(X_cv, y_cv))
    print("Test Score:", linear_regression_model.score(X_test, y_test))


def predict_house_price(location, agency, num_rooms, square, current_floor, total_floors, model, feature_columns):
    new_house = pd.DataFrame([{
        'NumberOfRooms': num_rooms,
        'square': square,
        'current_floor': current_floor,
        'total_floors': total_floors,
        'Location': location,
        'Agency': agency
    }])

    new_house_encoded = pd.get_dummies(new_house, columns=['Location', 'Agency'])
    
    new_house_aligned = new_house_encoded.reindex(columns=feature_columns, fill_value=0)
    
    predicted_price = model.predict(new_house_aligned)[0]
    return predicted_price

print("\nEnter the details of the house to predict its price:")
location_ = input("Location: ")
agency_ = input("Agency: (Agentlik or Kompleks) ")
num_rooms_ = int(input("Number of Rooms: "))
square_ = int(input("Square (in m²): "))
current_floor_ = int(input("Current Floor: "))
total_floors_ = int(input("Total Floors: "))

if (current_floor_ > total_floors_):
    print("Invalid floor information. Current floor cannot be greater than total floors.")
elif(current_floor_ < 1 or total_floors_ < 1):
    print("Invalid floor information. Floor numbers cannot be less than 1.")
elif(num_rooms_ <= 0 or square_ <= 0):
    print("Invalid input. Number of rooms and square footage must be positive.")
elif(agency_ not in ['Agentlik', 'Kompleks', 'agentlik', 'kompleks']):
    print("Invalid agency. Please enter either 'Agentlik' or 'Kompleks'.")
else:
    sample_prediction = predict_house_price(location_, agency_, num_rooms_, square_, current_floor_, total_floors_, linear_regression_model, X.columns)
    print(f"\nEstimated Price: {sample_prediction:,.2f} AZN")