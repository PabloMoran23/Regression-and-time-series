import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
 
# Como no disponemos de nuevos datos para este ejemplo vamos a usar los mismos datos
# que usamos para entrenar el modelo.
# También se asume que los datos vienen en el mismo formato que los datos originales,
# Por lo que habrá que realizarles las mismas transformaciones, excepto eliminar outliers,
# Que los quitamos en un principio para que no perjudicaran al entrenamiento.

#### Leer los datos de entrada

data = pd.read_csv("C:\\Users\\Pablo\\Downloads\\archive(1)\\Car details v3.csv")

# Eliminamos la variable objetivo
input_data = data.drop(["selling_price"], axis=1)


# Transformacion de los datos

input_data["mileage_num"] = input_data.mileage.str.extract('([+-]?[0-9]*[.]?[0-9]+)').astype(float)
input_data["engine_num"] = input_data.engine.str.extract('([+-]?[0-9]*[.]?[0-9]+)').astype(float)
input_data["max_power_num"] = input_data.max_power.str.extract('([+-]?[0-9]*[.]?[0-9]+)').astype(float)
input_data["torque_num"] = input_data.torque.str.extract('([0-9]*[.]?[0-9]+(?=\s*Nm))').astype(float)
input_data["torque_rpm"] = input_data.torque.str.extract('([0-9]*[.]?[0-9]+(?=\s*rpm))').astype(float)


input_data["year"] = 2021 - input_data["year"]


input_data["brand"] = input_data.name.str.split(" ").apply(lambda x: x[0])

transmission = input_data.pivot_table(index=input_data.index,columns="transmission", 
                                 values="year", aggfunc=lambda x: len(x.unique())).fillna(0)
fuel = input_data.pivot_table(index=input_data.index,columns="fuel", 
                                 values="year", aggfunc=lambda x: len(x.unique())).fillna(0)
seller_type = input_data.pivot_table(index=input_data.index,columns="seller_type", 
                                 values="year", aggfunc=lambda x: len(x.unique())).fillna(0)
brand = input_data.pivot_table(index=input_data.index,columns="brand", 
                                 values="year", aggfunc=lambda x: len(x.unique())).fillna(0)
input_data = input_data.join(transmission).join(fuel).join(seller_type).join(brand)

owner_dict = {"First Owner": 1, "Second Owner": 2, "Third Owner": 3, "Fourth & Above Owner": 4, "Test Drive Car": 0}
input_data = input_data.replace({"owner": owner_dict})

input_data = input_data.drop(["name","transmission", "fuel", "seller_type", "brand", "mileage", "torque", "engine", "max_power"], axis=1)
input_data = input_data.fillna(0)

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X =scaler.fit_transform(input_data)

# Prediccion

filename = 'C:\\Users\\Pablo\\Desktop\\Capgemini\\car_price_estimator.sav'
loaded_model = pickle.load(open(filename, 'rb'))
pred = loaded_model.predict(X)

data["estimated_price"] = pred

# Guardado de la prediccion
# En principio se sobreescribe el csv de salida, ya que es un csv y dependiendo del 
# volumen de datos y la periodicidad de las ejecuciones no es conveniente hacerlo incremental
# Aunque si fuera necesario sería tan sencillo como leer el antiguo y hacer un append 
# con las predicciones nuevas. 

data.to_csv("C:\\Users\\Pablo\\Desktop\\Capgemini\\Car_estimated_price.csv")


# Este script se puede automatizar en crontab o herramientas similares con la periodicidad que sea necesaria,
# y dependiendo del entorno del cliente se tendria que ver la manera de integrarlo,
# desde productivizarlo directamente en sus sistemas, en cloud o a través de un contenedor
# docker (incluyendo en el tanto este script como el modelo).