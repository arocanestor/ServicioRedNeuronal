import os
from Service.OverpassService import nearbySearch
from Service.geocodificaciónGoogle import convertirDireccion
from Service.geocodificaciónGoogle import geocodeAddress
from Service.OverpassService import consultar_osm
from Controller.CNT_IniciarEntrenamiento import CargarData
from Controller.modelo import LocationClassifier
from metadata import DATOS_ENTRENAMIENTO,ARCHIVO_SALIDA
import sys
from performance_middleware import performance_logger
from fastapi import FastAPI, Request

app = FastAPI()
# Agrega el middleware
app.middleware("http")(performance_logger)


def main():
    
    #nearbySearch()
    #convertirDireccion()
    CargarData()
    # ========== MODO 1: ENTRENAMIENTO ==========
    print("\n" + "🔹" * 35)
    print("MODO: ENTRENAMIENTO")
    print("🔹" * 35 + "\n")
    
    # Crear instancia del clasificador
    clasificador = LocationClassifier(model_dir="./Modelo")
    
    # Entrenar con el nuevo formato CSV
    metricas = clasificador.entrenar(
        csv_path=os.path.join(ARCHIVO_SALIDA, "datosEntrenamiento.csv"),  # Ruta a tu CSV
        sep=',',  # Separador del CSV
        target_column='tipo',  # Columna con las etiquetas
        exclude_columns=['nombre'],  # Columnas a no usar como características
        epochs=150,
        batch_size=16,
        early_stopping=True,
        patience=15
    )
    
    print(f"\n📊 Métricas finales:")
    for key, value in metricas.items():
        print(f"   {key}: {value}")
    
    
    # # ========== MODO 2: PREDICCIÓN ==========
    # print("\n\n" + "🔹" * 35)
    # print("MODO: PREDICCIÓN")
    # print("🔹" * 35 + "\n")
    
    # Cargar modelo entrenado (en una nueva sesión)
    clasificador_prediccion = LocationClassifier(model_dir="./Modelo")
    clasificador_prediccion.cargar_modelo()
    
    # Ejemplo 1: Predecir desde diccionario
    datos_nuevos ={
        "nombre": "Rock & Wine",
        "lat": 7.1188058,
        "lon": -73.11768479999999,
        "Abandoned": 0,
        "Administration": 0,
        "Agrarian": 0,
        "Alcohol": 0,
        "Antiques": 0,
        "Apartments": 5,
        "Appliance": 0,
        "Art": 0,
        "Arts Centre": 0,
        "Atm": 0,
        "Baby Goods": 0,
        "Bag": 1,
        "Bakery": 0,
        "Bakery Vegan Bakery Vegan": 0,
        "Baking Oven": 0,
        "Bank": 0,
        "Bar": 0,
        "Barn": 0,
        "Bathroom Furnishing": 0,
        "Beauty": 0,
        "Bed": 1,
        "Beverages": 0,
        "Bicycle": 0,
        "Bicycle Rental": 0,
        "Bicycle Rental;Toilets": 0,
        "Bookmaker": 1,
        "Books": 0,
        "Boutique": 0,
        "Bridge": 0,
        "Bridleway": 0,
        "Buil": 0,
        "Building Materials": 0,
        "Bus Station": 1,
        "Bus Stop": 0,
        "Busway": 0,
        "Butcher": 0,
        "Cabinet Maker": 0,
        "Cafe": 2,
        "Cake": 0,
        "Candles": 0,
        "Cannabis": 0,
        "Car": 0,
        "Car Parts": 0,
        "Car Rental": 0,
        "Car Repair": 1,
        "Car Wash": 0,
        "Carpet": 0,
        "Casino": 0,
        "Chapel": 0,
        "Charity": 0,
        "Cheese": 0,
        "Chemist": 0,
        "Chocolate":0,
        "Childcare": 0,
        "Church": 0,
        "Cinema": 0,
        "Civic": 0,
        "Clinic": 0,
        "Clothes": 3,
        "Coffee": 3,
        "Collector": 0,
        "College": 0,
        "Commercial": 2,
        "Community Centre": 0,
        "Computer": 0,
        "Confectionery": 0,
        "Conference Centre": 0,
        "Construction": 0,
        "Convenience": 1,
        "Copyshop": 1,
        "Corridor": 0,
        "Cosmetics": 1,
        "Courthouse": 0,
        "Craft": 0,
        "Crematorium": 1,
        "Curtain": 0,
        "Cycleway": 15,
        "Dairy": 0,
        "Deli": 0,
        "Dentist": 0,
        "Department Store": 0,
        "Desechables": 0,
        "Detached": 0,
        "Doctors": 2,
        "Doityourself": 1,
        "Dojo": 1,
        "Driving School": 0,
        "Dry Cleaning": 0,
        "E-Cigarette": 0,
        "Electrical": 0,
        "Electronics": 0,
        "Elevator": 0,
        "Erotic": 0,
        "Events Venue": 2,
        "Exhibition Centre": 0,
        "Fabric": 0,
        "Farm": 0,
        "Farmacia": 0,
        "Fashion Accessories": 0,
        "Fast Food": 5,
        "Fincanon & Cia": 0,
        "Fire Station": 0,
        "Fishing": 0,
        "Flooring": 0,
        "Florist": 0,
        "Food": 0,
        "Food Court": 0,
        "Footway": 96,
        "Frame": 0,
        "Frozen Food": 0,
        "Fuel": 8,
        "Funeral Directors": 0,
        "Funeral Hall": 0,
        "Furniture": 1,
        "Garage": 0,
        "Garages": 0,
        "Garden Centre": 0,
        "Gas": 0,
        "Gatehouse": 0,
        "General": 0,
        "Gift": 0,
        "Glaziery": 0,
        "Golf": 0,
        "Government": 0,
        "Grandstand": 0,
        "Greengrocer": 0,
        "Greenhouse": 0,
        "Grocery": 1,
        "Hairdresser": 0,
        "Handicraft": 0,
        "Hangar": 3,
        "Hardware": 0,
        "Health Food": 0,
        "Hearing Aids": 0,
        "Herbalist": 0,
        "Hifi": 0,
        "Hobby": 0,
        "Hospital": 0,
        "Hotel": 3,
        "House": 0,
        "Household": 0,
        "Household Linen": 0,
        "Houseware": 0,
        "Hut": 0,
        "Ice Cream": 0,
        "Industrial": 0,
        "Interior Decoration": 0,
        "Internet Cafe": 0,
        "Jewelry": 0,
        "Juice Bar": 0,
        "Kindergarten": 0,
        "Kiosk": 0,
        "Kitchen": 0,
        "Ladder": 0,
        "Language School": 0,
        "Laundry": 0,
        "Leather": 0,
        "Library": 0,
        "Licorera": 0,
        "Lighting": 0,
        "Living Street": 0,
        "Locksmith": 0,
        "Lottery": 0,
        "Love Hotel": 0,
        "Mall": 0,
        "Marketplace": 0,
        "Massage": 0,
        "Medical Supply": 0,
        "Mensajería Y Domicilios": 0,
        "Mobile Phone": 0,
        "Monastery": 0,
        "Money Lender": 0,
        "Money Transfer": 0,
        "Mortuary": 1,
        "Motorcycle": 0,
        "Motorcycle Parking": 0,
        "Motorcycle Repair": 0,
        "Music": 0,
        "Music School": 0,
        "Musical Instrument": 0,
        "Newsagent": 0,
        "Nightclub": 0,
        "No": 0,
        "Nursing Home": 0,
        "Nutrition Supplements": 0,
        "Office": 0,
        "Optician": 0,
        "Outbuilding": 1,
        "Outdoor": 0,
        "Paga Todo": 0,
        "Paint": 0,
        "Park": 0,
        "Parking": 0,
        "Parking Entrance": 0,
        "Party": 0,
        "Pasta": 0,
        "Pastry": 1,
        "Path": 0,
        "Pawnbroker": 0,
        "Payment Centre": 0,
        "Payment Terminal": 8,
        "Pedestrian": 0,
        "Perfumery": 0,
        "Pet": 0,
        "Pet Grooming": 0,
        "Pharmacy": 0,
        "Photo": 0,
        "Photo Studio": 0,
        "Pitch": 0,
        "Place Of Worship": 0,
        "Planetarium": 2,
        "Platform": 0,
        "Police": 0,
        "Pool": 5,
        "Posgrados Universidad Santo Tomas": 1,
        "Post Office": 0,
        "Pottery": 0,
        "Primary": 0,
        "Primary Link": 0,
        "Printer Ink": 0,
        "Printing": 0,
        "Proposed": 0,
        "Pub": 14,
        "Public": 1,
        "Public Bath": 0,
        "Public Building": 0,
        "Raceway": 0,
        "Radiotechnics": 0,
        "Recycling": 0,
        "Religion": 0,
        "Repair": 0,
        "Residential": 60,
        "Residential;Yes": 0,
        "Restaurant": 11,
        "Retail": 0,
        "Roof": 0,
        "Ruins": 0,
        "School": 16,
        "Scrapyard": 0,
        "Seafood": 0,
        "Second Hand": 0,
        "Secondary": 30,
        "Secondary Link": 6,
        "Semidetached House": 0,
        "Service": 27,
        "Services": 0,
        "Sewing": 0,
        "Shed": 0,
        "Shelter": 0,
        "Shoes": 0,
        "Social Centre": 0,
        "Social Facility": 0,
        "Spices": 3,
        "Sports": 0,
        "Sports Centre": 0,
        "Sports Hall": 0,
        "Stadium": 0,
        "Station": 1,
        "Stationery": 3,
        "Steps": 1,
        "Storage Rental": 0,
        "Storage Tank": 0,
        "Studio": 5,
        "Supermarket": 0,
        "Swimming Pool": 1,
        "Tailor": 0,
        "Tattoo": 0,
        "Taxi": 0,
        "Tea": 0,
        "Tecnologia": 0,
        "Telecommunication": 0,
        "Temple": 30,
        "Terrace": 1,
        "Tertiary": 0,
        "Tertiary Link": 0,
        "Theatre": 0,
        "Ticket": 0,
        "Tiles": 0,
        "Tobacco": 0,
        "Toilets": 0,
        "Townhall": 0,
        "Toys": 0,
        "Track": 0,
        "Trade": 0,
        "Train Station": 0,
        "Transportation": 20,
        "Travel Agency": 0,
        "Trunk": 0,
        "Trunk Link": 0,
        "Tyres": 0,
        "Unclassified": 0,
        "University": 0,
        "Vacant": 0,
        "Variety Store": 0,
        "Vehicle Inspection": 0,
        "Veterinary": 1,
        "Video": 0,
        "Video Games": 0,
        "Warehouse": 0,
        "Watches": 0,
        "Water Tank": 0,
        "Water Tower": 1,
        "Wholesale": 0,
        "Window Blind": 12,
        "Wine": 0,
        "Yes": 0,
        "tipo": "ZonaBares"
    }
    
    resultado = clasificador_prediccion.predecir(
        datos=datos_nuevos,
        mostrar_probabilidades=True
    )
    
    print("\n🎯 RESULTADO DE LA PREDICCIÓN:")
    print(f"   Ubicación: ({resultado.get('lat')}, {resultado.get('lon')})")
    print(f"   Categoría: {resultado['prediccion']}")
    print(f"   Confianza: {resultado['confianza']*100:.2f}%")
    
 
def manual():
    """
    Método que permite ingresar direcciones manualmente desde el teclado.
    Escribe 'salir' para terminar la ejecución.
    """
    print("=== Modo Manual de Consulta de Direcciones ===")
    print("Escribe 'salir' para terminar\n")
    
    while True:
        # Solicitar dirección al usuario
        direccion = input("Ingresa una dirección: ").strip()
        
        # Verificar si el usuario quiere salir
        if direccion.lower() == 'salir':
            print("Saliendo del modo manual...")
            break
        
        # Validar que no sea una entrada vacía
        if not direccion:
            print("Por favor, ingresa una dirección válida.\n")
            continue
        
        try:
            print(f"\nProcesando dirección: {direccion}")
            
            # Convertir dirección a coordenadas
            lat, lng = geocodeAddress(direccion)
            
            if lat is None or lng is None:
                print("No se pudo geocodificar la dirección. Intenta con otra.\n")
                continue
            
            print(f"Coordenadas obtenidas: Lat={lat}, Lng={lng}")
            
            # Consultar datos de OSM
            dataJson = consultar_osm(lat, lng)
            
            # Realizar predicción
            clasificador_prediccion = LocationClassifier(model_dir="./Modelo")
            clasificador_prediccion.cargar_modelo()
            resultado = clasificador_prediccion.predecir(
                datos=dataJson,
                mostrar_probabilidades=True
            )
            
            print(f"   Categoría: {resultado['prediccion']}")
            print(f"   Confianza: {resultado['confianza']*100:.2f}%")
            print("-" * 50 + "\n")
            
        except Exception as e:
            print(f"Error al procesar la dirección: {str(e)}\n")
            continue

def invocarRedNeuronal(direccion):
    """
    Método que permite invocar red neuronal 
    """
    # Convertir dirección a coordenadas
    lat, lng = geocodeAddress(direccion)
    if lat is None or lng is None:
        print("No se pudo geocodificar la dirección. Intenta con otra.\n")        
            
    print(f"Coordenadas obtenidas: Lat={lat}, Lng={lng}")
    
    # Consultar datos de OSM
    dataJson = consultar_osm(lat, lng)
    
    # Realizar predicción
    clasificador_prediccion = LocationClassifier(model_dir="./Modelo")
    clasificador_prediccion.cargar_modelo()
    resultado = clasificador_prediccion.predecir(
        datos=dataJson,
        mostrar_probabilidades=True
    )
    
    print(f"   Categoría: {resultado['prediccion']}")
    print(f"   Confianza: {resultado['confianza']*100:.2f}%")
    print("-" * 50 + "\n")
    return resultado['prediccion']
    
if __name__ == "__main__":
    # Capturar la dirección de los argumentos
    direccion = " ".join(sys.argv[1:])
    print(f"Procesando dirección: {direccion}\n")
    
    # Llamar a la función con el parámetro
    invocarRedNeuronal(direccion)



