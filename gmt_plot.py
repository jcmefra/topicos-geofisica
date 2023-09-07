import pandas as pd
import pygmt
import os

##cargan los datos
data= pd.read_csv('events.csv')
data.latitude = data['lat']
data.longitude = data['lon']
data.st= pd.read_csv('stations_2021-06.txt', sep=',', names= ['net_id', 'sta_id', 'lat', 'lon', 'elev(m)'])
data.st.lat = data.st['lat']
data.st.lon= data.st['lon']
## se cargan datos topográficos utilizando la función load_earth_relief de pygmt  (se almacenan en  grid)
grid = pygmt.datasets.load_earth_relief(resolution="03s", region=region)
dgrid = pygmt.grdgradient(grid=grid, radiance=[270, 30])
#defino la región de interés
region= region = [
    data.longitude.min() ,
    data.longitude.max() ,
    data.latitude.min() ,
    data.latitude.max() ,
]



fig = pygmt.Figure()
pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain")

# paleta de color
pygmt.makecpt(
    cmap='gray',
    series= [-1.5, 0.3, 0.3]
    #continuous=True
)

#mostrar la imagen del relieve terrestre en la región 
fig.grdimage(
    region=region,
    grid= dgrid,
    cmap= True,
    projection="M15c",
    #shading= True,
    #transparency= 80,
    #frame="ag",
)
# agrego la línea costera de la region al mapa
fig.coast(
    region=region,
    projection='M15c',
    #shorelines=True,
    water= "skyblue",
    frame=True,
    #borders="1/0.5p",
    )

#representar los sismos como círculos negros 
fig.plot(
         x=data.longitude,
         y=data.latitude,
         style="c0.03c",
         color="black",
         pen="black",
         #cmap=True
         transparency= 90,
         #label= 'Terremotos'
)
#estaciones como triángulos amarillos.
fig.plot(
         x=data.st.lon,
         y=data.st.lat,
         style="t0.2i",
         color="yellow",
         pen="black",
         #label= 'Estaciones'
)
#agrega una barra de color para los valores topográficos.
fig.colorbar(
    frame='+l"Topografía"'
    )
# un marco y un título al mapa
fig.basemap(frame=["a", "+tParque nacional de Yellowstone"])
fig.show()
