import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import matplotlib.patheffects as PathEffects
import os

# Schimbam directorul de lucru la locatia scriptului
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Citim datele religioase din Excel
df_excel = pd.read_excel('rel1.xlsx', header=None)
df_excel.columns = df_excel.iloc[3]
df_data = df_excel.iloc[6:].copy()
df_data.reset_index(drop=True, inplace=True)

# Coloana 0 = Judet, 4 = Reformata
df_data.columns = list(range(len(df_data.columns)))
df_data = df_data[[0, 4]].copy()
df_data.columns = ['Judet', 'Reformata']

df_data = df_data[df_data['Judet'].notna()].copy()

# Convertim la numeric
df_data['Reformata'] = pd.to_numeric(df_data['Reformata'], errors='coerce').fillna(0)

df_data = df_data[df_data['Reformata'] > 0].copy()
df_data['Judet'] = df_data['Judet'].str.strip()

print(f"Total Reformati in Romania: {int(df_data['Reformata'].sum()):,}")
print(f"\nTop 5 judete:")
print(df_data.nlargest(5, 'Reformata')[['Judet', 'Reformata']])

# Incarcam GeoJSON
gdf = gpd.read_file('ro_uat_poligon.geojson')

# Mapam datele
judet_dict_reformati = dict(zip(df_data['Judet'], df_data['Reformata']))
judet_dict_reformati_normalized = {k.strip().upper(): v for k, v in judet_dict_reformati.items()}

gdf['county_normalized'] = gdf['county'].str.strip().str.upper()
gdf['Reformata'] = gdf['county_normalized'].map(judet_dict_reformati_normalized).fillna(0)

# Cream figura
fig, ax = plt.subplots(1, 1, figsize=(10, 12))

# Cream o schema de culori personalizata: alb pentru 0, verde inchis pentru valori
cmap = colors.LinearSegmentedColormap.from_list('custom_green', ['white', "#034903", "#052105"])

# Plotam harta
gdf.plot(column='Reformata', 
         ax=ax, 
         legend=True,
         cmap=cmap,
         edgecolor="#D7F3D7",
         linewidth=0.2,
         vmin=0,
         vmax=df_data['Reformata'].max(),
         legend_kwds={'label': "Numar Reformati per Judet",
                     'orientation': "horizontal",
                     'shrink': 0.5,
                     'pad': 0.05})

# Adaugam etichete pentru orasele importante
max_value = df_data['Reformata'].max()
for idx, row in gdf.iterrows():
    if row['Reformata'] > 100 and (row.get('pop2020', 0) > 20000 or row.get('natLevName') in ['Municipiu', 'Oras']):
        centroid = row['geometry'].centroid
        # Determinam culoarea textului bazat pe intensitatea valorii
        # Daca valoarea e mai mare de 30% din max, folosim alb, altfel verde inchis
        text_color = 'white' if row['Reformata'] > (max_value * 0.3) else '#006400'
        text_obj = ax.annotate(text=f"{row['name']}\n({int(row['Reformata'])})",
                   xy=(centroid.x, centroid.y),
                   ha='center',
                   fontsize=7 if text_color == 'white' else 6,
                   weight='bold' if text_color == 'white' else 'normal',
                   color=text_color)
        # Adaugam contur negru pentru textul alb
        if text_color == 'white':
            text_obj.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])

ax.axis('off')
ax.set_title('Distributia Confesiunii Reformate in Romania (2021)', 
            fontsize=20, 
            fontweight='bold',
            pad=20)

plt.tight_layout()
plt.savefig('harta_reformati_matplotlib.png', dpi=300, bbox_inches='tight')
print("\n✓ Harta salvata ca 'harta_reformati_matplotlib.png'")
plt.show()


