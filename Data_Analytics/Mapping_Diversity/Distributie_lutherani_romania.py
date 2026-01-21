import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import os

# Schimbam directorul de lucru la locatia scriptului
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Citim datele religioase din Excel
df_excel = pd.read_excel('rel1.xlsx', header=None)
df_excel.columns = df_excel.iloc[3]
df_data = df_excel.iloc[6:].copy()
df_data.reset_index(drop=True, inplace=True)

# Coloana 0 = Judet, 14 = Lutherana, 17 = Augustana
df_data.columns = list(range(len(df_data.columns)))
df_data = df_data[[0, 14, 17]].copy()
df_data.columns = ['Judet', 'Evanghelica_Lutherana', 'Evanghelica_Augustana']

df_data = df_data[df_data['Judet'].notna()].copy()

# Convertim la numeric
df_data['Evanghelica_Lutherana'] = pd.to_numeric(df_data['Evanghelica_Lutherana'], errors='coerce').fillna(0)
df_data['Evanghelica_Augustana'] = pd.to_numeric(df_data['Evanghelica_Augustana'], errors='coerce').fillna(0)

df_data['Total_Lutherani'] = df_data['Evanghelica_Lutherana'] + df_data['Evanghelica_Augustana']
df_data = df_data[df_data['Total_Lutherani'] > 0].copy()
df_data['Judet'] = df_data['Judet'].str.strip()

print(f"Total Lutherani in Romania: {int(df_data['Total_Lutherani'].sum()):,}")
print(f"\nTop 5 judete:")
print(df_data.nlargest(5, 'Total_Lutherani')[['Judet', 'Total_Lutherani']])

# Incarcam GeoJSON
gdf = gpd.read_file('ro_uat_poligon.geojson')

# Mapam datele
judet_dict_lutherani = dict(zip(df_data['Judet'], df_data['Total_Lutherani']))
judet_dict_lutherani_normalized = {k.strip().upper(): v for k, v in judet_dict_lutherani.items()}

gdf['county_normalized'] = gdf['county'].str.strip().str.upper()
gdf['Total_Lutherani'] = gdf['county_normalized'].map(judet_dict_lutherani_normalized).fillna(0)

# Cream figura
fig, ax = plt.subplots(1, 1, figsize=(10, 12))

# Cream o schema de culori personalizata: alb pentru 0, rosu inchis pentru valori
cmap = colors.LinearSegmentedColormap.from_list('custom_red', ['white', '#8B0000', '#550000'])

# Plotam harta
gdf.plot(column='Total_Lutherani', 
         ax=ax, 
         legend=True,
         cmap=cmap,
         edgecolor='white',
         linewidth=0.2,
         vmin=0,
         vmax=df_data['Total_Lutherani'].max(),
         legend_kwds={'label': "Numar Lutherani per Judet",
                     'orientation': "horizontal",
                     'shrink': 0.5,
                     'pad': 0.05})

# Adaugam etichete pentru orasele importante
max_value = df_data['Total_Lutherani'].max()
for idx, row in gdf.iterrows():
    if row['Total_Lutherani'] > 100 and (row.get('pop2020', 0) > 20000 or row.get('natLevName') in ['Municipiu', 'Oras']):
        centroid = row['geometry'].centroid
        # Determinam culoarea textului bazat pe intensitatea valorii
        # Daca valoarea e mai mare de 50% din max, folosim alb, altfel rosu inchis
        text_color = 'white' if row['Total_Lutherani'] > (max_value * 0.3) else '#8B0000'
        ax.annotate(text=f"{row['name']}\n({int(row['Total_Lutherani'])})",
                   xy=(centroid.x, centroid.y),
                   ha='center',
                   fontsize=6,
                   color=text_color)

ax.axis('off')
ax.set_title('Distributia Confesiunii Lutherane in Romania (2021)', 
            fontsize=20, 
            fontweight='bold',
            pad=20)

plt.tight_layout()
plt.savefig('harta_lutherani_matplotlib.png', dpi=300, bbox_inches='tight')
print("\n✓ Harta salvata ca 'harta_lutherani_matplotlib.png'")
plt.show()
