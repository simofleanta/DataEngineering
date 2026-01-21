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

# Coloana 0 = Judet, 5 = Greco-Catolica (verifica numarul coloanei)
df_data.columns = list(range(len(df_data.columns)))
df_data = df_data[[0, 5]].copy()
df_data.columns = ['Judet', 'Grecocatolica']

df_data = df_data[df_data['Judet'].notna()].copy()

# Convertim la numeric
df_data['Grecocatolica'] = pd.to_numeric(df_data['Grecocatolica'], errors='coerce').fillna(0)

df_data = df_data[df_data['Grecocatolica'] > 0].copy()
df_data['Judet'] = df_data['Judet'].str.strip()

print(f"Total Greco-Catolici in Romania: {int(df_data['Grecocatolica'].sum()):,}")
print(f"\nTop 5 judete:")
print(df_data.nlargest(5, 'Grecocatolica')[['Judet', 'Grecocatolica']])

# Incarcam GeoJSON
gdf = gpd.read_file('ro_uat_poligon.geojson')

# Mapam datele
judet_dict_grecocatolica = dict(zip(df_data['Judet'], df_data['Grecocatolica']))
judet_dict_grecocatolica_normalized = {k.strip().upper(): v for k, v in judet_dict_grecocatolica.items()}

gdf['county_normalized'] = gdf['county'].str.strip().str.upper()
gdf['Grecocatolica'] = gdf['county_normalized'].map(judet_dict_grecocatolica_normalized).fillna(0)

# Cream figura
fig, ax = plt.subplots(1, 1, figsize=(10, 12))

# Cream o schema de culori personalizata: alb pentru 0, albastru inchis pentru valori
cmap = colors.LinearSegmentedColormap.from_list('custom_blue', ['white', "#0B69A0", "#043850"])

# Plotam harta
gdf.plot(column='Grecocatolica', 
         ax=ax, 
         legend=True,
         cmap=cmap,
         edgecolor="#C6E1F1",
         linewidth=0.2,
         vmin=0,
         vmax=df_data['Grecocatolica'].max(),
         legend_kwds={'label': "Numar Greco-Catolici per Judet",
                     'orientation': "horizontal",
                     'shrink': 0.5,
                     'pad': 0.05})

# Adaugam etichete pentru orasele importante
max_value = df_data['Grecocatolica'].max()
for idx, row in gdf.iterrows():
    if row['Grecocatolica'] > 100 and (row.get('pop2020', 0) > 20000 or row.get('natLevName') in ['Municipiu', 'Oras']):
        centroid = row['geometry'].centroid
        # Determinam culoarea textului bazat pe intensitatea valorii
        # Daca valoarea e mai mare de 30% din max, folosim alb, altfel albastru inchis
        text_color = 'white' if row['Grecocatolica'] > (max_value * 0.3) else '#065073'
        text_obj = ax.annotate(text=f"{row['name']}\n({int(row['Grecocatolica'])})",
                   xy=(centroid.x, centroid.y),
                   ha='center',
                   fontsize=7 if text_color == 'white' else 6,
                   weight='bold' if text_color == 'white' else 'normal',
                   color=text_color)
        # Adaugam contur negru pentru textul alb
        if text_color == 'white':
            text_obj.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])

ax.axis('off')
ax.set_title('Greco-Catolici pe Romania (2021)', 
            fontsize=20, 
            fontweight='bold',
            pad=20)

plt.tight_layout()
plt.savefig('harta_grecocatolica_matplotlib.png', dpi=300, bbox_inches='tight')
print("\n✓ Harta salvata ca 'harta_grecocatolica_matplotlib.png'")
plt.show()
